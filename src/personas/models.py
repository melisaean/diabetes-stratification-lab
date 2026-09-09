"""Medical autoencoder and training loop with early stopping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from personas.config import Settings


class MedicalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden: tuple[int, int] = (32, 16)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, input_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    @torch.no_grad()
    def fingerprints(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        _, latent = self(x.to(device))
        return latent.cpu().numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, input_dim: int, latent_dim: int = 8, hidden: tuple[int, int] = (32, 16)) -> MedicalAutoencoder:
        model = cls(input_dim=input_dim, latent_dim=latent_dim, hidden=hidden)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, settings: Settings, device: str | None = None) -> tuple[MedicalAutoencoder, dict]:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dev = torch.device(device)
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    model = MedicalAutoencoder(input_dim=X_train.shape[1], latent_dim=settings.latent_dim, hidden=settings.hidden_dims).to(dev)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(dev)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(dev)
    history: dict = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0
    best_state = None
    for _epoch in range(settings.epochs):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, X_train_t.size(0), settings.batch_size):
            idx = perm[i : i + settings.batch_size]
            batch = X_train_t[idx]
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        model.eval()
        with torch.no_grad():
            val_recon, _ = model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= settings.early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history