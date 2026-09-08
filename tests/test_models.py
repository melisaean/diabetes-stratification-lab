"""Tests for MedicalAutoencoder and train_autoencoder."""

from __future__ import annotations

import numpy as np
import torch

from personas.models import MedicalAutoencoder, train_autoencoder


class TestMedicalAutoencoder:
    def test_forward_shapes(self):
        model = MedicalAutoencoder(input_dim=43, latent_dim=8)
        x = torch.randn(10, 43)
        recon, latent = model(x)
        assert recon.shape == (10, 43)
        assert latent.shape == (10, 8)

    def test_fingerprints_shape(self):
        model = MedicalAutoencoder(input_dim=43, latent_dim=8)
        x = torch.randn(10, 43)
        fp = model.fingerprints(x)
        assert fp.shape == (10, 8)
        assert isinstance(fp, np.ndarray)

    def test_deterministic_fingerprints(self):
        model = MedicalAutoencoder(input_dim=43, latent_dim=8)
        x = torch.randn(10, 43)
        fp1 = model.fingerprints(x)
        fp2 = model.fingerprints(x)
        np.testing.assert_array_equal(fp1, fp2)

    def test_save_load_roundtrip(self, tmp_path):
        model = MedicalAutoencoder(input_dim=43, latent_dim=8)
        save_path = tmp_path / "model.pt"
        model.save(save_path)
        loaded = MedicalAutoencoder.load(save_path, input_dim=43, latent_dim=8)
        x = torch.randn(5, 43)
        np.testing.assert_array_equal(model.fingerprints(x), loaded.fingerprints(x))

    def test_custom_hidden_dims(self):
        model = MedicalAutoencoder(input_dim=10, latent_dim=4, hidden=(64, 32))
        x = torch.randn(5, 10)
        recon, latent = model(x)
        assert recon.shape == (5, 10)
        assert latent.shape == (5, 4)


class TestTrainAutoencoder:
    def test_training_decreases_loss(self, sample_settings):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 43)).astype(np.float32)
        model, history = train_autoencoder(X[:160], X[160:], sample_settings, device="cpu")
        assert history["val_loss"][-1] <= history["val_loss"][0]

    def test_model_is_returned(self, sample_settings):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 43)).astype(np.float32)
        model, history = train_autoencoder(X, X, sample_settings, device="cpu")
        assert isinstance(model, MedicalAutoencoder)
        assert "train_loss" in history
        assert "val_loss" in history