"""End-to-end pipeline: preprocess → train → cluster → UMAP → save artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from personas.artifacts import ModelArtifacts, save_artifacts
from personas.clustering import PatientClustering
from personas.config import Settings
from personas.data import DataLoader
from personas.dimensionality import DimensionalityReducer
from personas.features import FeatureEngineer
from personas.models import train_autoencoder


def run_pipeline(settings: Settings, device: str | None = None) -> dict[str, Path]:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc_dir = settings.processed_dir
    proc_dir.mkdir(parents=True, exist_ok=True)
    models_dir = settings.outputs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = settings.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(settings)
    df_raw = loader.load_data()
    dropped = loader.drop_high_null_columns()
    print(f"Dropped high-null columns: {dropped}")
    loader.save_to_processed(proc_dir / "diabetic_cleaned_base.csv")
    df_base = loader.df.copy()  # type: ignore[union-attr]
    engineer = FeatureEngineer(settings)
    engineer.fit(df_raw)
    X_full = engineer.transform(df_raw)
    feature_columns = list(X_full.columns)
    print(f"Feature matrix: {X_full.shape[0]} patients x {X_full.shape[1]} features")
    df_base = df_base.iloc[:len(X_full)].reset_index(drop=True)
    X_full.to_csv(proc_dir / "diabetic_features_scaled.csv", index=False)
    X_np = X_full.values.astype(np.float32)
    X_train, X_val = train_test_split(X_np, test_size=settings.val_split, random_state=settings.seed)
    print(f"Training on {len(X_train)}, validating on {len(X_val)}...")
    model, history = train_autoencoder(X_train, X_val, settings, device=device)
    print(f"Training complete. Final val loss: {history['val_loss'][-1]:.6f}")
    model.save(models_dir / "autoencoder.pt")
    model.eval()
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    latent_np = model.fingerprints(X_tensor)
    print(f"Fingerprints: {latent_np.shape}")
    df_latent = pd.DataFrame(latent_np, columns=[f"latent_{i}" for i in range(latent_np.shape[1])])
    df_latent.to_csv(proc_dir / "patient_fingerprints.csv", index=False)
    clusterer = PatientClustering(n_clusters=settings.n_clusters, random_state=settings.seed, n_init=settings.n_init)
    persona_labels = clusterer.fit_predict(latent_np)
    df_base["persona_id"] = persona_labels
    print(f"Persona distribution:\n{pd.Series(persona_labels).value_counts().sort_index().to_string()}")
    reducer = DimensionalityReducer(n_components=2, random_state=settings.seed)
    projection = reducer.apply_umap(latent_np, n_neighbors=settings.umap_neighbors, min_dist=settings.umap_min_dist, metric=settings.umap_metric)
    df_coords = pd.DataFrame(projection, columns=["x", "y"])
    df_coords.to_csv(proc_dir / "map_coords.csv", index=False)
    print(f"UMAP projection: {projection.shape}")
    df_base.to_csv(proc_dir / "diabetic_cleaned_base.csv", index=False)
    artifacts = ModelArtifacts(encoder=model, engineer=engineer, clusterer=clusterer, settings=settings, feature_columns=feature_columns, input_dim=X_np.shape[1], latent_dim=settings.latent_dim)
    save_artifacts(artifacts, models_dir)
    manifest_path = models_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["git_sha"] = _get_git_sha(settings.project_root)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Pipeline complete. Artefacts in {models_dir}")
    return {"base": proc_dir / "diabetic_cleaned_base.csv", "fingerprints": proc_dir / "patient_fingerprints.csv", "map_coords": proc_dir / "map_coords.csv", "features": proc_dir / "diabetic_features_scaled.csv", "model": models_dir / "autoencoder.pt", "engineer": models_dir / "engineer.joblib", "manifest": models_dir / "manifest.json"}


def _get_git_sha(project_root: Path) -> str | None:
    import subprocess
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=project_root, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None