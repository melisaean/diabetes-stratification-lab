"""Persist and reload all model + feature-engineer artefacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from personas.clustering import PatientClustering
from personas.config import Settings, settings
from personas.features import FeatureEngineer
from personas.models import MedicalAutoencoder


@dataclass
class ModelArtifacts:
    """Container for everything needed to serve predictions."""
    encoder: MedicalAutoencoder
    engineer: FeatureEngineer
    clusterer: PatientClustering
    settings: Settings
    feature_columns: list[str] = field(default_factory=list)
    input_dim: int = 0
    latent_dim: int = 8


def save_artifacts(artifacts: ModelArtifacts, models_dir: Path) -> None:
    """Persist artefacts to models_dir."""
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts.encoder.save(models_dir / "autoencoder.pt")
    artifacts.engineer.save(models_dir / "engineer.joblib")
    artifacts.clusterer.save(models_dir / "clusterer.joblib")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_dim": artifacts.input_dim,
        "latent_dim": artifacts.latent_dim,
        "feature_columns": artifacts.feature_columns,
        "config_hash": _hash_settings(artifacts.settings),
    }
    with open(models_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_artifacts(models_dir: Path, settings_override: Settings | None = None) -> ModelArtifacts:
    """Load persisted artefacts from models_dir."""
    s = settings_override or settings
    with open(models_dir / "manifest.json") as f:
        manifest = json.load(f)
    input_dim = manifest["input_dim"]
    latent_dim = manifest["latent_dim"]
    encoder = MedicalAutoencoder.load(
        models_dir / "autoencoder.pt",
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden=s.hidden_dims,
    )
    engineer = FeatureEngineer.load(models_dir / "engineer.joblib")
    clusterer = PatientClustering.load(models_dir / "clusterer.joblib")
    return ModelArtifacts(
        encoder=encoder,
        engineer=engineer,
        clusterer=clusterer,
        settings=s,
        feature_columns=manifest["feature_columns"],
        input_dim=input_dim,
        latent_dim=latent_dim,
    )


def _hash_settings(s: Settings) -> str:
    """Deterministic hash of the settings for the manifest."""
    raw = json.dumps(s.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]