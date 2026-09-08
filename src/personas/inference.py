"""Encode a new patient record into a persona with neural twins."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from personas.clustering import TwinIndex, PatientClustering
from personas.features import FeatureEngineer
from personas.models import MedicalAutoencoder


def encode_record(
    record: Dict[str, Any],
    engineer: FeatureEngineer,
    encoder: MedicalAutoencoder,
    clusterer: PatientClustering,
    twin_index: TwinIndex,
    df_base: pd.DataFrame,
    persona_stats: Dict[int, Dict[str, Any]],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Map a single patient record to its persona and neural twins.

    Parameters
    ----------
    record : dict
        Raw clinical record (same columns as the training CSV).
    engineer : FeatureEngineer
        Fitted feature engineer.
    encoder : MedicalAutoencoder
        Trained autoencoder.
    clusterer : PatientClustering
        Fitted KMeans.
    twin_index : TwinIndex
        Prebuilt twin index.
    df_base : pd.DataFrame
        Base DataFrame with encounter_id, age, diag_1, etc.
    persona_stats : dict
        Precomputed persona statistics (from app state).

    Returns
    -------
    dict with persona_id, readmission_risk, averages, neural_twins, distances.
    """
    import torch

    # Build a single-row DataFrame and transform
    df_raw = pd.DataFrame([record])
    X = engineer.transform(df_raw)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    # Fingerprint → persona
    import torch.nn as nn
    encoder.eval()
    with torch.no_grad():
        _, latent = encoder(X_tensor)
    latent_np = latent.cpu().numpy()

    persona_id = int(clusterer.predict(latent_np)[0])

    # Neural twins — find the index closest to this patient in the full dataset
    # For a new patient, we append to the latent matrix temporarily
    full_latent = np.vstack([twin_index._latent, latent_np])
    new_idx = len(full_latent) - 1
    temp_index = TwinIndex.from_fingerprints(full_latent, n_neighbors=twin_index.n_neighbors)
    dists, indices = temp_index.query(new_idx, k=5)

    twins_data = df_base.iloc[indices][["age", "diag_1", "time_in_hospital", "persona_id"]].to_dict(orient="records")

    stats = persona_stats.get(persona_id, {})

    return {
        "persona_id": persona_id,
        "readmission_risk": stats.get("readmission_risk", 0.0),
        "averages": stats.get("averages", {}),
        "neural_twins": twins_data,
        "distances": dists.tolist(),
    }