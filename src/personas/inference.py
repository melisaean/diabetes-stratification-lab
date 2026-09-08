"""Encode a new patient record into a persona with neural twins."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from personas.clustering import PatientClustering, TwinIndex
from personas.features import FeatureEngineer
from personas.models import MedicalAutoencoder


def encode_record(record: dict[str, Any], engineer: FeatureEngineer, encoder: MedicalAutoencoder, clusterer: PatientClustering, twin_index: TwinIndex, df_base: pd.DataFrame, persona_stats: dict[int, dict[str, Any]], device: str = "cpu") -> dict[str, Any]:
    import torch
    df_raw = pd.DataFrame([record])
    X = engineer.transform(df_raw)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    encoder.eval()
    with torch.no_grad():
        _, latent = encoder(X_tensor)
    latent_np = latent.cpu().numpy()
    persona_id = int(clusterer.predict(latent_np)[0])
    full_latent = np.vstack([twin_index._latent, latent_np])
    new_idx = len(full_latent) - 1
    temp_index = TwinIndex.from_fingerprints(full_latent, n_neighbors=twin_index.n_neighbors)
    dists, indices = temp_index.query(new_idx, k=5)
    twins_data = df_base.iloc[indices][["age", "diag_1", "time_in_hospital", "persona_id"]].to_dict(orient="records")
    stats = persona_stats.get(persona_id, {})
    return {"persona_id": persona_id, "readmission_risk": stats.get("readmission_risk", 0.0), "averages": stats.get("averages", {}), "neural_twins": twins_data, "distances": dists.tolist()}