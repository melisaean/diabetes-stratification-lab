"""Route: POST /encode — new-patient inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter

from app.schemas import EncodeRequest, EncodeResponse
from app.state import AppState
from personas.clustering import TwinIndex

router = APIRouter()


@router.post("/encode", response_model=EncodeResponse)
def encode_patient(record: EncodeRequest, state: AppState = None) -> dict:
    data = record.model_dump(exclude_none=True)
    df_raw = pd.DataFrame([data])
    X = state.artifacts.engineer.transform(df_raw)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    state.artifacts.encoder.eval()
    with torch.no_grad():
        _, latent = state.artifacts.encoder(X_tensor)
    latent_np = latent.numpy()
    persona_id = int(state.artifacts.clusterer.predict(latent_np)[0])
    full_latent = np.vstack([state.df_latent.values, latent_np])
    new_idx = len(full_latent) - 1
    temp_index = TwinIndex.from_fingerprints(full_latent, n_neighbors=state.twin_index.n_neighbors)
    dists, indices = temp_index.query(new_idx, k=state.artifacts.settings.n_twins)
    twins = []
    for idx in indices:
        row = state.df_base.iloc[idx]
        twins.append({"age": str(row.get("age", "")), "diag_1": str(row.get("diag_1", "")), "time_in_hospital": int(row.get("time_in_hospital", 0)), "persona_id": int(row.get("persona_id", 0))})
    stats = state.persona_stats.get(persona_id, {})
    return {"persona_id": persona_id, "readmission_risk": stats.get("readmission_risk", 0.0), "averages": stats.get("averages", {}), "neural_twins": twins, "distances": dists.tolist()}