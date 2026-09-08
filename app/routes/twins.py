"""Route: GET /patient/{idx}/twins"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import TwinsResponse
from app.state import AppState

router = APIRouter()


@router.get("/patient/{patient_idx}/twins", response_model=TwinsResponse)
def get_patient_twins(patient_idx: int, state: AppState = None) -> dict:
    n = len(state.df_latent)
    if patient_idx < 0 or patient_idx >= n:
        raise HTTPException(status_code=404, detail="Patient index out of range")
    dists, indices = state.twin_index.query(patient_idx, k=state.artifacts.settings.n_twins)
    twins = []
    for idx in indices:
        row = state.df_base.iloc[idx]
        twins.append({"age": str(row.get("age", "")), "diag_1": str(row.get("diag_1", "")), "time_in_hospital": int(row.get("time_in_hospital", 0)), "persona_id": int(row.get("persona_id", 0))})
    return {"target_index": patient_idx, "neural_twins": twins, "distances": dists.tolist()}