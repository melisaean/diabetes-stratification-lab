"""Route: GET /persona/{id}"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.schemas import PersonaSummary
from app.state import AppState

router = APIRouter()


@router.get("/persona/{persona_id}", response_model=PersonaSummary)
def get_persona_summary(persona_id: int, state: AppState = None) -> dict:
    stats = state.persona_stats.get(persona_id)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Persona {persona_id} not found")
    return {"persona_id": persona_id, "n_patients": stats["n_patients"], "averages": stats["averages"], "readmission_risk": stats["readmission_risk"]}