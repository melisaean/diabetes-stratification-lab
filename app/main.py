"""FastAPI application factory with lifespan-based artefact loading."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.routes import encode, map, persona, twins
from app.state import AppState
from personas.artifacts import load_artifacts
from personas.clustering import TwinIndex
from personas.config import Settings


def compute_persona_stats(df_base, settings) -> dict:
    stats = {}
    metrics = ["time_in_hospital", "num_medications", "num_lab_procedures", "number_diagnoses"]
    for pid in sorted(df_base["persona_id"].unique()):
        persona_data = df_base[df_base["persona_id"] == pid]
        averages = {m: round(float(persona_data[m].mean()), 2) for m in metrics if m in persona_data.columns}
        readmission_risk = 0.0
        if "readmitted" in persona_data.columns:
            readmission_risk = round(((persona_data["readmitted"] == "<30").sum() / len(persona_data)) * 100, 2)
        stats[int(pid)] = {"n_patients": len(persona_data), "averages": averages, "readmission_risk": readmission_risk}
    return stats


def create_app(settings=None) -> FastAPI:
    s = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        proc = s.processed_dir
        models_dir = s.outputs_dir / "models"
        artifacts = load_artifacts(models_dir, s)
        df_base = pd.read_csv(proc / "diabetic_cleaned_base.csv", low_memory=False)
        df_latent = pd.read_csv(proc / "patient_fingerprints.csv")
        projection = pd.read_csv(proc / "map_coords.csv").values
        if "Unnamed: 0" in df_latent.columns:
            df_latent = df_latent.drop(columns=["Unnamed: 0"])
        assert len(df_base) == len(df_latent) == len(projection)
        latent_np = df_latent.values
        twin_index = TwinIndex.from_fingerprints(latent_np, n_neighbors=s.n_twins, metric=s.twin_metric)
        persona_stats = compute_persona_stats(df_base, s)
        from app.routes.map import build_map_figure

        state = AppState(
            df_base=df_base,
            df_latent=df_latent,
            projection=projection,
            twin_index=twin_index,
            artifacts=artifacts,
            persona_stats=persona_stats,
        )
        state.map_figure = build_map_figure(state)
        app.state.state = state
        print(f"Startup complete. {len(df_base)} patients loaded.")
        yield

    app = FastAPI(title="Diabetes Stratification Lab", version="1.0.0", lifespan=lifespan)
    app.include_router(map.router)
    app.include_router(persona.router)
    app.include_router(twins.router)
    app.include_router(encode.router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ui")
    def get_ui():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return app


app = create_app()