"""Route: GET /map — cached Plotly figure JSON."""

from __future__ import annotations

import json

import plotly.express as px
from fastapi import APIRouter

from app.state import AppState

router = APIRouter()


def build_map_figure(state: AppState) -> dict:
    plot_df = state.df_base[["age", "gender", "diag_1", "time_in_hospital", "persona_id"]].copy()
    plot_df["UMAP_1"] = state.projection[:, 0]
    plot_df["UMAP_2"] = state.projection[:, 1]
    plot_df["index"] = plot_df.index
    plot_df["persona_id"] = plot_df["persona_id"].astype(str)
    if len(plot_df) > state.artifacts.settings.map_sample_size:
        plot_df = plot_df.sample(n=state.artifacts.settings.map_sample_size, random_state=42)
    fig = px.scatter(plot_df, x="UMAP_1", y="UMAP_2", color="persona_id",
                     hover_data=["age", "diag_1", "time_in_hospital", "index"],
                     title="Patient Persona Map",
                     template=state.artifacts.settings.plotly_template)
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return json.loads(fig.to_json())


@router.get("/map")
def get_map(state: AppState = None) -> dict:
    return state.map_figure