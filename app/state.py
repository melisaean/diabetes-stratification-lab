"""Typed application state — loaded once at startup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from personas.artifacts import ModelArtifacts
from personas.clustering import TwinIndex


@dataclass
class AppState:
    df_base: pd.DataFrame
    df_latent: pd.DataFrame
    projection: np.ndarray
    twin_index: TwinIndex
    artifacts: ModelArtifacts
    persona_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    map_figure: dict = field(default_factory=dict)