"""Data ingestion and initial cleaning for the UCI Diabetes 130-US Hospitals dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from personas.config import Settings


class DataLoader:
    """Load raw CSVs, apply null handling, decode admission mappings."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.df: pd.DataFrame | None = None
        self.mappings: dict[str, pd.DataFrame] = {}

    def load_data(self) -> pd.DataFrame:
        path = self.s.raw_dir / "diabetic_data.csv"
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found at {path} — run `make data` first")
        self.df = pd.read_csv(path, low_memory=False)
        self.df.replace("?", np.nan, inplace=True)
        return self.df

    def load_mappings(self) -> dict[str, pd.DataFrame]:
        path = self.s.raw_dir / "IDS_mapping.csv"
        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found at {path}")
        all_mappings = pd.read_csv(path)
        self.mappings["admission_type"] = all_mappings.iloc[0:8, :]
        self.mappings["discharge_disposition"] = all_mappings.iloc[10:40, :]
        self.mappings["admission_source"] = all_mappings.iloc[42:67, :]
        return self.mappings

    def drop_high_null_columns(self, threshold: float | None = None) -> list[str]:
        if self.df is None:
            raise ValueError("Call load_data() first")
        threshold = threshold or self.s.null_threshold
        null_pct = self.df.isnull().mean()
        to_drop = null_pct[null_pct > threshold].index.tolist()
        self.df.drop(columns=to_drop, inplace=True)
        return to_drop

    def save_to_processed(self, output_path: Path) -> None:
        if self.df is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(output_path, index=False)