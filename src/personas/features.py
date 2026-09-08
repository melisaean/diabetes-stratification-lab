"""ICD-9 mapping and semantic feature engineering for diabetic clinical records."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from personas.config import Settings


# ── ICD-9 mapping ──────────────────────────────────────────────────────────


def map_icd9(code: object) -> str:
    """Collapse a granular ICD-9 code to one of 9 clinical categories."""
    if pd.isnull(code):
        return "Other"

    val = str(code).strip()

    if val.startswith(("V", "E")):
        return "Other"

    if val.startswith("250"):
        return "Diabetes"

    try:
        num = float(val)
    except ValueError:
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if 580 <= num <= 629 or num == 788:
        return "Urogenital"
    if 140 <= num <= 239:
        return "Neoplasms"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 800 <= num <= 999:
        return "Injury"

    return "Other"


# ── FeatureEngineer ────────────────────────────────────────────────────────


class FeatureEngineer:
    """End-to-end clinical feature encoder with fitted state persistence."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.feature_columns: List[str] = []
        self._ordinal_maps: Dict[str, Dict[str, int]] = {}
        self._nominal_encoders: Dict[str, Dict[str, int]] = {}
        self._one_hot_columns: List[str] = []
        self._scaler: MinMaxScaler = MinMaxScaler()

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        df = df.copy()
        df = self._filter_gender(df)
        df = self._drop_ids_and_target(df)
        df = self._map_icd9_columns(df)

        self._fit_ordinal("age", self.s.age_order, df)
        for col in self.s.med_dose_columns:
            if col in df.columns:
                self._fit_ordinal(col, self.s.med_levels["dose"], df)
        if "change" in df.columns:
            self._fit_ordinal("change", self.s.med_levels["change"], df)
        if "diabetesMed" in df.columns:
            self._fit_ordinal("diabetesMed", self.s.med_levels["diabetesMed"], df)

        for col in self.s.nominal_columns:
            if col in df.columns:
                self._fit_frequency(col, df)

        self.feature_columns = (
            list(self.s.numeric_columns)
            + list(self._ordinal_maps.keys())
            + list(self._nominal_encoders.keys())
        )
        self.feature_columns = [c for c in self.feature_columns if c in df.columns]

        num_cols = [c for c in self.s.numeric_columns if c in df.columns]
        X_num = df[num_cols].values.astype(float)
        self._scaler.fit(X_num)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._filter_gender(df)
        df = self._drop_ids_and_target(df)
        df = self._map_icd9_columns(df)

        for col, mapping in self._ordinal_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        for col, mapping in self._nominal_encoders.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        num_cols = [c for c in self.s.numeric_columns if c in df.columns]
        X_num = df[num_cols].values.astype(float)
        df[num_cols] = self._scaler.transform(X_num)

        result = df[[c for c in self.feature_columns if c in df.columns]].copy()
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_columns": self.feature_columns,
            "ordinal_maps": self._ordinal_maps,
            "nominal_encoders": self._nominal_encoders,
            "scaler": self._scaler,
            "one_hot_columns": self._one_hot_columns,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> FeatureEngineer:
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        from personas.config import settings as default_settings
        obj = cls(default_settings)
        obj.feature_columns = state["feature_columns"]
        obj._ordinal_maps = state["ordinal_maps"]
        obj._nominal_encoders = state["nominal_encoders"]
        obj._scaler = state["scaler"]
        obj._one_hot_columns = state["one_hot_columns"]
        return obj

    def _filter_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        if "gender" in df.columns:
            mask = df["gender"] != self.s.unknown_gender
            return df[mask].reset_index(drop=True)
        return df

    def _drop_ids_and_target(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in (*self.s.id_columns, self.s.target_column) if c in df.columns]
        return df.drop(columns=cols)

    def _map_icd9_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("diag_1", "diag_2", "diag_3"):
            if col in df.columns:
                df[col] = df[col].apply(map_icd9)
        return df

    def _fit_ordinal(self, col: str, order: tuple | list, df: pd.DataFrame) -> None:
        mapping = {v: i for i, v in enumerate(order)}
        self._ordinal_maps[col] = mapping

    def _fit_frequency(self, col: str, df: pd.DataFrame) -> None:
        freq = df[col].value_counts(normalize=True).to_dict()
        self._nominal_encoders[col] = freq