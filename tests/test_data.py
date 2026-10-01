"""Tests for DataLoader helpers and base-frame alignment."""

from __future__ import annotations

import pandas as pd

from personas.config import Settings
from personas.data import align_base_frame


def test_align_base_frame_drops_unknown_gender_anywhere():
    settings = Settings()
    df = pd.DataFrame(
        {"gender": ["Male", "Unknown/Invalid", "Female", "Male"], "readmitted": ["NO", "<30", "NO", ">30"]}
    )
    aligned = align_base_frame(df, settings)
    assert list(aligned["gender"]) == ["Male", "Female", "Male"]
    assert aligned.index.tolist() == [0, 1, 2]


def test_transform_and_align_same_row_count(sample_settings, sample_df):
    from personas.features import FeatureEngineer

    engineer = FeatureEngineer(sample_settings).fit(sample_df)
    X = engineer.transform(sample_df)
    aligned = align_base_frame(sample_df, sample_settings)
    assert len(X) == len(aligned)
