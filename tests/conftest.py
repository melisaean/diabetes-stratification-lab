"""Shared test fixtures for the diabetes stratification test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_settings(tmp_path: Path):
    from personas.config import Settings
    return Settings(project_root=tmp_path, data_dir=tmp_path / "data", raw_dir=tmp_path / "data/raw", processed_dir=tmp_path / "data/processed", outputs_dir=tmp_path / "outputs", figures_dir=tmp_path / "outputs/figures", models_dir=tmp_path / "outputs/models", evaluation_dir=tmp_path / "outputs/evaluation", epochs=2, n_clusters=3, n_twins=3)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame({"encounter_id": range(1000, 1000 + n), "patient_nbr": rng.integers(1, 50, size=n), "race": rng.choice(["Caucasian", "AfricanAmerican", "Asian"], size=n), "gender": rng.choice(["Male", "Female", "Unknown/Invalid"], size=n), "age": rng.choice(["[0-10)", "[50-60)", "[70-80)", "[80-90)"], size=n), "admission_type_id": rng.integers(1, 7, size=n), "discharge_disposition_id": rng.integers(1, 30, size=n), "admission_source_id": rng.integers(1, 20, size=n), "time_in_hospital": rng.integers(1, 15, size=n), "payer_code": rng.choice(["MC", "HM", "SP", "?"], size=n), "num_lab_procedures": rng.integers(1, 100, size=n), "num_procedures": rng.integers(0, 7, size=n), "num_medications": rng.integers(1, 30, size=n), "number_outpatient": rng.integers(0, 10, size=n), "number_emergency": rng.integers(0, 5, size=n), "number_inpatient": rng.integers(0, 10, size=n), "diag_1": rng.choice(["250.00", "428.0", "496", "E11", "V58.61"], size=n), "diag_2": rng.choice(["250.00", "401.1", "272.4", "585", np.nan], size=n), "diag_3": rng.choice(["250.00", "250.10", "401.1", "272.4", np.nan], size=n), "number_diagnoses": rng.integers(1, 15, size=n), "metformin": rng.choice(["No", "Steady", "Up"], size=n), "insulin": rng.choice(["No", "Down", "Steady", "Up"], size=n), "change": rng.choice(["No", "Ch"], size=n), "diabetesMed": rng.choice(["No", "Yes"], size=n), "readmitted": rng.choice(["NO", ">30", "<30"], size=n)})


@pytest.fixture
def tiny_latent() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((30, 8)).astype(np.float32)