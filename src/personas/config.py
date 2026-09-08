"""Centralised configuration for the diabetes stratification pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All hyperparameters, paths, and feature schema in one frozen config."""
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")
    figures_dir: Path = Path("outputs/figures")
    models_dir: Path = Path("outputs/models")
    evaluation_dir: Path = Path("outputs/evaluation")
    null_threshold: float = 0.4
    unknown_gender: str = "Unknown/Invalid"
    id_columns: tuple[str, ...] = ("encounter_id", "patient_nbr")
    target_column: str = "readmitted"
    age_order: tuple[str, ...] = ("[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)")
    med_dose_columns: tuple[str, ...] = ("metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone")
    med_levels: dict = {"dose": ["No", "Down", "Steady", "Up"], "change": ["No", "Ch"], "diabetesMed": ["No", "Yes"]}
    nominal_columns: tuple[str, ...] = ("race", "gender", "payer_code", "admission_type_id", "discharge_disposition_id", "admission_source_id", "diag_1", "diag_2", "diag_3")
    numeric_columns: tuple[str, ...] = ("time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses")
    latent_dim: int = 8
    hidden_dims: tuple[int, int] = (32, 16)
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    val_split: float = 0.2
    early_stopping_patience: int = 10
    n_clusters: int = 8
    n_init: int = 10
    n_twins: int = 5
    twin_metric: str = "cosine"
    umap_neighbors: int = 15
    umap_min_dist: float = 0.05
    umap_metric: str = "euclidean"
    map_sample_size: int = 5000
    plotly_template: str = "plotly_dark"
    seed: int = 42
    model_config = {"env_prefix": "DSL_", "env_file": ".env", "extra": "ignore"}


settings = Settings()