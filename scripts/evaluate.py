"""CLI entry point for evaluation (reconstruction, cluster sweep, downstream)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from personas.config import Settings
from personas.artifacts import load_artifacts
from personas.evaluation import (
    bootstrap_stability,
    cluster_sweep,
    downstream_scores,
    persona_profiles,
    reconstruction_metrics,
)


def main() -> None:
    settings = Settings()
    proc = settings.processed_dir
    models_dir = settings.outputs_dir / "models"
    eval_dir = settings.evaluation_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load artefacts
    artifacts = load_artifacts(models_dir, settings)

    # Load data
    X_df = pd.read_csv(proc / "diabetic_features_scaled.csv")
    X_np = X_df.values.astype(np.float32)
    df_base = pd.read_csv(proc / "diabetic_cleaned_base.csv", low_memory=False)
    latent = np.array(pd.read_csv(proc / "patient_fingerprints.csv"))

    # Train/test split (same seed as pipeline)
    X_train, X_test = train_test_split(X_np, test_size=0.2, random_state=settings.seed)

    # 1. Reconstruction metrics
    rec = reconstruction_metrics(artifacts.encoder, X_train, X_test)
    print(f"Reconstruction: AE MSE={rec['ae_test_mse']:.6f}, PCA MSE={rec['pca_test_mse']:.6f}")

    # 2. Cluster sweep
    sweep = cluster_sweep(latent)
    best_k = sweep.loc[sweep["silhouette"].idxmax(), "k"]
    print(f"Best k by silhouette: {best_k}")

    # 3. Bootstrap stability
    stability = bootstrap_stability(latent, df_base["persona_id"].values)
    print(f"Bootstrap stability (mean ARI): {stability:.3f}")

    # 4. Downstream benchmark (if readmitted available)
    downstream = None
    if "readmitted" in df_base.columns:
        y_bin = (df_base["readmitted"] == "<30").astype(int).values
        pca = PCA(n_components=8, random_state=42)
        pca.fit(X_train)
        downstream = downstream_scores(X_np, latent, pca.transform(X_np), y_bin)
        print("Downstream readmission benchmark:")
        print(downstream.to_string(index=False))

    # 5. Persona profiles
    profiles = persona_profiles(
        df_base,
        metric_cols=["time_in_hospital", "num_medications", "num_lab_procedures", "number_diagnoses"],
        categorical_cols=["gender", "race"],
    )

    # Save all results
    results = {
        "reconstruction": rec,
        "cluster_sweep": sweep.to_dict(orient="records"),
        "bootstrap_stability": stability,
        "best_k": int(best_k),
    }
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    sweep.to_csv(eval_dir / "cluster_sweep.csv", index=False)
    profiles.to_csv(eval_dir / "persona_profiles.csv", index=False)
    if downstream is not None:
        downstream.to_csv(eval_dir / "downstream_benchmark.csv", index=False)

    print(f"Evaluation complete. Results in {eval_dir}")


if __name__ == "__main__":
    main()