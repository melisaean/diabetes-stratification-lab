"""Evaluation: reconstruction, cluster quality, downstream benchmark, persona profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from personas.models import MedicalAutoencoder


def reconstruction_metrics(model: MedicalAutoencoder, X_train: np.ndarray, X_test: np.ndarray, pca_n_components: int = 8) -> dict:
    import torch
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        recon, _ = model(X_t)
    ae_mse = float(((recon.numpy() - X_test) ** 2).mean())
    pca = PCA(n_components=pca_n_components)
    pca.fit(X_train)
    X_pca = pca.inverse_transform(pca.transform(X_test))
    pca_mse = float(((X_pca - X_test) ** 2).mean())
    return {"ae_test_mse": ae_mse, "pca_test_mse": pca_mse, "improvement_pct": (pca_mse - ae_mse) / pca_mse * 100}


def cluster_sweep(latent: np.ndarray, k_range: range = range(4, 13)) -> pd.DataFrame:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(latent)
        sil = silhouette_score(latent, labels)
        db = davies_bouldin_score(latent, labels)
        ch = calinski_harabasz_score(latent, labels)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})
    return pd.DataFrame(rows)


def bootstrap_stability(latent: np.ndarray, labels: np.ndarray, n_boot: int = 50, seed: int = 42) -> float:
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(len(latent), size=len(latent), replace=True)
        km = KMeans(n_clusters=len(set(labels)), random_state=42, n_init=5)
        new_labels = km.fit_predict(latent[idx])
        aris.append(adjusted_rand_score(labels[idx], new_labels))
    return float(np.mean(aris))


def downstream_scores(X_scaled: np.ndarray, fingerprints: np.ndarray, pca_components: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold
    datasets = {"scaled_features": X_scaled, "ae_fingerprints": fingerprints, "pca_components": pca_components}
    classifiers = {"LogisticRegression": LogisticRegression(max_iter=500, random_state=42), "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)}
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for ds_name, X in datasets.items():
        for clf_name, clf in classifiers.items():
            auroc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
            ap_scores = cross_val_score(clf, X, y, cv=cv, scoring="average_precision")
            rows.append({"dataset": ds_name, "classifier": clf_name, "auroc_mean": auroc_scores.mean(), "auroc_std": auroc_scores.std(), "ap_mean": ap_scores.mean(), "ap_std": ap_scores.std()})
    return pd.DataFrame(rows)


def persona_profiles(df_base: pd.DataFrame, metric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    from scipy import stats
    rows = []
    for persona_id in sorted(df_base["persona_id"].unique()):
        persona = df_base[df_base["persona_id"] == persona_id]
        rest = df_base[df_base["persona_id"] != persona_id]
        n = len(persona)
        for col in metric_cols:
            if col in persona.columns:
                m_p = persona[col].mean()
                m_c = df_base[col].mean()
                s_c = df_base[col].std()
                cohens_d = (m_p - m_c) / s_c if s_c > 0 else 0.0
                t_stat, p_val = stats.ttest_ind(persona[col].dropna(), rest[col].dropna())
                rows.append({"persona_id": persona_id, "feature": col, "persona_mean": m_p, "cohort_mean": m_c, "cohens_d": cohens_d, "p_value": p_val, "n": n})
    return pd.DataFrame(rows)