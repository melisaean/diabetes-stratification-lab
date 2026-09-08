"""Dimensionality reduction wrappers for PCA and UMAP."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


class DimensionalityReducer:
    """Thin wrappers around PCA and UMAP that store fitted models."""

    def __init__(self, n_components: int = 2, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.pca_model: PCA | None = None
        self.umap_model: UMAP | None = None

    def apply_pca(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.pca_model = PCA(n_components=self.n_components, random_state=self.random_state)
        projection = self.pca_model.fit_transform(np.nan_to_num(data))
        return projection, self.pca_model.explained_variance_ratio_

    def apply_umap(self, data: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.05, metric: str = "euclidean") -> np.ndarray:
        self.umap_model = UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state,
        )
        return self.umap_model.fit_transform(np.nan_to_num(data))