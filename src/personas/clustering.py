"""K-Means clustering and prebuilt nearest-neighbour twin index."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class PatientClustering:
    """K-Means wrapper for patient persona assignment."""

    def __init__(self, n_clusters: int, random_state: int, n_init: int = 10) -> None:
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    def fit_predict(self, latent: np.ndarray) -> np.ndarray:
        return self.kmeans.fit_predict(latent)

    def predict(self, latent: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(latent)


class TwinIndex:
    """Prebuilt nearest-neighbour index for the twin-retrieval feature."""

    def __init__(self, latent: np.ndarray, n_neighbors: int, metric: str = "cosine") -> None:
        self.n_neighbors = n_neighbors
        self._nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
        self._nn.fit(latent)
        self._latent = latent

    def query(self, index: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k > self.n_neighbors:
            raise ValueError(f"k={k} exceeds configured n_neighbors={self.n_neighbors}")
        target = self._latent[[index]]
        distances, indices = self._nn.kneighbors(target, n_neighbors=k + 1)
        return distances[0][1:], indices[0][1:]

    @classmethod
    def from_fingerprints(cls, latent: np.ndarray, n_neighbors: int = 5, metric: str = "cosine") -> TwinIndex:
        return cls(latent=latent, n_neighbors=n_neighbors, metric=metric)