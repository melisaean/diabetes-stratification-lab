"""Tests for PatientClustering and TwinIndex."""

from __future__ import annotations

import numpy as np
import pytest

from personas.clustering import PatientClustering, TwinIndex


class TestPatientClustering:
    def test_fit_predict_shape(self, tiny_latent):
        pc = PatientClustering(n_clusters=3, random_state=42)
        labels = pc.fit_predict(tiny_latent)
        assert labels.shape == (30,)
        assert set(labels) <= {0, 1, 2}

    def test_predict_after_fit(self, tiny_latent):
        pc = PatientClustering(n_clusters=3, random_state=42)
        pc.fit_predict(tiny_latent)
        preds = pc.predict(tiny_latent[:5])
        assert preds.shape == (5,)

    def test_deterministic(self, tiny_latent):
        pc1 = PatientClustering(n_clusters=3, random_state=42)
        pc2 = PatientClustering(n_clusters=3, random_state=42)
        np.testing.assert_array_equal(pc1.fit_predict(tiny_latent), pc2.fit_predict(tiny_latent))


class TestTwinIndex:
    def test_query_excludes_self(self, tiny_latent):
        idx = TwinIndex.from_fingerprints(tiny_latent, n_neighbors=5)
        dists, indices = idx.query(0, k=5)
        assert 0 not in indices
        assert len(dists) == 5
        assert len(indices) == 5

    def test_distances_are_non_negative(self, tiny_latent):
        idx = TwinIndex.from_fingerprints(tiny_latent, n_neighbors=5)
        dists, _ = idx.query(0, k=5)
        assert np.all(dists >= 0)

    def test_raises_if_k_exceeds_n_neighbors(self, tiny_latent):
        idx = TwinIndex.from_fingerprints(tiny_latent, n_neighbors=3)
        with pytest.raises(ValueError, match="exceeds"):
            idx.query(0, k=5)

    def test_same_indices_across_calls(self, tiny_latent):
        idx = TwinIndex.from_fingerprints(tiny_latent, n_neighbors=5)
        d1, i1 = idx.query(5, k=3)
        d2, i2 = idx.query(5, k=3)
        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_array_almost_equal(d1, d2)