"""Tests for ICD-9 mapping and FeatureEngineer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from personas.features import FeatureEngineer, map_icd9


class TestMapICD9:
    def test_diabetes_code(self):
        assert map_icd9("250.00") == "Diabetes"
        assert map_icd9("250.10") == "Diabetes"
        assert map_icd9("250.83") == "Diabetes"

    def test_circulatory_range(self):
        assert map_icd9("428.0") == "Circulatory"
        assert map_icd9("390") == "Circulatory"
        assert map_icd9("459") == "Circulatory"
        assert map_icd9("785") == "Circulatory"

    def test_respiratory(self):
        assert map_icd9("496") == "Respiratory"
        assert map_icd9("786") == "Respiratory"

    def test_digestive(self):
        assert map_icd9("530") == "Digestive"
        assert map_icd9("787") == "Digestive"

    def test_urogenital(self):
        assert map_icd9("585") == "Urogenital"
        assert map_icd9("788") == "Urogenital"

    def test_neoplasms(self):
        assert map_icd9("140") == "Neoplasms"
        assert map_icd9("239") == "Neoplasms"

    def test_musculoskeletal(self):
        assert map_icd9("710") == "Musculoskeletal"
        assert map_icd9("739") == "Musculoskeletal"

    def test_injury(self):
        assert map_icd9("800") == "Injury"
        assert map_icd9("999") == "Injury"

    def test_v_prefix(self):
        assert map_icd9("V58.61") == "Other"
        assert map_icd9("V10") == "Other"

    def test_e_prefix(self):
        assert map_icd9("E11") == "Other"
        assert map_icd9("E950") == "Other"

    def test_nan_returns_other(self):
        assert map_icd9(np.nan) == "Other"
        assert map_icd9(None) == "Other"

    def test_unknown_code_returns_other(self):
        assert map_icd9("9999") == "Other"
        assert map_icd9("abc") == "Other"


class TestFeatureEngineer:
    def test_fit_produces_ordinal_maps(self, sample_df, sample_settings):
        fe = FeatureEngineer(sample_settings)
        fe.fit(sample_df)
        assert "age" in fe._ordinal_maps
        assert fe._ordinal_maps["age"]["[0-10)"] == 0
        assert fe._ordinal_maps["age"]["[90-100)"] == 9

    def test_fit_produces_nominal_encoders(self, sample_df, sample_settings):
        fe = FeatureEngineer(sample_settings)
        fe.fit(sample_df)
        assert "race" in fe._nominal_encoders
        assert "diag_1" in fe._nominal_encoders

    def test_no_target_leakage(self, sample_df, sample_settings):
        fe = FeatureEngineer(sample_settings)
        result = fe.fit_transform(sample_df)
        assert not {"readmitted", "encounter_id", "patient_nbr"} & set(result.columns)

    def test_gender_filtered(self, sample_df, sample_settings):
        fe = FeatureEngineer(sample_settings)
        result = fe.fit_transform(sample_df)
        assert len(result) <= len(sample_df)

    def test_output_is_numeric(self, sample_df, sample_settings):
        fe = FeatureEngineer(sample_settings)
        result = fe.fit_transform(sample_df)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"

    def test_save_load_roundtrip(self, sample_df, sample_settings, tmp_path):
        fe = FeatureEngineer(sample_settings)
        result1 = fe.fit_transform(sample_df)
        save_path = tmp_path / "engineer.joblib"
        fe.save(save_path)
        fe2 = FeatureEngineer.load(save_path)
        result2 = fe2.transform(sample_df)
        pd.testing.assert_frame_equal(result1, result2)