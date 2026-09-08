"""Pydantic response and request models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class PersonaSummary(BaseModel):
    persona_id: int
    n_patients: int
    averages: dict[str, float]
    readmission_risk: float


class TwinResult(BaseModel):
    age: str
    diag_1: str
    time_in_hospital: int
    persona_id: int


class TwinsResponse(BaseModel):
    target_index: int
    neural_twins: list[TwinResult]
    distances: list[float]


class EncodeRequest(BaseModel):
    model_config = {"extra": "allow"}
    encounter_id: int | None = None
    patient_nbr: int | None = None
    race: str = "Caucasian"
    gender: str = "Female"
    age: str = "[50-60)"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 1
    time_in_hospital: int = 3
    payer_code: str = "MC"
    num_lab_procedures: int = 40
    num_procedures: int = 1
    num_medications: int = 10
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0
    diag_1: str = "Circulatory"
    diag_2: str = "Diabetes"
    diag_3: str = "Other"
    number_diagnoses: int = 5
    metformin: str = "No"
    insulin: str = "No"
    change: str = "No"
    diabetesMed: str = "No"


class EncodeResponse(BaseModel):
    persona_id: int
    readmission_risk: float
    averages: dict[str, float]
    neural_twins: list[TwinResult]
    distances: list[float]