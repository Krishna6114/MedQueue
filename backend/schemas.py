"""
Pydantic v2 Request & Response Schemas for MedQueue REST API.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TokenAttributionSchema(BaseModel):
    token: str
    tfidf_weight: float
    model_coefficient: float
    contribution: float


class PatientIntakeRequest(BaseModel):
    patient_name: str = Field(default="Patient", min_length=1, max_length=120)
    patient_age: Optional[int] = Field(default=None, ge=0, le=120)
    symptom_text: str = Field(..., min_length=3, max_length=2000, description="Free-text symptom description")


class CaseSubmissionResponse(BaseModel):
    case_id: str
    patient_name: str
    patient_age: Optional[int]
    symptom_text: str
    urgency_score: float
    confidence: float
    explanation_summary: str
    top_contributors: List[TokenAttributionSchema]
    arrival_minute: float
    current_rank_position: int
    total_waiting: int
    disclaimer: str = (
        "Non-diagnostic triage notice: MedQueue estimates urgency solely to prioritize consultation order. "
        "A licensed clinician reviews every case."
    )


class TermBreakdownSchema(BaseModel):
    urgency_term: float
    wait_term: float
    confidence_term: float


class RankedQueueItemSchema(BaseModel):
    rank_position: int
    case_id: str
    patient_name: str
    patient_age: Optional[int]
    symptom_text: str
    urgency_score: float
    confidence: float
    arrival_minute: float
    elapsed_wait: float
    composite_score: float
    term_breakdown: TermBreakdownSchema
    explanation_summary: str
    top_contributors: List[TokenAttributionSchema]
    true_severity: Optional[float] = None


class WeightVectorSchema(BaseModel):
    w1: float
    w2: float
    w3: float


class QueueStateResponse(BaseModel):
    current_queue_minute: float
    waiting_count: int
    in_consultation_case: Optional[RankedQueueItemSchema]
    waiting_queue: List[RankedQueueItemSchema]
    weights: WeightVectorSchema
    recalibration_mode: str
    window_size: int
    pending_feedback_count: int
    completed_recalibration_cycles: int


class StartConsultationRequest(BaseModel):
    override_case_id: Optional[str] = Field(
        default=None,
        description="Optional case_id if clinician overrides top-ranked queue order.",
    )


class SubmitFeedbackRequest(BaseModel):
    case_id: str
    feedback_label: Literal["correctly_prioritized", "over_flagged", "under_flagged"]
    clinician_note: str = Field(default="", max_length=1000)


class RecalibrationEventSchema(BaseModel):
    cycle_index: int
    window_size: int
    mode: str
    p_correct: float
    p_over: float
    p_under: float
    p_over_wait: float
    p_over_score: float
    raw_delta_w1: float
    applied_delta_w1: float
    previous_weights: WeightVectorSchema
    updated_weights: WeightVectorSchema
    timestamp: float


class FeedbackSubmissionResponse(BaseModel):
    case_id: str
    feedback_label: str
    over_flag_cause: str
    recalibration_triggered: bool
    recalibration_event: Optional[RecalibrationEventSchema]
    current_weights: WeightVectorSchema
    pending_feedback_count: int
    window_size: int


class DemoSeedRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=50)
    stagger_minutes: float = Field(default=2.0, ge=0.0, le=30.0)


class AdvanceTimeRequest(BaseModel):
    minutes: float = Field(default=15.0, gt=0.0, le=300.0)


class ConfigUpdateRequest(BaseModel):
    recalibration_mode: Literal["fixed", "baseline_paper", "root_cause_aware"]
    window_size: int = Field(default=20, ge=2, le=100)
    delta: float = Field(default=0.08, gt=0.0, le=1.0)