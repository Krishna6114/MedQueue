"""
Unit & Integration Test Suite for MedQueue Core Algorithms and FastAPI Backend.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.service import medqueue_manager
from core.dataset import generate_synthetic_dataset
from core.queue_engine import AdaptivePriorityQueue, QueueCase
from core.recalibrator import FeedbackLabel, FeedbackRecord, RecalibrationEngine, WeightVector
from core.scorer import UrgencyScorer


def test_synthetic_dataset_stratification():
    df = generate_synthetic_dataset(n_cases=400, seed=42)
    assert len(df) == 400
    assert (df["urgency_tier"] == "High").sum() == 100
    assert (df["urgency_tier"] == "Moderate").sum() == 120
    assert (df["urgency_tier"] == "Low").sum() == 180
    assert df["true_severity"].min() >= 5.0
    assert df["true_severity"].max() <= 99.0


def test_urgency_scorer_and_token_xai():
    df = generate_synthetic_dataset(n_cases=400, seed=42)
    scorer = UrgencyScorer(random_state=42)
    scorer.fit(df["symptom_text"], df["true_severity"])

    res_high = scorer.score_case("crushing chest pain radiating to left arm and jaw with cold sweats")
    res_low = scorer.score_case("routine prescription refill request for stable blood pressure medication")

    assert 0.0 <= res_high.urgency_score <= 100.0
    assert 0.0 <= res_low.urgency_score <= 100.0
    assert res_high.urgency_score > res_low.urgency_score
    assert 0.0 <= res_high.confidence <= 1.0
    assert len(res_high.top_contributors) > 0


def test_starvation_prevention_overtaking():
    pq = AdaptivePriorityQueue(policy="medqueue")
    mild = QueueCase(
        case_id="MILD",
        symptom_text="mild sore throat",
        urgency_score=20.0,
        confidence=0.80,
        arrival_time=0.0,
        true_severity=20.0,
    )
    urgent = QueueCase(
        case_id="URGENT",
        symptom_text="severe chest pain",
        urgency_score=90.0,
        confidence=0.80,
        arrival_time=145.0,
        true_severity=90.0,
    )
    pq.enqueue(mild)
    pq.enqueue(urgent)

    # At t = 145 min, MILD has waited 145 min -> R(145) = 12 + 43.5 + 8 = 63.5 > URGENT (62.0)
    ranked = pq.get_ranked_queue(current_time=145.0)
    assert ranked[0].case.case_id == "MILD"
    assert np.isclose(ranked[0].composite_score, 63.5)


def test_recalibration_bounds_and_option_a_sign_fix():
    # Verify Option A increases w1 when over-flags are wait-term dominant
    recal_a = RecalibrationEngine(window_size=4, delta=0.20, delta_max=0.05, mode="root_cause_aware")
    for i in range(4):
        ev = recal_a.record_feedback(
            FeedbackRecord(
                case_id=f"C-{i}",
                label=FeedbackLabel.OVER_FLAGGED,
                over_flag_cause="wait_term_dominant",
            )
        )
    assert ev is not None
    assert ev.updated_weights.w1 > ev.previous_weights.w1
    assert np.isclose(ev.applied_delta_w1, 0.05)  # Capped at delta_max = 0.05
    assert np.isclose(
        ev.updated_weights.w1 + ev.updated_weights.w2 + ev.updated_weights.w3,
        1.0,
    )

    # Verify Baseline Paper Eq. 3 decreases w1 on the exact same over-flags
    recal_base = RecalibrationEngine(window_size=4, delta=0.20, delta_max=0.05, mode="baseline_paper")
    for i in range(4):
        ev_b = recal_base.record_feedback(
            FeedbackRecord(
                case_id=f"C-{i}",
                label=FeedbackLabel.OVER_FLAGGED,
                over_flag_cause="wait_term_dominant",
            )
        )
    assert ev_b is not None
    assert ev_b.updated_weights.w1 < ev_b.previous_weights.w1


def test_fastapi_end_to_end_workflow():
    client = TestClient(app)
    medqueue_manager.reset_system_state()

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    # Submit two patients
    r1 = client.post(
        "/api/intake",
        json={"patient_name": "Routine User", "patient_age": 30, "symptom_text": "routine prescription refill request"},
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/api/intake",
        json={"patient_name": "Urgent User", "patient_age": 60, "symptom_text": "crushing chest pain with cold sweats"},
    )
    assert r2.status_code == 200
    urgent_id = r2.json()["case_id"]

    # Urgent patient should rank #1 ahead of routine patient
    q_res = client.get("/api/queue").json()
    assert q_res["waiting_count"] == 2
    assert q_res["waiting_queue"][0]["case_id"] == urgent_id

    # Start consultation & submit clinician feedback
    start_res = client.post("/api/consultation/start", json={})
    assert start_res.status_code == 200
    assert start_res.json()["active_case"]["case_id"] == urgent_id

    fb_res = client.post(
        "/api/consultation/feedback",
        json={"case_id": urgent_id, "feedback_label": "correctly_prioritized", "clinician_note": "Appropriate priority"},
    )
    assert fb_res.status_code == 200
    assert fb_res.json()["pending_feedback_count"] == 1

    # Clean up test state
    medqueue_manager.reset_system_state()