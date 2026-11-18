"""
MedQueue FastAPI Application Entrypoint.

Exposes REST endpoints for:
  - Patient symptom intake & real-time urgency/explainability scoring
  - Doctor Triage Dashboard (live ranked queue, clinician override, 3-label feedback)
  - Recalibration configuration, weight drift history, and resolved audit logs
  - Research benchmark summaries and generated figures
  - Interactive live demo controls (synthetic batch seeding, time-warp, state reset)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.schemas import (
    AdvanceTimeRequest,
    CaseSubmissionResponse,
    ConfigUpdateRequest,
    DemoSeedRequest,
    FeedbackSubmissionResponse,
    PatientIntakeRequest,
    QueueStateResponse,
    StartConsultationRequest,
    SubmitFeedbackRequest,
)
from backend.service import BENCHMARK_JSON_PATH, ROOT_DIR, medqueue_manager

app = FastAPI(
    title="MedQueue Tele-Consultation Priority Queueing API",
    description=(
        "Clinician-feedback-driven adaptive priority queueing system with "
        "starvation prevention and token-level explainability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/api/health")
def health_check() -> Dict:
    return {
        "status": "healthy",
        "service": "MedQueue API",
        "weights": medqueue_manager.recalibrator.weights.to_dict(),
        "recalibration_mode": medqueue_manager.recalibrator.mode,
    }


@app.post("/api/intake", response_model=CaseSubmissionResponse)
def submit_patient_intake(payload: PatientIntakeRequest) -> Dict:
    """Patient submits free-text symptoms and receives priority queue confirmation."""
    return medqueue_manager.submit_patient_case(
        patient_name=payload.patient_name,
        patient_age=payload.patient_age,
        symptom_text=payload.symptom_text,
    )


@app.get("/api/queue", response_model=QueueStateResponse)
def get_live_queue() -> Dict:
    """Retrieve live composite-ranked queue, active consultation, and recalibration state."""
    return medqueue_manager.get_queue_snapshot()


@app.post("/api/consultation/start")
def start_consultation(payload: StartConsultationRequest) -> Dict:
    """Clinician starts consulting the top-ranked patient (or overrides with a chosen case_id)."""
    try:
        active_case = medqueue_manager.start_next_consultation(
            override_case_id=payload.override_case_id
        )
        return {"status": "started", "active_case": active_case}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/consultation/feedback", response_model=FeedbackSubmissionResponse)
def submit_clinician_feedback(payload: SubmitFeedbackRequest) -> Dict:
    """Clinician resolves active consultation and submits structured 3-label feedback."""
    try:
        return medqueue_manager.resolve_and_submit_feedback(
            case_id=payload.case_id,
            feedback_label=payload.feedback_label,
            clinician_note=payload.clinician_note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/history/weights")
def get_weight_history() -> List[Dict]:
    """Retrieve historical trajectory of (w1, w2, w3) across recalibration windows."""
    return medqueue_manager.get_weight_history()


@app.get("/api/history/resolved")
def get_resolved_cases(limit: int = 50) -> List[Dict]:
    """Retrieve resolved consultation audit log."""
    return medqueue_manager.get_resolved_history(limit=limit)


@app.get("/api/benchmarks")
def get_research_benchmarks() -> Dict:
    """Serve the 30-run x 2000-case simulation summary tables and figure URLs."""
    if not os.path.exists(BENCHMARK_JSON_PATH):
        raise HTTPException(
            status_code=404,
            detail="Benchmark results not generated yet. Run 'python -m simulation.run_benchmarks'.",
        )
    with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["figures"] = [
        {"id": "fig1", "title": "FIFO vs. MedQueue High-Urgency Wait", "url": "/static/outputs/fig1_fifo_vs_medqueue.png"},
        {"id": "fig2", "title": "Starvation Ablation (Max Low-Urgency Wait)", "url": "/static/outputs/fig2_starvation_ablation.png"},
        {"id": "fig3", "title": "Baseline vs. Refined Option A Recalibration", "url": "/static/outputs/fig3_recalibration_comparison.png"},
        {"id": "fig4", "title": "Urgency Weight (w1) Evolution Trajectory", "url": "/static/outputs/fig4_weight_trajectory.png"},
    ]
    return data


@app.post("/api/config/recalibration")
def update_recalibration_settings(payload: ConfigUpdateRequest) -> Dict:
    """Switch recalibration mode ('root_cause_aware', 'baseline_paper', 'fixed') or window size."""
    return medqueue_manager.update_recalibration_config(
        mode=payload.recalibration_mode,
        window_size=payload.window_size,
        delta=payload.delta,
    )


@app.post("/api/demo/seed-batch")
def demo_seed_patients(payload: DemoSeedRequest) -> Dict:
    """Seed realistic synthetic tele-consultation cases for live demonstration."""
    cases = medqueue_manager.seed_synthetic_batch(
        count=payload.count,
        stagger_minutes=payload.stagger_minutes,
    )
    return {"seeded_count": len(cases), "cases": cases}


@app.post("/api/demo/advance-time")
def demo_advance_clock(payload: AdvanceTimeRequest) -> Dict:
    """Advance queue clock by N minutes to demonstrate wait-time escalation."""
    new_min = medqueue_manager.advance_time(payload.minutes)
    return {"current_queue_minute": new_min, "queue": medqueue_manager.get_queue_snapshot()}


@app.post("/api/demo/reset")
def demo_reset_all() -> Dict:
    """Reset database and weights back to initial state."""
    snapshot = medqueue_manager.reset_system_state()
    return {"status": "reset", "queue": snapshot}