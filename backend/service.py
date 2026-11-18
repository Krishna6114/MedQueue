"""
MedQueue Application Service Layer.

Bridges FastAPI + SQLite persistence with the shared `core` algorithmic engine:
  - Loads or trains the UrgencyScorer artifact on startup.
  - Maintains the single-source-of-truth AdaptivePriorityQueue and RecalibrationEngine.
  - Synchronizes case states, clinician overrides, feedback windows, and weight history
    with SQLite (`medqueue.db`).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.database import PatientCaseModel, SessionLocal, WeightHistoryModel, init_db
from core.dataset import generate_synthetic_dataset
from core.queue_engine import AdaptivePriorityQueue, QueueCase, RankedCaseView
from core.recalibrator import FeedbackLabel, RecalibrationEngine, RecalibrationMode, WeightVector
from core.scorer import UrgencyScorer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV_PATH = os.path.join(ROOT_DIR, "data", "synthetic_cases.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "urgency_scorer.joblib")
BENCHMARK_JSON_PATH = os.path.join(ROOT_DIR, "outputs", "benchmark_summary.json")


class MedQueueManager:
    """
    Singleton coordinator managing live scoring, queue state, recalibration, and persistence.
    """

    def __init__(self) -> None:
        init_db()
        self.scorer = self._load_or_train_scorer()
        self.recalibrator = RecalibrationEngine(
            initial_weights=WeightVector(0.60, 0.30, 0.10),
            window_size=20,
            delta=0.08,
            delta_max=0.05,
            w_min=0.30,
            w_max=0.90,
            mode="root_cause_aware",
        )
        self.queue_engine = AdaptivePriorityQueue(
            recalibrator=self.recalibrator,
            policy="medqueue",
            inversion_tolerance=12.0,
        )

        # Logical queue clock (starts at 0.0 and advances automatically with real wall-clock
        # elapsed minutes PLUS any manual demo time-warps)
        self._base_wall_time = time.time()
        self._time_offset_minutes = 0.0
        self._case_counter = 0
        self._in_consultation_case_id: Optional[str] = None
        self._patient_meta: Dict[str, Dict] = {}

        self._restore_state_from_db()

    def _load_or_train_scorer(self) -> UrgencyScorer:
        if os.path.exists(MODEL_PATH):
            return UrgencyScorer.load(MODEL_PATH)

        if os.path.exists(DATA_CSV_PATH):
            df = pd.read_csv(DATA_CSV_PATH)
        else:
            df = generate_synthetic_dataset(n_cases=400, seed=42, output_csv_path=DATA_CSV_PATH)

        scorer = UrgencyScorer(random_state=42)
        scorer.fit(df["symptom_text"], df["true_severity"])
        scorer.save(MODEL_PATH)
        return scorer

    def current_queue_minute(self) -> float:
        """Current logical queue time in minutes."""
        wall_elapsed_min = (time.time() - self._base_wall_time) / 60.0
        return round(wall_elapsed_min + self._time_offset_minutes, 3)

    def advance_time(self, minutes: float) -> float:
        """Advance logical queue time (for live starvation & recalibration demonstrations)."""
        self._time_offset_minutes += minutes
        return self.current_queue_minute()

    def _restore_state_from_db(self) -> None:
        """Rebuild in-memory queue and latest weights from SQLite on server startup."""
        db: Session = SessionLocal()
        try:
            all_cases = db.query(PatientCaseModel).order_by(PatientCaseModel.id.asc()).all()
            self._case_counter = len(all_cases)

            # Restore latest recalibrated weights if history exists
            latest_weight = (
                db.query(WeightHistoryModel).order_by(WeightHistoryModel.id.desc()).first()
            )
            if latest_weight is not None:
                self.recalibrator.weights = WeightVector(
                    w1=latest_weight.new_w1,
                    w2=latest_weight.new_w2,
                    w3=latest_weight.new_w3,
                )
                self.recalibrator._cycle_count = latest_weight.cycle_index

            max_minute = 0.0
            for row in all_cases:
                max_minute = max(max_minute, row.arrival_minute)
                self._patient_meta[row.case_id] = {
                    "patient_name": row.patient_name,
                    "patient_age": row.patient_age,
                }
                contributors = json.loads(row.top_contributors_json or "[]")
                q_case = QueueCase(
                    case_id=row.case_id,
                    symptom_text=row.symptom_text,
                    urgency_score=row.urgency_score,
                    confidence=row.confidence,
                    arrival_time=row.arrival_minute,
                    true_severity=row.true_severity,
                    explanation_summary=row.explanation_summary,
                    top_contributors=contributors,
                    service_start_time=row.service_start_minute,
                    resolution_time=row.resolution_minute,
                    clinician_override=row.clinician_override,
                )
                if row.status == "waiting":
                    self.queue_engine.enqueue(q_case)
                elif row.status == "in_consultation":
                    self.queue_engine._resolved.append(q_case)
                    self._in_consultation_case_id = row.case_id
                elif row.status == "resolved":
                    self.queue_engine._resolved.append(q_case)

            if max_minute > 0.0:
                self._time_offset_minutes = max_minute + 1.0
        finally:
            db.close()

    def _serialize_ranked_view(self, view: RankedCaseView) -> Dict:
        meta = self._patient_meta.get(view.case.case_id, {"patient_name": "Patient", "patient_age": None})
        base = view.to_dict()
        base["patient_name"] = meta["patient_name"]
        base["patient_age"] = meta["patient_age"]
        base["arrival_minute"] = base.pop("arrival_time")
        return base

    def submit_patient_case(
        self,
        patient_name: str,
        symptom_text: str,
        patient_age: Optional[int] = None,
        true_severity: Optional[float] = None,
        arrival_minute_override: Optional[float] = None,
    ) -> Dict:
        """Score a new patient symptom submission, persist to SQLite, and enqueue."""
        self._case_counter += 1
        case_id = f"MQ-{self._case_counter:04d}"
        arr_min = (
            arrival_minute_override
            if arrival_minute_override is not None
            else self.current_queue_minute()
        )

        scoring_res = self.scorer.score_case(symptom_text, top_k_tokens=5)
        res_dict = scoring_res.to_dict()

        q_case = QueueCase(
            case_id=case_id,
            symptom_text=symptom_text.strip(),
            urgency_score=res_dict["urgency_score"],
            confidence=res_dict["confidence"],
            arrival_time=arr_min,
            true_severity=true_severity,
            explanation_summary=res_dict["explanation_summary"],
            top_contributors=res_dict["top_contributors"],
        )
        self.queue_engine.enqueue(q_case)
        self._patient_meta[case_id] = {
            "patient_name": patient_name.strip() or "Patient",
            "patient_age": patient_age,
        }

        db: Session = SessionLocal()
        try:
            db_case = PatientCaseModel(
                case_id=case_id,
                patient_name=self._patient_meta[case_id]["patient_name"],
                patient_age=patient_age,
                symptom_text=q_case.symptom_text,
                urgency_score=q_case.urgency_score,
                confidence=q_case.confidence,
                prediction_std=res_dict["prediction_std"],
                explanation_summary=q_case.explanation_summary,
                top_contributors_json=json.dumps(q_case.top_contributors),
                arrival_minute=arr_min,
                status="waiting",
                true_severity=true_severity,
            )
            db.add(db_case)
            db.commit()
        finally:
            db.close()

        now_min = self.current_queue_minute()
        ranked = self.queue_engine.get_ranked_queue(now_min)
        rank_pos = next((r.rank_position for r in ranked if r.case.case_id == case_id), len(ranked))

        return {
            "case_id": case_id,
            "patient_name": self._patient_meta[case_id]["patient_name"],
            "patient_age": patient_age,
            "symptom_text": q_case.symptom_text,
            "urgency_score": q_case.urgency_score,
            "confidence": q_case.confidence,
            "explanation_summary": q_case.explanation_summary,
            "top_contributors": q_case.top_contributors,
            "arrival_minute": round(arr_min, 2),
            "current_rank_position": rank_pos,
            "total_waiting": len(ranked),
        }

    def get_queue_snapshot(self) -> Dict:
        """Return live ranked waiting queue, active consultation case, and recalibration state."""
        now_min = self.current_queue_minute()
        ranked_views = self.queue_engine.get_ranked_queue(now_min)
        serialized_queue = [self._serialize_ranked_view(v) for v in ranked_views]

        active_case_data = None
        if self._in_consultation_case_id is not None:
            active_q_case = next(
                (c for c in self.queue_engine._resolved if c.case_id == self._in_consultation_case_id),
                None,
            )
            if active_q_case is not None:
                svc_t = active_q_case.service_start_minute or now_min
                comp, u_t, w_t, c_t = self.queue_engine.compute_rank_terms(active_q_case, svc_t)
                active_view = RankedCaseView(
                    rank_position=0,
                    case=active_q_case,
                    composite_score=comp,
                    urgency_component=u_t,
                    wait_component=w_t,
                    confidence_component=c_t,
                    elapsed_wait=max(0.0, svc_t - active_q_case.arrival_time),
                )
                active_case_data = self._serialize_ranked_view(active_view)

        return {
            "current_queue_minute": round(now_min, 2),
            "waiting_count": len(serialized_queue),
            "in_consultation_case": active_case_data,
            "waiting_queue": serialized_queue,
            "weights": self.recalibrator.weights.to_dict(),
            "recalibration_mode": self.recalibrator.mode,
            "window_size": self.recalibrator.window_size,
            "pending_feedback_count": len(self.recalibrator.pending_buffer),
            "completed_recalibration_cycles": self.recalibrator._cycle_count,
        }

    def start_next_consultation(self, override_case_id: Optional[str] = None) -> Dict:
        """Pop the top-ranked case (or clinician-overridden case) into active consultation."""
        if self._in_consultation_case_id is not None:
            raise ValueError(
                f"Case '{self._in_consultation_case_id}' is currently in consultation. "
                "Submit feedback to resolve it before starting the next consultation."
            )

        now_min = self.current_queue_minute()
        popped = self.queue_engine.pop_next_case(current_time=now_min, override_case_id=override_case_id)
        if popped is None:
            raise ValueError("The waiting queue is currently empty.")

        selected_view, _, _, _ = popped
        self._in_consultation_case_id = selected_view.case.case_id

        db: Session = SessionLocal()
        try:
            db_case = (
                db.query(PatientCaseModel)
                .filter(PatientCaseModel.case_id == selected_view.case.case_id)
                .first()
            )
            if db_case:
                db_case.status = "in_consultation"
                db_case.service_start_minute = now_min
                db_case.clinician_override = bool(override_case_id is not None)
                db.commit()
        finally:
            db.close()

        return self._serialize_ranked_view(selected_view)

    def resolve_and_submit_feedback(
        self,
        case_id: str,
        feedback_label: str,
        clinician_note: str = "",
    ) -> Dict:
        """Resolve an active consultation, record clinician feedback, and trigger batch recalibration."""
        now_min = self.current_queue_minute()
        label_enum = FeedbackLabel(feedback_label)

        recal_event = self.queue_engine.submit_feedback(
            case_id=case_id,
            label=label_enum,
            current_time=now_min,
            over_flag_cause=None,
            clinician_note=clinician_note,
        )

        # Retrieve inferred over-flag cause from the last buffer entry or event
        inferred_cause = "none"
        if label_enum == FeedbackLabel.OVER_FLAGGED:
            if self.recalibrator.pending_buffer:
                inferred_cause = self.recalibrator.pending_buffer[-1].over_flag_cause
            else:
                resolved_c = next((c for c in self.queue_engine._resolved if c.case_id == case_id), None)
                if resolved_c:
                    svc_t = resolved_c.service_start_time or now_min
                    inferred_cause = self.queue_engine.infer_over_flag_cause_from_terms(resolved_c, svc_t)

        if self._in_consultation_case_id == case_id:
            self._in_consultation_case_id = None

        db: Session = SessionLocal()
        try:
            db_case = db.query(PatientCaseModel).filter(PatientCaseModel.case_id == case_id).first()
            if db_case:
                db_case.status = "resolved"
                db_case.resolution_minute = now_min
                db_case.feedback_label = label_enum.value
                db_case.over_flag_cause = inferred_cause
                db_case.clinician_note = clinician_note

            if recal_event is not None:
                ev_dict = recal_event.to_dict()
                db.add(
                    WeightHistoryModel(
                        cycle_index=ev_dict["cycle_index"],
                        window_size=ev_dict["window_size"],
                        recalibration_mode=ev_dict["mode"],
                        p_correct=ev_dict["p_correct"],
                        p_over=ev_dict["p_over"],
                        p_under=ev_dict["p_under"],
                        p_over_wait=ev_dict["p_over_wait"],
                        p_over_score=ev_dict["p_over_score"],
                        raw_delta_w1=ev_dict["raw_delta_w1"],
                        applied_delta_w1=ev_dict["applied_delta_w1"],
                        prev_w1=ev_dict["previous_weights"]["w1"],
                        prev_w2=ev_dict["previous_weights"]["w2"],
                        prev_w3=ev_dict["previous_weights"]["w3"],
                        new_w1=ev_dict["updated_weights"]["w1"],
                        new_w2=ev_dict["updated_weights"]["w2"],
                        new_w3=ev_dict["updated_weights"]["w3"],
                        queue_minute=now_min,
                    )
                )
            db.commit()
        finally:
            db.close()

        return {
            "case_id": case_id,
            "feedback_label": label_enum.value,
            "over_flag_cause": inferred_cause,
            "recalibration_triggered": recal_event is not None,
            "recalibration_event": recal_event.to_dict() if recal_event is not None else None,
            "current_weights": self.recalibrator.weights.to_dict(),
            "pending_feedback_count": len(self.recalibrator.pending_buffer),
            "window_size": self.recalibrator.window_size,
        }

    def get_weight_history(self) -> List[Dict]:
        """Return chronological weight recalibration history from SQLite."""
        db: Session = SessionLocal()
        try:
            rows = db.query(WeightHistoryModel).order_by(WeightHistoryModel.id.asc()).all()
            history = [
                {
                    "cycle_index": 0,
                    "w1": 0.60,
                    "w2": 0.30,
                    "w3": 0.10,
                    "p_correct": None,
                    "p_over": None,
                    "p_under": None,
                    "applied_delta_w1": 0.0,
                    "mode": self.recalibrator.mode,
                    "queue_minute": 0.0,
                }
            ]
            for r in rows:
                history.append(
                    {
                        "cycle_index": r.cycle_index,
                        "w1": round(r.new_w1, 4),
                        "w2": round(r.new_w2, 4),
                        "w3": round(r.new_w3, 4),
                        "p_correct": round(r.p_correct, 3),
                        "p_over": round(r.p_over, 3),
                        "p_under": round(r.p_under, 3),
                        "applied_delta_w1": round(r.applied_delta_w1, 4),
                        "mode": r.recalibration_mode,
                        "queue_minute": round(r.queue_minute, 2),
                    }
                )
            return history
        finally:
            db.close()

    def get_resolved_history(self, limit: int = 50) -> List[Dict]:
        """Return recently resolved cases with clinician feedback labels."""
        db: Session = SessionLocal()
        try:
            rows = (
                db.query(PatientCaseModel)
                .filter(PatientCaseModel.status == "resolved")
                .order_by(PatientCaseModel.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "case_id": r.case_id,
                    "patient_name": r.patient_name,
                    "patient_age": r.patient_age,
                    "symptom_text": r.symptom_text,
                    "urgency_score": r.urgency_score,
                    "confidence": r.confidence,
                    "wait_duration": round((r.service_start_minute or r.arrival_minute) - r.arrival_minute, 2),
                    "clinician_override": r.clinician_override,
                    "feedback_label": r.feedback_label,
                    "over_flag_cause": r.over_flag_cause,
                    "clinician_note": r.clinician_note,
                }
                for r in rows
            ]
        finally:
            db.close()

    def seed_synthetic_batch(self, count: int = 10, stagger_minutes: float = 2.0) -> List[Dict]:
        """Populate the live queue with a realistic mix of ESI-1 to ESI-5 synthetic patients."""
        if os.path.exists(DATA_CSV_PATH):
            df = pd.read_csv(DATA_CSV_PATH)
        else:
            df = generate_synthetic_dataset(n_cases=400, seed=42, output_csv_path=DATA_CSV_PATH)

        sample_names = [
            ("Aarav Sharma", 58), ("Priya Verma", 34), ("Rohan Mehta", 67),
            ("Ananya Singh", 26), ("Vikram Nair", 49), ("Kavita Joshi", 72),
            ("Siddharth Rao", 31), ("Meera Kulkarni", 44), ("Arjun Patel", 61),
            ("Neha Gupta", 29), ("Rajesh Tiwari", 53), ("Sunita Devi", 64),
        ]

        rng = np.random.default_rng()
        # Sample stratified across High, Moderate, and Low urgency so queue reordering is immediately visible
        sampled_rows = df.sample(n=min(count, len(df)), random_state=int(rng.integers(1, 100_000)))

        base_min = self.current_queue_minute()
        created_cases = []
        for idx, (_, row) in enumerate(sampled_rows.iterrows()):
            name, age = sample_names[idx % len(sample_names)]
            arr_t = base_min + (idx * stagger_minutes)
            res = self.submit_patient_case(
                patient_name=f"{name}",
                patient_age=age,
                symptom_text=str(row["symptom_text"]),
                true_severity=float(row["true_severity"]),
                arrival_minute_override=arr_t,
            )
            created_cases.append(res)

        if count > 0 and stagger_minutes > 0:
            self._time_offset_minutes += count * stagger_minutes

        return created_cases

    def update_recalibration_config(
        self,
        mode: RecalibrationMode,
        window_size: int = 20,
        delta: float = 0.08,
    ) -> Dict:
        """Update recalibration mode, window size, or step size at runtime."""
        self.recalibrator.mode = mode
        self.recalibrator.window_size = window_size
        self.recalibrator.delta = delta
        return {
            "recalibration_mode": self.recalibrator.mode,
            "window_size": self.recalibrator.window_size,
            "delta": self.recalibrator.delta,
            "weights": self.recalibrator.weights.to_dict(),
        }

    def reset_system_state(self) -> Dict:
        """Clear SQLite tables and reset weights to initial (0.60, 0.30, 0.10) for fresh demos."""
        db: Session = SessionLocal()
        try:
            db.query(PatientCaseModel).delete()
            db.query(WeightHistoryModel).delete()
            db.commit()
        finally:
            db.close()

        self.recalibrator = RecalibrationEngine(
            initial_weights=WeightVector(0.60, 0.30, 0.10),
            window_size=self.recalibrator.window_size,
            delta=self.recalibrator.delta,
            delta_max=0.05,
            w_min=0.30,
            w_max=0.90,
            mode=self.recalibrator.mode,
        )
        self.queue_engine = AdaptivePriorityQueue(
            recalibrator=self.recalibrator,
            policy="medqueue",
            inversion_tolerance=12.0,
        )
        self._base_wall_time = time.time()
        self._time_offset_minutes = 0.0
        self._case_counter = 0
        self._in_consultation_case_id = None
        self._patient_meta.clear()
        return self.get_queue_snapshot()


# Global singleton instance
medqueue_manager = MedQueueManager()