"""
Bounded Clinician-Feedback-Driven Recalibration Engine for MedQueue.

Supports two recalibration update rules over review windows of N resolved cases:
  1. 'baseline_paper' (Paper Eqs. 3-4):
       Delta_w1 = delta * (p_under - p_over), clipped to [-delta_max, +delta_max]
       w1_new = clip(w1 + Delta_w1, w_min, w_max)
       w2_new = 0.75 * (1 - w1_new)
       w3_new = 0.25 * (1 - w1_new)

  2. 'root_cause_aware' (Refined Option A Update Rule):
       Preserves the exact 3 clinician feedback labels ('correctly_prioritized',
       'over_flagged', 'under_flagged') while decomposing 'over_flagged' events by
       dominant ranking term at the moment of service:
         - Wait-time-induced over-flags (p_over_wait): mild case jumped ahead because
           elapsed wait term w2*(t - t_i) exceeded urgency term w1*u_i -> increases w1
           (decreasing w2) to curb premature wait escalation.
         - Score-induced over-flags (p_over_score): case jumped ahead due to high
           urgency/confidence score -> decreases w1.
         - Under-flags (p_under): urgent case was delayed -> increases w1.
       Net update:
         Delta_w1 = delta * (p_under + p_over_wait - p_over_score)
       Bounded by |Delta_w1| <= delta_max and w1 in [w_min, w_max], preserving
       w1 + w2 + w3 = 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional

import numpy as np


class FeedbackLabel(str, Enum):
    CORRECTLY_PRIORITIZED = "correctly_prioritized"
    OVER_FLAGGED = "over_flagged"
    UNDER_FLAGGED = "under_flagged"


OverFlagCause = Literal["none", "wait_term_dominant", "score_term_dominant"]
RecalibrationMode = Literal["fixed", "baseline_paper", "root_cause_aware"]


@dataclass
class FeedbackRecord:
    """Single resolved case feedback entry accumulated in a review window."""
    case_id: str
    label: FeedbackLabel
    over_flag_cause: OverFlagCause = "none"
    timestamp: float = 0.0
    clinician_note: str = ""


@dataclass
class WeightVector:
    """Normalized composite ranking weights (w1 + w2 + w3 = 1.0)."""
    w1: float = 0.60  # Urgency weight
    w2: float = 0.30  # Elapsed wait-time weight
    w3: float = 0.10  # Confidence weight

    def __post_init__(self) -> None:
        total = self.w1 + self.w2 + self.w3
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

    def to_dict(self) -> Dict[str, float]:
        return {
            "w1": round(self.w1, 6),
            "w2": round(self.w2, 6),
            "w3": round(self.w3, 6),
        }


@dataclass
class RecalibrationEvent:
    """Audit record of a single batch recalibration step."""
    cycle_index: int
    window_size: int
    mode: RecalibrationMode
    p_correct: float
    p_over: float
    p_under: float
    p_over_wait: float
    p_over_score: float
    raw_delta_w1: float
    applied_delta_w1: float
    previous_weights: WeightVector
    updated_weights: WeightVector
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "cycle_index": self.cycle_index,
            "window_size": self.window_size,
            "mode": self.mode,
            "p_correct": round(self.p_correct, 4),
            "p_over": round(self.p_over, 4),
            "p_under": round(self.p_under, 4),
            "p_over_wait": round(self.p_over_wait, 4),
            "p_over_score": round(self.p_over_score, 4),
            "raw_delta_w1": round(self.raw_delta_w1, 6),
            "applied_delta_w1": round(self.applied_delta_w1, 6),
            "previous_weights": self.previous_weights.to_dict(),
            "updated_weights": self.updated_weights.to_dict(),
            "timestamp": round(self.timestamp, 2),
        }


class RecalibrationEngine:
    """
    Bounded batch-wise weight recalibrator driven by structured clinician feedback.
    """

    def __init__(
        self,
        initial_weights: Optional[WeightVector] = None,
        window_size: int = 20,
        delta: float = 0.08,
        delta_max: float = 0.05,
        w_min: float = 0.30,
        w_max: float = 0.90,
        w2_ratio: float = 0.75,
        w3_ratio: float = 0.25,
        mode: RecalibrationMode = "root_cause_aware",
    ) -> None:
        if not np.isclose(w2_ratio + w3_ratio, 1.0, atol=1e-6):
            raise ValueError("w2_ratio and w3_ratio must sum to 1.0.")

        self.weights = initial_weights if initial_weights is not None else WeightVector(0.60, 0.30, 0.10)
        self.window_size = window_size
        self.delta = delta
        self.delta_max = delta_max
        self.w_min = w_min
        self.w_max = w_max
        self.w2_ratio = w2_ratio
        self.w3_ratio = w3_ratio
        self.mode: RecalibrationMode = mode

        self.pending_buffer: List[FeedbackRecord] = []
        self.history: List[RecalibrationEvent] = []
        self._cycle_count: int = 0

    def record_feedback(
        self,
        record: FeedbackRecord,
        current_time: float = 0.0,
    ) -> Optional[RecalibrationEvent]:
        """
        Append a resolved case's clinician feedback to the active review window.
        Triggers a bounded weight update once `window_size` cases have accumulated.
        """
        if self.mode == "fixed":
            return None

        self.pending_buffer.append(record)
        if len(self.pending_buffer) < self.window_size:
            return None

        event = self._apply_window_update(current_time=current_time)
        self.pending_buffer.clear()
        return event

    def _apply_window_update(self, current_time: float = 0.0) -> RecalibrationEvent:
        """Compute proportions over the completed window and apply bounded update."""
        n = len(self.pending_buffer)
        n_correct = sum(1 for r in self.pending_buffer if r.label == FeedbackLabel.CORRECTLY_PRIORITIZED)
        n_over = sum(1 for r in self.pending_buffer if r.label == FeedbackLabel.OVER_FLAGGED)
        n_under = sum(1 for r in self.pending_buffer if r.label == FeedbackLabel.UNDER_FLAGGED)

        n_over_wait = sum(
            1
            for r in self.pending_buffer
            if r.label == FeedbackLabel.OVER_FLAGGED and r.over_flag_cause == "wait_term_dominant"
        )
        n_over_score = sum(
            1
            for r in self.pending_buffer
            if r.label == FeedbackLabel.OVER_FLAGGED and r.over_flag_cause != "wait_term_dominant"
        )

        p_correct = n_correct / n
        p_over = n_over / n
        p_under = n_under / n
        p_over_wait = n_over_wait / n
        p_over_score = n_over_score / n

        if self.mode == "baseline_paper":
            # Original Paper Eq. (3): Delta_w1 = delta * (p_under - p_over)
            raw_delta = self.delta * (p_under - p_over)
        elif self.mode == "root_cause_aware":
            # Refined Option A: wait-induced over-flags increase w1 (reducing w2),
            # while score-induced over-flags decrease w1.
            raw_delta = self.delta * (p_under + p_over_wait - p_over_score)
        else:
            raw_delta = 0.0

        # Enforce per-cycle step cap [-delta_max, +delta_max]
        bounded_delta = float(np.clip(raw_delta, -self.delta_max, self.delta_max))

        prev_weights = WeightVector(self.weights.w1, self.weights.w2, self.weights.w3)
        new_w1 = float(np.clip(prev_weights.w1 + bounded_delta, self.w_min, self.w_max))
        remainder = 1.0 - new_w1
        new_w2 = float(self.w2_ratio * remainder)
        new_w3 = float(self.w3_ratio * remainder)

        # Normalize to eliminate floating-point drift
        norm_sum = new_w1 + new_w2 + new_w3
        new_weights = WeightVector(
            w1=new_w1 / norm_sum,
            w2=new_w2 / norm_sum,
            w3=new_w3 / norm_sum,
        )

        applied_delta = new_weights.w1 - prev_weights.w1
        self.weights = new_weights
        self._cycle_count += 1

        event = RecalibrationEvent(
            cycle_index=self._cycle_count,
            window_size=n,
            mode=self.mode,
            p_correct=p_correct,
            p_over=p_over,
            p_under=p_under,
            p_over_wait=p_over_wait,
            p_over_score=p_over_score,
            raw_delta_w1=raw_delta,
            applied_delta_w1=applied_delta,
            previous_weights=prev_weights,
            updated_weights=new_weights,
            timestamp=current_time,
        )
        self.history.append(event)
        return event