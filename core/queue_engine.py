"""
Adaptive Priority Queue Engine for MedQueue.

Implements:
  1. Single source of truth for waiting tele-consultation cases.
  2. Composite linear ranking function (Paper Eq. 2 / Synopsis Eq. 6.2):
       R_i(t) = w1 * u_i + w2 * (t - t_i) + w3 * (100 * c_i)
     with starvation-prevention guarantee via monotonic wait-time escalation.
  3. Deterministic tie-breaking (earliest arrival time t_i, then insertion sequence).
  4. Root-cause diagnostic tagging upon case resolution to support root-cause-aware
     recalibration and experimental error attribution.
  5. Clinician override tracking (preserving full clinical autonomy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.recalibrator import (
    FeedbackLabel,
    FeedbackRecord,
    OverFlagCause,
    RecalibrationEngine,
    RecalibrationEvent,
    WeightVector,
)


@dataclass
class QueueCase:
    """Represents a patient case managed inside the Adaptive Priority Queue."""
    case_id: str
    symptom_text: str
    urgency_score: float              # u_i in [0, 100]
    confidence: float                 # c_i in [0, 1]
    arrival_time: float               # t_i (simulation minutes or Unix epoch minutes)
    true_severity: Optional[float] = None  # Known ground-truth severity in simulation/evaluation
    explanation_summary: str = ""
    top_contributors: List[Dict] = field(default_factory=list)
    insertion_seq: int = 0

    # Tracking fields for inversion & starvation analysis
    max_less_urgent_passed_by: float = 0.0  # Largest (true_severity - passer_severity) seen while waiting
    service_start_time: Optional[float] = None
    resolution_time: Optional[float] = None
    clinician_override: bool = False


@dataclass
class RankedCaseView:
    """Snapshot of a waiting case's composite score and term breakdown at time t."""
    rank_position: int
    case: QueueCase
    composite_score: float
    urgency_component: float
    wait_component: float
    confidence_component: float
    elapsed_wait: float

    def to_dict(self) -> Dict:
        return {
            "rank_position": self.rank_position,
            "case_id": self.case.case_id,
            "symptom_text": self.case.symptom_text,
            "urgency_score": round(self.case.urgency_score, 2),
            "confidence": round(self.case.confidence, 4),
            "arrival_time": round(self.case.arrival_time, 2),
            "elapsed_wait": round(self.elapsed_wait, 2),
            "composite_score": round(self.composite_score, 3),
            "term_breakdown": {
                "urgency_term": round(self.urgency_component, 3),
                "wait_term": round(self.wait_component, 3),
                "confidence_term": round(self.confidence_component, 3),
            },
            "explanation_summary": self.case.explanation_summary,
            "top_contributors": self.case.top_contributors,
            "true_severity": self.case.true_severity,
        }


class AdaptivePriorityQueue:
    """
    Stateful, starvation-preventing priority queue coupled to a RecalibrationEngine.
    """

    def __init__(
        self,
        recalibrator: Optional[RecalibrationEngine] = None,
        policy: str = "medqueue",  # 'medqueue', 'fifo', or 'urgency_only'
        inversion_tolerance: float = 12.0,
    ) -> None:
        if policy not in ("medqueue", "fifo", "urgency_only"):
            raise ValueError(f"Unsupported queue policy: {policy}")

        self.policy = policy
        self.inversion_tolerance = inversion_tolerance
        self.recalibrator = recalibrator if recalibrator is not None else RecalibrationEngine()

        self._waiting: List[QueueCase] = []
        self._resolved: List[QueueCase] = []
        self._seq_counter: int = 0

    @property
    def weights(self) -> WeightVector:
        return self.recalibrator.weights

    def __len__(self) -> int:
        return len(self._waiting)

    def enqueue(self, case: QueueCase) -> None:
        """Insert a newly scored patient case into the waiting queue."""
        self._seq_counter += 1
        case.insertion_seq = self._seq_counter
        self._waiting.append(case)

    def compute_rank_terms(self, case: QueueCase, current_time: float) -> Tuple[float, float, float, float]:
        """
        Compute R_i(t) and its three constituent terms at time t.
        Returns (composite_score, urgency_term, wait_term, confidence_term).
        """
        elapsed_wait = max(0.0, current_time - case.arrival_time)

        if self.policy == "fifo":
            # Pure arrival order: negate arrival_time so earlier arrivals score higher
            return (-case.arrival_time, 0.0, elapsed_wait, 0.0)

        w = self.weights
        u_term = w.w1 * case.urgency_score
        c_term = w.w3 * (100.0 * case.confidence)

        if self.policy == "urgency_only":
            # Ablation condition (w2 = 0): ignore elapsed wait time
            w_term = 0.0
        else:
            w_term = w.w2 * elapsed_wait

        composite = u_term + w_term + c_term
        return (composite, u_term, w_term, c_term)

    def get_ranked_queue(self, current_time: float) -> List[RankedCaseView]:
        """
        Dynamically rank all waiting cases at time t without mutating arrival metadata.
        Tie-breaker: higher composite score -> earlier arrival_time -> earlier insertion_seq.
        """
        scored_items = []
        for case in self._waiting:
            comp, u_term, w_term, c_term = self.compute_rank_terms(case, current_time)
            elapsed = max(0.0, current_time - case.arrival_time)
            scored_items.append((comp, -case.arrival_time, -case.insertion_seq, case, u_term, w_term, c_term, elapsed))

        scored_items.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        ranked_views: List[RankedCaseView] = []
        for idx, item in enumerate(scored_items, start=1):
            comp, _, _, case, u_term, w_term, c_term, elapsed = item
            ranked_views.append(
                RankedCaseView(
                    rank_position=idx,
                    case=case,
                    composite_score=comp,
                    urgency_component=u_term,
                    wait_component=w_term,
                    confidence_component=c_term,
                    elapsed_wait=elapsed,
                )
            )
        return ranked_views

    def diagnose_service_decision(
        self,
        selected_view: RankedCaseView,
        all_views: List[RankedCaseView],
    ) -> Tuple[bool, OverFlagCause, bool]:
        """
        Evaluate a service decision against currently waiting cases (when ground-truth
        severity is available in simulation/evaluation) and determine whether any
        inversion was driven by the wait-time term or the urgency/confidence score.

        Returns
        -------
        (is_inversion, over_flag_cause, was_under_flagged)
        """
        selected_case = selected_view.case
        is_inversion = False
        over_flag_cause: OverFlagCause = "none"

        if selected_case.true_severity is not None:
            for other_view in all_views:
                other_case = other_view.case
                if other_case.case_id == selected_case.case_id or other_case.true_severity is None:
                    continue

                severity_gap = other_case.true_severity - selected_case.true_severity
                if severity_gap > self.inversion_tolerance:
                    is_inversion = True
                    # Check whether wait-time escalation caused selected_case to overtake other_case
                    if selected_view.wait_component > selected_view.urgency_component:
                        over_flag_cause = "wait_term_dominant"
                    else:
                        over_flag_cause = "score_term_dominant"
                    break

            # Update pass-over tracking for remaining waiting cases that are clearly more urgent
            for other_view in all_views:
                other_case = other_view.case
                if other_case.case_id == selected_case.case_id or other_case.true_severity is None:
                    continue
                gap = other_case.true_severity - selected_case.true_severity
                if gap > other_case.max_less_urgent_passed_by:
                    other_case.max_less_urgent_passed_by = gap

        was_under_flagged = selected_case.max_less_urgent_passed_by > self.inversion_tolerance
        return is_inversion, over_flag_cause, was_under_flagged

    def infer_over_flag_cause_from_terms(self, case: QueueCase, service_time: float) -> OverFlagCause:
        """
        In live clinical operation (where ground-truth severity is unknown), inspect the
        served case's ranking term breakdown at service time to determine whether its rank
        was dominated by wait-time escalation or urgency score.
        """
        _, u_term, w_term, _ = self.compute_rank_terms(case, service_time)
        if w_term > u_term:
            return "wait_term_dominant"
        return "score_term_dominant"

    def pop_next_case(
        self,
        current_time: float,
        override_case_id: Optional[str] = None,
    ) -> Optional[Tuple[RankedCaseView, bool, OverFlagCause, bool]]:
        """
        Select and remove the highest-ranked waiting case (or a clinician-overridden case)
        for consultation at `current_time`.

        Returns
        -------
        Tuple of (selected_view, is_inversion, over_flag_cause, was_under_flagged), or None if empty.
        """
        if not self._waiting:
            return None

        ranked_views = self.get_ranked_queue(current_time)

        selected_view: Optional[RankedCaseView] = None
        if override_case_id is not None:
            for view in ranked_views:
                if view.case.case_id == override_case_id:
                    selected_view = view
                    selected_view.case.clinician_override = True
                    break
            if selected_view is None:
                raise KeyError(f"Override case_id '{override_case_id}' not found in waiting queue.")
        else:
            selected_view = ranked_views[0]

        is_inversion, over_flag_cause, was_under_flagged = self.diagnose_service_decision(
            selected_view, ranked_views
        )

        selected_view.case.service_start_time = current_time
        self._waiting = [c for c in self._waiting if c.case_id != selected_view.case.case_id]
        self._resolved.append(selected_view.case)

        return selected_view, is_inversion, over_flag_cause, was_under_flagged

    def submit_feedback(
        self,
        case_id: str,
        label: FeedbackLabel,
        current_time: float,
        over_flag_cause: Optional[OverFlagCause] = None,
        clinician_note: str = "",
    ) -> Optional[RecalibrationEvent]:
        """
        Record clinician feedback on a resolved case and trigger batch recalibration
        when the review window completes.
        """
        resolved_case = next((c for c in self._resolved if c.case_id == case_id), None)
        if resolved_case is None:
            raise KeyError(f"Case '{case_id}' has not been resolved yet.")

        resolved_case.resolution_time = current_time

        cause: OverFlagCause = "none"
        if label == FeedbackLabel.OVER_FLAGGED:
            if over_flag_cause is not None and over_flag_cause != "none":
                cause = over_flag_cause
            else:
                svc_t = resolved_case.service_start_time if resolved_case.service_start_time is not None else current_time
                cause = self.infer_over_flag_cause_from_terms(resolved_case, svc_t)

        record = FeedbackRecord(
            case_id=case_id,
            label=label,
            over_flag_cause=cause,
            timestamp=current_time,
            clinician_note=clinician_note,
        )
        return self.recalibrator.record_feedback(record, current_time=current_time)


if __name__ == "__main__":
    # Quick sanity verification of starvation overtaking & bounded recalibration
    recal = RecalibrationEngine(window_size=2, delta=0.5, mode="root_cause_aware")
    pq = AdaptivePriorityQueue(recalibrator=recal, policy="medqueue")

    mild_case = QueueCase(
        case_id="MILD-1",
        symptom_text="mild sore throat",
        urgency_score=20.0,
        confidence=0.80,
        arrival_time=0.0,
        true_severity=20.0,
    )
    urgent_case = QueueCase(
        case_id="URG-1",
        symptom_text="acute chest pain",
        urgency_score=90.0,
        confidence=0.80,
        arrival_time=145.0,
        true_severity=90.0,
    )

    pq.enqueue(mild_case)
    pq.enqueue(urgent_case)

    ranked = pq.get_ranked_queue(current_time=145.0)
    print("=== Starvation-Prevention Overtaking Check at t=145 min ===")
    for item in ranked:
        print(
            f"Rank #{item.rank_position}: {item.case.case_id} | "
            f"R(t)={item.composite_score:.2f} "
            f"(u_term={item.urgency_component:.2f}, w_term={item.wait_component:.2f}, c_term={item.confidence_component:.2f})"
        )

    # Pop top case and verify root-cause-aware recalibration
    popped = pq.pop_next_case(current_time=145.0)
    assert popped is not None
    view, is_inv, cause, _ = popped
    print(f"\nServed: {view.case.case_id} | Inversion={is_inv} | Cause={cause}")
    pq.submit_feedback(view.case.case_id, FeedbackLabel.OVER_FLAGGED, current_time=151.0, over_flag_cause=cause)

    popped2 = pq.pop_next_case(current_time=151.0)
    assert popped2 is not None
    view2, _, _, was_under = popped2
    event = pq.submit_feedback(view2.case.case_id, FeedbackLabel.UNDER_FLAGGED, current_time=157.0)
    if event:
        print("\n=== Recalibration Triggered (Window=2) ===")
        print(f"Previous weights: {event.previous_weights.to_dict()}")
        print(f"Updated weights : {event.updated_weights.to_dict()}")