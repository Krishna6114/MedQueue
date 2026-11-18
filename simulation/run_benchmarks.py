"""
Discrete-Event Simulation & Statistical Benchmark Suite for MedQueue.

Reproduces and extends the experimental methodology in Section IV of the Research Paper:
  - 30 independent Monte Carlo runs of 2,000 cases each (sampled with replacement
    from the 400 out-of-fold scored synthetic cases).
  - Poisson arrivals (mean inter-arrival gap = 7.5 min), single clinician with
    fixed service duration = 6.0 min (utilization rho = 0.80).
  - Paired design: every queue policy processes the exact same case sequence and
    arrival timestamps within each run.
  - Compares:
      1. FIFO (Baseline)
      2. Urgency-Only (w2 = 0 ablation)
      3. MedQueue Fixed Weights (0.60, 0.30, 0.10)
      4. MedQueue Baseline Recalibration (Paper Eq. 3, delta = 0.08)
      5. MedQueue Refined Option A Recalibration (Root-cause-aware, delta = 0.08)
      6. MedQueue Refined Option A Recalibration (Root-cause-aware, delta = 0.25)
      7. MedQueue Refined Option A + Noisy/Biased Clinician Feedback (15% noise + systematic bias)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless rendering for reliable PNG generation on Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from core.queue_engine import AdaptivePriorityQueue, QueueCase
from core.recalibrator import FeedbackLabel, RecalibrationEngine, RecalibrationMode, WeightVector


@dataclass
class RunMetrics:
    """Evaluation metrics recorded for a single 2,000-case simulation run."""
    condition: str
    run_id: int
    high_urgency_avg_wait: float     # True severity >= 75
    inversion_rate: float            # Fraction of service decisions passing over >12pt more urgent case
    low_urgency_max_wait: float      # True severity < 40 (starvation max)
    low_urgency_p95_wait: float      # True severity < 40 (starvation 95th percentile)
    final_w1: float
    final_w2: float
    final_w3: float
    w1_trajectory: List[float]


def generate_run_arrivals(
    scored_df: pd.DataFrame,
    n_cases: int = 2000,
    mean_interarrival: float = 7.5,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Sample `n_cases` with replacement and generate Poisson arrival timestamps."""
    if rng is None:
        rng = np.random.default_rng(42)

    sampled_indices = rng.integers(0, len(scored_df), size=n_cases)
    interarrivals = rng.exponential(scale=mean_interarrival, size=n_cases)
    arrival_times = np.cumsum(interarrivals)

    events = []
    for idx, (row_idx, arr_t) in enumerate(zip(sampled_indices, arrival_times), start=1):
        row = scored_df.iloc[int(row_idx)]
        events.append(
            {
                "sim_case_id": f"SIM-{idx:04d}",
                "symptom_text": str(row["symptom_text"]),
                "predicted_urgency": float(row["predicted_urgency"]),
                "confidence": float(row["confidence"]),
                "true_severity": float(row["true_severity"]),
                "arrival_time": float(arr_t),
            }
        )
    return events


def simulate_clinician_feedback(
    is_inversion: bool,
    was_under_flagged: bool,
    noise_rate: float = 0.0,
    over_flag_bias: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> FeedbackLabel:
    """
    Generate ground-truth-informed clinician feedback (with optional noise/bias injection).
    """
    if is_inversion:
        true_label = FeedbackLabel.OVER_FLAGGED
    elif was_under_flagged:
        true_label = FeedbackLabel.UNDER_FLAGGED
    else:
        true_label = FeedbackLabel.CORRECTLY_PRIORITIZED

    if rng is None or (noise_rate <= 0.0 and over_flag_bias <= 0.0):
        return true_label

    # Inject systematic bias (e.g., clinician tends to tag borderline cases as over-flagged)
    if true_label == FeedbackLabel.CORRECTLY_PRIORITIZED and over_flag_bias > 0.0:
        if rng.random() < over_flag_bias:
            return FeedbackLabel.OVER_FLAGGED

    # Inject random inter-clinician label noise
    if noise_rate > 0.0 and rng.random() < noise_rate:
        choices = [
            FeedbackLabel.CORRECTLY_PRIORITIZED,
            FeedbackLabel.OVER_FLAGGED,
            FeedbackLabel.UNDER_FLAGGED,
        ]
        other_choices = [c for c in choices if c != true_label]
        return other_choices[int(rng.integers(0, len(other_choices)))]

    return true_label


def run_single_condition(
    run_id: int,
    condition_name: str,
    arrival_events: List[Dict],
    policy: str = "medqueue",
    recal_mode: RecalibrationMode = "fixed",
    delta: float = 0.08,
    service_duration: float = 6.0,
    window_size: int = 20,
    inversion_tolerance: float = 12.0,
    feedback_noise: float = 0.0,
    feedback_bias: float = 0.0,
    feedback_seed: int = 100,
) -> RunMetrics:
    """
    Execute an exact event-driven simulation of one queue policy over `arrival_events`.
    """
    rng = np.random.default_rng(feedback_seed)

    recalibrator = RecalibrationEngine(
        initial_weights=WeightVector(0.60, 0.30, 0.10),
        window_size=window_size,
        delta=delta,
        delta_max=0.05,
        w_min=0.30,
        w_max=0.90,
        mode=recal_mode,
    )
    pq = AdaptivePriorityQueue(
        recalibrator=recalibrator,
        policy=policy,
        inversion_tolerance=inversion_tolerance,
    )

    n_total = len(arrival_events)
    next_arrival_idx = 0
    server_free_time = 0.0

    high_urgency_waits: List[float] = []
    low_urgency_waits: List[float] = []
    inversion_count = 0
    served_count = 0
    w1_trajectory: List[float] = [recalibrator.weights.w1]

    while served_count < n_total:
        # Advance clock to next arrival if queue is currently empty
        if len(pq) == 0 and next_arrival_idx < n_total:
            next_arr = arrival_events[next_arrival_idx]["arrival_time"]
            if next_arr > server_free_time:
                server_free_time = next_arr

        # Enqueue all patients who arrived on or before current service start time
        while next_arrival_idx < n_total and arrival_events[next_arrival_idx]["arrival_time"] <= server_free_time:
            ev = arrival_events[next_arrival_idx]
            pq.enqueue(
                QueueCase(
                    case_id=ev["sim_case_id"],
                    symptom_text=ev["symptom_text"],
                    urgency_score=ev["predicted_urgency"],
                    confidence=ev["confidence"],
                    arrival_time=ev["arrival_time"],
                    true_severity=ev["true_severity"],
                )
            )
            next_arrival_idx += 1

        # Pop highest-priority patient at `server_free_time`
        popped = pq.pop_next_case(current_time=server_free_time)
        assert popped is not None
        selected_view, is_inversion, over_flag_cause, was_under_flagged = popped

        wait_time = selected_view.elapsed_wait
        true_sev = selected_view.case.true_severity
        assert true_sev is not None

        if true_sev >= 75.0:
            high_urgency_waits.append(wait_time)
        elif true_sev < 40.0:
            low_urgency_waits.append(wait_time)

        if is_inversion:
            inversion_count += 1

        served_count += 1
        resolution_time = server_free_time + service_duration

        # Collect clinician feedback upon case resolution
        if recal_mode != "fixed":
            fb_label = simulate_clinician_feedback(
                is_inversion=is_inversion,
                was_under_flagged=was_under_flagged,
                noise_rate=feedback_noise,
                over_flag_bias=feedback_bias,
                rng=rng,
            )
            # In live operation, over_flag_cause is inferred from ranking terms at service
            event = pq.submit_feedback(
                case_id=selected_view.case.case_id,
                label=fb_label,
                current_time=resolution_time,
                over_flag_cause=None,
            )
            if event is not None:
                w1_trajectory.append(event.updated_weights.w1)

        server_free_time = resolution_time

    return RunMetrics(
        condition=condition_name,
        run_id=run_id,
        high_urgency_avg_wait=float(np.mean(high_urgency_waits)),
        inversion_rate=float(inversion_count / served_count),
        low_urgency_max_wait=float(np.max(low_urgency_waits)),
        low_urgency_p95_wait=float(np.percentile(low_urgency_waits, 95)),
        final_w1=recalibrator.weights.w1,
        final_w2=recalibrator.weights.w2,
        final_w3=recalibrator.weights.w3,
        w1_trajectory=w1_trajectory,
    )


def mean_and_ci95(values: np.ndarray) -> Tuple[float, float]:
    """Compute sample mean and 95% t-distribution half-width confidence interval."""
    n = len(values)
    mean_val = float(np.mean(values))
    if n < 2:
        return mean_val, 0.0
    sem = float(stats.sem(values))
    ci_half = float(sem * stats.t.ppf(0.975, df=n - 1))
    return mean_val, ci_half


def generate_benchmark_figures(
    results_df: pd.DataFrame,
    trajectories: Dict[str, List[List[float]]],
    output_dir: str,
) -> None:
    """Generate publication-quality Matplotlib figures with 95% CI whiskers."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # --- Figure 1: FIFO vs. MedQueue (Fixed Weights) High-Urgency Wait ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    conds_1 = ["FIFO", "MedQueue (Fixed)"]
    means_1, cis_1 = [], []
    for c in conds_1:
        vals = results_df[results_df["condition"] == c]["high_urgency_avg_wait"].to_numpy()
        m, ci = mean_and_ci95(vals)
        means_1.append(m)
        cis_1.append(ci)

    bars = ax.bar(conds_1, means_1, yerr=cis_1, capsize=6, color=["#7f8c8d", "#2e86c1"], width=0.5, edgecolor="black")
    ax.set_ylabel("Avg. Wait Time for High-Urgency Cases (min)")
    ax.set_title("High-Urgency Wait Time: FIFO vs. MedQueue (30 Runs, 95% CI)")
    ax.set_ylim(0, max(means_1) * 1.25)
    for bar, m in zip(bars, means_1):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.5, f"{m:.1f} min", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_fifo_vs_medqueue.png"), dpi=300)
    plt.close(fig)

    # --- Figure 2: Starvation Ablation (Max Low-Urgency Wait) ---
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    conds_2 = ["FIFO", "Urgency-Only (w2=0)", "MedQueue (Fixed)"]
    labels_2 = ["FIFO\n(No Prioritization)", "Urgency-Only\n(No Wait Term)", "MedQueue\n(Full Ranking)"]
    means_2, cis_2 = [], []
    for c in conds_2:
        vals = results_df[results_df["condition"] == c]["low_urgency_max_wait"].to_numpy()
        m, ci = mean_and_ci95(vals)
        means_2.append(m)
        cis_2.append(ci)

    bars = ax.bar(labels_2, means_2, yerr=cis_2, capsize=6, color=["#7f8c8d", "#c0392b", "#27ae60"], width=0.5, edgecolor="black")
    ax.set_ylabel("Max Wait Time, Low-Urgency Cases (min)")
    ax.set_title("Starvation Ablation: Maximum Low-Urgency Wait (30 Runs, 95% CI)")
    ax.set_ylim(0, max(means_2) * 1.25)
    for bar, m, ci in zip(bars, means_2, cis_2):
        ax.text(bar.get_x() + bar.get_width() / 2, m + ci + 10, f"{m:.0f} min", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_starvation_ablation.png"), dpi=300)
    plt.close(fig)

    # --- Figure 3: Baseline Recalibration vs. Refined Option A Recalibration ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    conds_3 = [
        "MedQueue (Fixed)",
        "Baseline Recal (d=0.08)",
        "Refined Option A (d=0.08)",
        "Refined Option A (d=0.25)",
    ]
    short_labels_3 = ["Fixed\nWeights", "Baseline\n(d=0.08)", "Refined A\n(d=0.08)", "Refined A\n(d=0.25)"]
    colors_3 = ["#2e86c1", "#e67e22", "#16a085", "#1abc9c"]

    # Panel (a): High-Urgency Avg Wait
    m_hu, ci_hu = [], []
    for c in conds_3:
        vals = results_df[results_df["condition"] == c]["high_urgency_avg_wait"].to_numpy()
        m, ci = mean_and_ci95(vals)
        m_hu.append(m)
        ci_hu.append(ci)
    axes[0].bar(short_labels_3, m_hu, yerr=ci_hu, capsize=5, color=colors_3, width=0.55, edgecolor="black")
    axes[0].set_ylabel("High-Urgency Avg. Wait (min)")
    axes[0].set_title("(a) High-Urgency Wait Time")
    axes[0].set_ylim(0, max(m_hu) * 1.28)
    for idx, m in enumerate(m_hu):
        axes[0].text(idx, m + ci_hu[idx] + 0.08, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")

    # Panel (b): Inversion (Over-Flag) Rate
    m_inv, ci_inv = [], []
    for c in conds_3:
        vals = results_df[results_df["condition"] == c]["inversion_rate"].to_numpy()
        m, ci = mean_and_ci95(vals)
        m_inv.append(m)
        ci_inv.append(ci)
    axes[1].bar(short_labels_3, m_inv, yerr=ci_inv, capsize=5, color=colors_3, width=0.55, edgecolor="black")
    axes[1].set_ylabel("Inversion (Over-Flag) Rate")
    axes[1].set_title("(b) Urgency Ordering Inversion Rate")
    axes[1].set_ylim(0, max(m_inv) * 1.28)
    for idx, m in enumerate(m_inv):
        axes[1].text(idx, m + ci_inv[idx] + 0.002, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Recalibration Rule Comparison: Baseline Sign Flaw vs. Root-Cause-Aware Option A", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_recalibration_comparison.png"), dpi=300)
    plt.close(fig)

    # --- Figure 4: Mean w1 Weight Trajectory Over 100 Review Windows ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    traj_styles = {
        "Baseline Recal (d=0.08)": ("#e67e22", "--", "Baseline Paper Eq. 3 (d=0.08)"),
        "Refined Option A (d=0.08)": ("#16a085", "-", "Refined Option A (d=0.08)"),
        "Refined Option A (d=0.25)": ("#1f618d", "-", "Refined Option A (d=0.25)"),
        "Refined Option A + Noise/Bias": ("#8e44ad", "-.", "Refined Option A + 15% Noise/Bias"),
    }
    for cond_key, (color, ls, label) in traj_styles.items():
        if cond_key in trajectories and trajectories[cond_key]:
            min_len = min(len(t) for t in trajectories[cond_key])
            mat = np.array([t[:min_len] for t in trajectories[cond_key]])
            mean_curve = np.mean(mat, axis=0)
            ax.plot(range(min_len), mean_curve, color=color, linestyle=ls, linewidth=2.0, label=label)

    ax.axhline(0.60, color="gray", linestyle=":", linewidth=1.2, label="Initial w1 (0.60)")
    ax.set_xlabel("Review Window (k = 20 resolved cases per window)")
    ax.set_ylabel("Urgency Weight w1")
    ax.set_title("Urgency Weight (w1) Evolution Across 100 Review Windows (Mean of 30 Runs)")
    ax.set_ylim(0.28, 0.92)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_weight_trajectory.png"), dpi=300)
    plt.close(fig)


def run_full_benchmark_suite(
    scored_csv_path: str = os.path.join("data", "scored_synthetic_cases.csv"),
    output_dir: str = "outputs",
    n_runs: int = 30,
    cases_per_run: int = 2000,
    master_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all 7 experimental conditions across 30 paired Monte Carlo runs."""
    if not os.path.exists(scored_csv_path):
        raise FileNotFoundError(
            f"Out-of-fold scored dataset not found at {scored_csv_path}. Run 'python -m core.scorer' first."
        )

    scored_df = pd.read_csv(scored_csv_path)
    master_rng = np.random.default_rng(master_seed)

    conditions = [
        {"name": "FIFO", "policy": "fifo", "recal": "fixed", "delta": 0.0, "noise": 0.0, "bias": 0.0},
        {"name": "Urgency-Only (w2=0)", "policy": "urgency_only", "recal": "fixed", "delta": 0.0, "noise": 0.0, "bias": 0.0},
        {"name": "MedQueue (Fixed)", "policy": "medqueue", "recal": "fixed", "delta": 0.0, "noise": 0.0, "bias": 0.0},
        {"name": "Baseline Recal (d=0.08)", "policy": "medqueue", "recal": "baseline_paper", "delta": 0.08, "noise": 0.0, "bias": 0.0},
        {"name": "Refined Option A (d=0.08)", "policy": "medqueue", "recal": "root_cause_aware", "delta": 0.08, "noise": 0.0, "bias": 0.0},
        {"name": "Refined Option A (d=0.25)", "policy": "medqueue", "recal": "root_cause_aware", "delta": 0.25, "noise": 0.0, "bias": 0.0},
        {"name": "Refined Option A + Noise/Bias", "policy": "medqueue", "recal": "root_cause_aware", "delta": 0.08, "noise": 0.15, "bias": 0.05},
    ]

    all_records: List[Dict] = []
    trajectories: Dict[str, List[List[float]]] = {c["name"]: [] for c in conditions}

    print(f"Running {n_runs} paired discrete-event simulation runs x {cases_per_run} cases across {len(conditions)} conditions...")

    for run_idx in range(1, n_runs + 1):
        run_seed = int(master_rng.integers(1, 1_000_000))
        run_rng = np.random.default_rng(run_seed)
        arrivals = generate_run_arrivals(
            scored_df=scored_df,
            n_cases=cases_per_run,
            mean_interarrival=7.5,
            rng=run_rng,
        )

        for cond in conditions:
            metrics = run_single_condition(
                run_id=run_idx,
                condition_name=cond["name"],
                arrival_events=arrivals,
                policy=cond["policy"],
                recal_mode=cond["recal"],
                delta=cond["delta"],
                service_duration=6.0,
                window_size=20,
                inversion_tolerance=12.0,
                feedback_noise=cond["noise"],
                feedback_bias=cond["bias"],
                feedback_seed=run_seed + 999,
            )
            trajectories[cond["name"]].append(metrics.w1_trajectory)
            all_records.append(
                {
                    "run_id": metrics.run_id,
                    "condition": metrics.condition,
                    "high_urgency_avg_wait": metrics.high_urgency_avg_wait,
                    "inversion_rate": metrics.inversion_rate,
                    "low_urgency_max_wait": metrics.low_urgency_max_wait,
                    "low_urgency_p95_wait": metrics.low_urgency_p95_wait,
                    "final_w1": metrics.final_w1,
                    "final_w2": metrics.final_w2,
                    "final_w3": metrics.final_w3,
                }
            )

    results_df = pd.DataFrame(all_records)

    # Build aggregate summary table (Mean +/- 95% CI)
    summary_rows = []
    for cond in conditions:
        cname = cond["name"]
        sub = results_df[results_df["condition"] == cname]
        hu_m, hu_ci = mean_and_ci95(sub["high_urgency_avg_wait"].to_numpy())
        inv_m, inv_ci = mean_and_ci95(sub["inversion_rate"].to_numpy())
        max_m, max_ci = mean_and_ci95(sub["low_urgency_max_wait"].to_numpy())
        p95_m, p95_ci = mean_and_ci95(sub["low_urgency_p95_wait"].to_numpy())
        w1_m, _ = mean_and_ci95(sub["final_w1"].to_numpy())

        summary_rows.append(
            {
                "Condition": cname,
                "High_Urgency_Wait_Min": f"{hu_m:.2f} +/- {hu_ci:.2f}",
                "Inversion_Rate": f"{inv_m:.4f} +/- {inv_ci:.4f}",
                "Low_Urgency_Max_Wait_Min": f"{max_m:.1f} +/- {max_ci:.1f}",
                "Low_Urgency_P95_Wait_Min": f"{p95_m:.1f} +/- {p95_ci:.1f}",
                "Mean_Final_w1": round(w1_m, 3),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    # Build paired difference table vs. MedQueue (Fixed)
    fixed_sub = results_df[results_df["condition"] == "MedQueue (Fixed)"].sort_values("run_id").reset_index(drop=True)
    paired_rows = []
    for cname in [
        "Baseline Recal (d=0.08)",
        "Refined Option A (d=0.08)",
        "Refined Option A (d=0.25)",
        "Refined Option A + Noise/Bias",
    ]:
        comp_sub = results_df[results_df["condition"] == cname].sort_values("run_id").reset_index(drop=True)
        d_hu_m, d_hu_ci = mean_and_ci95((comp_sub["high_urgency_avg_wait"] - fixed_sub["high_urgency_avg_wait"]).to_numpy())
        d_inv_m, d_inv_ci = mean_and_ci95((comp_sub["inversion_rate"] - fixed_sub["inversion_rate"]).to_numpy())
        d_p95_m, d_p95_ci = mean_and_ci95((comp_sub["low_urgency_p95_wait"] - fixed_sub["low_urgency_p95_wait"]).to_numpy())
        w1_m = float(comp_sub["final_w1"].mean())

        paired_rows.append(
            {
                "Recalibration_Condition": cname,
                "Delta_High_Urgency_Wait": f"{d_hu_m:+.2f} +/- {d_hu_ci:.2f}",
                "Delta_Inversion_Rate": f"{d_inv_m:+.4f} +/- {d_inv_ci:.4f}",
                "Delta_Low_Urgency_P95": f"{d_p95_m:+.1f} +/- {d_p95_ci:.1f}",
                "Mean_Final_w1": round(w1_m, 3),
            }
        )
    paired_df = pd.DataFrame(paired_rows)

    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "raw_30run_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)
    paired_df.to_csv(os.path.join(output_dir, "paired_recalibration_comparison.csv"), index=False)

    # Save JSON summary for backend Analytics dashboard endpoint
    with open(os.path.join(output_dir, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_runs": n_runs,
                "cases_per_run": cases_per_run,
                "summary_table": summary_rows,
                "paired_recalibration_table": paired_rows,
            },
            f,
            indent=2,
        )

    generate_benchmark_figures(results_df, trajectories, output_dir)
    return summary_df, paired_df


if __name__ == "__main__":
    summary_table, paired_table = run_full_benchmark_suite()
    print("\n=== TABLE 1: AGGREGATE BENCHMARK RESULTS (30 RUNS x 2000 CASES, MEAN +/- 95% CI) ===")
    print(summary_table.to_string(index=False))
    print("\n=== TABLE 2: PAIRED DIFFERENCE VS. FIXED-WEIGHT MEDQUEUE ===")
    print(paired_table.to_string(index=False))
    print("\nSaved CSV tables and 4 high-resolution PNG figures into outputs/ directory.")