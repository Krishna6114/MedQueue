"""
Synthetic ESI-Style Symptom-Severity Dataset Generator for MedQueue.

Generates N=400 synthetic tele-consultation patient cases grounded in an
illustrative Emergency Severity Index (ESI) 5-level ordering:
  - ESI-1 (Immediate / Life-Threatening): Base severity 90-98
  - ESI-2 (Emergent / High Risk):         Base severity 76-89  (High-Urgency >= 75)
  - ESI-3 (Urgent / Moderate):            Base severity 50-74
  - ESI-4 (Less Urgent):                  Base severity 26-39  (Low-Urgency < 40)
  - ESI-5 (Non-Urgent / Routine):         Base severity 8-25   (Low-Urgency < 40)

Note: This reference table is illustrative for algorithmic benchmarking and
is NOT a validated clinical triage instrument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SymptomTemplate:
    """Represents a structured clinical presentation template."""
    esi_level: int
    category: str
    base_severity: float
    primary_symptoms: Tuple[str, ...]
    modifiers: Tuple[str, ...]
    durations: Tuple[str, ...]


# Illustrative ESI-ordered clinical symptom reference table
SYMPTOM_REFERENCE_TABLE: List[SymptomTemplate] = [
    # --- ESI Level 1: Immediate / Life-Threatening (Severity 90 - 98) ---
    SymptomTemplate(
        esi_level=1,
        category="Acute Cardiopulmonary Distress",
        base_severity=95.0,
        primary_symptoms=(
            "crushing chest pain radiating to left arm and jaw",
            "severe chest tightness with cold sweats and dizziness",
            "sudden severe shortness of breath unable to speak full sentences",
            "acute chest pressure with nausea and near fainting",
        ),
        modifiers=(
            "worsening rapidly",
            "accompanied by profuse sweating and pallor",
            "with palpitations and lightheadedness",
            "occurring at rest",
        ),
        durations=(
            "started 15 minutes ago",
            "for the past 30 minutes",
            "began suddenly 10 minutes ago",
            "worsening over the last hour",
        ),
    ),
    SymptomTemplate(
        esi_level=1,
        category="Acute Neurological Emergency",
        base_severity=94.0,
        primary_symptoms=(
            "sudden slurred speech and weakness on right side of face and arm",
            "acute facial droop with confusion and loss of balance",
            "sudden worst headache of life with blurred vision and vomiting",
            "uncontrolled active seizure episode followed by confusion",
        ),
        modifiers=(
            "onset was abrupt",
            "patient struggling to stay awake",
            "with numbness in left arm and leg",
            "never experienced before",
        ),
        durations=(
            "started 20 minutes ago",
            "for the past 25 minutes",
            "noticed 15 minutes ago",
            "began 40 minutes ago",
        ),
    ),
    SymptomTemplate(
        esi_level=1,
        category="Severe Anaphylaxis / Hemorrhage",
        base_severity=92.0,
        primary_symptoms=(
            "rapid tongue and throat swelling with wheezing after medication",
            "vomiting bright red blood with severe dizziness",
            "heavy uncontrolled bleeding from deep laceration",
            "severe allergic reaction with hives and difficulty swallowing",
        ),
        modifiers=(
            "feeling faint when standing",
            "breathing becoming noisy",
            "heart racing rapidly",
            "weak and clammy skin",
        ),
        durations=(
            "started 10 minutes ago",
            "over the past 20 minutes",
            "for 15 minutes",
            "began 30 minutes ago",
        ),
    ),

    # --- ESI Level 2: Emergent / High Risk (Severity 76 - 88; >= 75 High Urgency) ---
    SymptomTemplate(
        esi_level=2,
        category="High-Risk Febrile / Sepsis / Abdominal",
        base_severity=82.0,
        primary_symptoms=(
            "high fever 104F with rigors confusion and rapid breathing",
            "severe sharp lower right abdominal pain with persistent vomiting",
            "persistent chest discomfort on exertion with shortness of breath",
            "acute severe flank pain radiating to groin with blood in urine",
        ),
        modifiers=(
            "unable to keep fluids down",
            "pain rated 9 out of 10",
            "feeling extremely weak and lethargic",
            "fever not responding to paracetamol",
        ),
        durations=(
            "for the past 4 hours",
            "since last night",
            "worsening over 6 hours",
            "started 3 hours ago",
        ),
    ),
    SymptomTemplate(
        esi_level=2,
        category="Acute Respiratory / Metabolic Exacerbation",
        base_severity=79.0,
        primary_symptoms=(
            "acute asthma flare with audible wheezing and chest tightness",
            "diabetic patient with persistent vomiting abdominal pain and drowsiness",
            "sudden severe eye pain with blurred vision and halos around lights",
            "pregnant patient with severe headache swelling and blurred vision",
        ),
        modifiers=(
            "rescue inhaler giving minimal relief",
            "breathing faster than normal",
            "marked fatigue and dry mouth",
            "symptoms escalating steadily",
        ),
        durations=(
            "for 2 hours",
            "since early morning",
            "over the past 5 hours",
            "started 3 hours ago",
        ),
    ),

    # --- ESI Level 3: Urgent / Moderate Severity (Severity 52 - 70) ---
    SymptomTemplate(
        esi_level=3,
        category="Moderate Infection / Acute Pain",
        base_severity=62.0,
        primary_symptoms=(
            "moderate fever 101.5F with productive cough and body aches",
            "painful urination with lower abdominal discomfort and mild fever",
            "moderate abdominal cramps with repeated watery diarrhea",
            "painful swollen ankle after twisting injury unable to bear weight",
        ),
        modifiers=(
            "feeling fatigued but alert and breathing comfortably",
            "able to drink fluids slowly",
            "no chest pain or shortness of breath",
            "moderate discomfort interfering with sleep",
        ),
        durations=(
            "for 2 days",
            "since yesterday",
            "over the past 36 hours",
            "for 24 hours",
        ),
    ),
    SymptomTemplate(
        esi_level=3,
        category="Subacute Clinical Evaluation",
        base_severity=56.0,
        primary_symptoms=(
            "persistent ear pain with mild fever and muffled hearing",
            "localized red warm skin swelling on forearm",
            "moderate migraine headache with nausea and light sensitivity",
            "persistent dry cough with mild wheeze in known asthmatic",
        ),
        modifiers=(
            "no confusion or neck stiffness",
            "vital signs feel stable",
            "mild relief with rest",
            "gradual onset",
        ),
        durations=(
            "for 2 days",
            "since yesterday evening",
            "for 3 days",
            "over the last 48 hours",
        ),
    ),

    # --- ESI Level 4: Less Urgent (Severity 28 - 37; < 40 Low Urgency) ---
    SymptomTemplate(
        esi_level=4,
        category="Minor Acute Ailment",
        base_severity=33.0,
        primary_symptoms=(
            "mild sore throat with runny nose and occasional sneezing",
            "mild lower back stiffness after lifting boxes",
            "itchy localized skin rash on arm without fever or swelling",
            "mild acid reflux and bloating after heavy meal",
        ),
        modifiers=(
            "no fever or breathing difficulty",
            "able to eat drink and walk normally",
            "mild discomfort only",
            "otherwise feeling well",
        ),
        durations=(
            "for 2 days",
            "for 3 days",
            "since yesterday",
            "for the past 4 days",
        ),
    ),
    SymptomTemplate(
        esi_level=4,
        category="Mild Self-Limiting Symptoms",
        base_severity=29.0,
        primary_symptoms=(
            "mild tension headache relieved partially by rest",
            "minor superficial scrape on knee needing dressing advice",
            "mild seasonal nasal congestion and watery eyes",
            "mild dry cough without fever or chest pain",
        ),
        modifiers=(
            "no red flag symptoms",
            "normal appetite and energy",
            "very mild intensity",
            "stable since onset",
        ),
        durations=(
            "for 3 days",
            "for 5 days",
            "since two days ago",
            "for 4 days",
        ),
    ),

    # --- ESI Level 5: Non-Urgent / Routine (Severity 10 - 22; < 40 Low Urgency) ---
    SymptomTemplate(
        esi_level=5,
        category="Routine Administrative / Follow-up",
        base_severity=16.0,
        primary_symptoms=(
            "routine prescription refill request for stable blood pressure medication",
            "follow up review of normal routine blood test report",
            "request for general diet and lifestyle advice for mild cholesterol",
            "routine medication renewal for well controlled thyroid condition",
        ),
        modifiers=(
            "feeling completely well with no new symptoms",
            "no complaints or discomfort",
            "for routine continuation only",
            "asymptomatic and stable",
        ),
        durations=(
            "scheduled monthly check",
            "ongoing routine care",
            "no acute symptoms",
            "since last month",
        ),
    ),
    SymptomTemplate(
        esi_level=5,
        category="Minor Chronic / Cosmetic Inquiry",
        base_severity=12.0,
        primary_symptoms=(
            "mild dry skin on hands during weather change",
            "general question about multivitamin supplementation",
            "mild dandruff and scalp dryness",
            "routine certificate renewal for fitness checkup",
        ),
        modifiers=(
            "no pain fever or systemic symptoms",
            "completely stable",
            "non urgent inquiry",
            "feeling healthy overall",
        ),
        durations=(
            "for 2 weeks",
            "for several weeks",
            "routine query",
            "for 1 month",
        ),
    ),
]

# Balanced case allocation across ESI tiers to total exactly N = 400 cases:
# Ensures realistic mix of High-Urgency (>=75: ~25%), Moderate (40-74: ~30%), Low-Urgency (<40: ~45%)
ESI_TIER_COUNTS: Dict[int, int] = {
    1: 40,   # 10.0% Immediate (Severity ~90-98)
    2: 60,   # 15.0% Emergent  (Severity ~76-88) -> Total High-Urgency (>=75) = 25.0%
    3: 120,  # 30.0% Urgent    (Severity ~52-70)
    4: 100,  # 25.0% Less Urgent (Severity ~28-37)
    5: 80,   # 20.0% Non-Urgent  (Severity ~10-22) -> Total Low-Urgency (<40) = 45.0%
}


def generate_synthetic_dataset(
    n_cases: int = 400,
    seed: int = 42,
    noise_std: float = 2.5,
    output_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate N synthetic patient cases from the ESI-style symptom reference table.

    Parameters
    ----------
    n_cases : int
        Total number of synthetic cases to generate (default 400 per Synopsis/Paper).
    seed : int
        Random seed for exact reproducibility.
    noise_std : float
        Standard deviation of Gaussian severity jitter so identical symptom
        combinations do not map deterministically to identical scores.
    output_csv_path : str | None
        Optional filepath to save the generated CSV dataset.

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'esi_level', 'category', 'symptom_text', 'true_severity', 'urgency_tier']
    """
    if n_cases != sum(ESI_TIER_COUNTS.values()):
        raise ValueError(
            f"n_cases ({n_cases}) must match sum of ESI_TIER_COUNTS ({sum(ESI_TIER_COUNTS.values())})."
        )

    rng = np.random.default_rng(seed)
    templates_by_esi: Dict[int, List[SymptomTemplate]] = {lvl: [] for lvl in range(1, 6)}
    for tmpl in SYMPTOM_REFERENCE_TABLE:
        templates_by_esi[tmpl.esi_level].append(tmpl)

    records = []
    case_idx = 1

    for esi_level, count in ESI_TIER_COUNTS.items():
        tier_templates = templates_by_esi[esi_level]
        for _ in range(count):
            tmpl = tier_templates[int(rng.integers(0, len(tier_templates)))]
            primary = tmpl.primary_symptoms[int(rng.integers(0, len(tmpl.primary_symptoms)))]
            modifier = tmpl.modifiers[int(rng.integers(0, len(tmpl.modifiers)))]
            duration = tmpl.durations[int(rng.integers(0, len(tmpl.durations)))]

            # Construct natural free-text patient symptom submission
            pattern_choice = int(rng.integers(0, 3))
            if pattern_choice == 0:
                symptom_text = f"{primary}, {modifier}, {duration}."
            elif pattern_choice == 1:
                symptom_text = f"Patient reports {primary} ({duration}); {modifier}."
            else:
                symptom_text = f"{primary} {duration}. Notes: {modifier}."

            # Add controlled stochastic variation and clamp strictly within ESI tier bounds
            jitter = float(rng.normal(0.0, noise_std))
            raw_severity = tmpl.base_severity + jitter

            if esi_level == 1:
                true_severity = float(np.clip(raw_severity, 88.0, 99.0))
            elif esi_level == 2:
                true_severity = float(np.clip(raw_severity, 75.0, 87.9))
            elif esi_level == 3:
                true_severity = float(np.clip(raw_severity, 45.0, 74.0))
            elif esi_level == 4:
                true_severity = float(np.clip(raw_severity, 24.0, 39.5))
            else:
                true_severity = float(np.clip(raw_severity, 5.0, 23.5))

            if true_severity >= 75.0:
                urgency_tier = "High"
            elif true_severity < 40.0:
                urgency_tier = "Low"
            else:
                urgency_tier = "Moderate"

            records.append(
                {
                    "case_id": f"CASE-{case_idx:04d}",
                    "esi_level": esi_level,
                    "category": tmpl.category,
                    "symptom_text": symptom_text,
                    "true_severity": round(true_severity, 2),
                    "urgency_tier": urgency_tier,
                }
            )
            case_idx += 1

    # Shuffle deterministically so ESI levels are interleaved naturally
    df = pd.DataFrame(records)
    shuffle_idx = rng.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)

    return df


if __name__ == "__main__":
    default_path = os.path.join("data", "synthetic_cases.csv")
    dataset_df = generate_synthetic_dataset(n_cases=400, seed=42, output_csv_path=default_path)
    print(f"Generated {len(dataset_df)} synthetic cases -> {default_path}")
    print("\nTier Distribution:")
    print(dataset_df["urgency_tier"].value_counts().to_string())
    print("\nSeverity Summary by ESI Level:")
    print(dataset_df.groupby("esi_level")["true_severity"].agg(["count", "mean", "min", "max"]).to_string())