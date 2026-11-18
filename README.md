# MedQueue: Clinician-Feedback-Driven Recalibration for Tele-Consultation Priority Queueing

**MedQueue** is an adaptive, starvation-preventing priority queueing system and research benchmark suite for asynchronous tele-consultation platforms. It combines lightweight urgency and uncertainty estimation, analytic token-level explainability (XAI), monotonic wait-time escalation, and bounded batch weight recalibration driven by structured clinician feedback.

> **Non-Diagnostic Scope Notice:** MedQueue is strictly a queue prioritization and decision-support prototype. It does **not** perform medical diagnosis or autonomously reject care; licensed clinicians retain full override authority over queue order.

---

## Core Architecture

```text
MedQueue/
├── core/                  # Shared ML scorer, token XAI, priority queue & recalibrator
│   ├── dataset.py         # ESI-inspired 5-tier synthetic dataset generator (N = 400)
│   ├── scorer.py          # TF-IDF (1,2)-grams + 15-estimator Bagged Ridge + token XAI
│   ├── queue_engine.py    # Composite linear ranking R_i(t) with tie-breaking & override
│   └── recalibrator.py    # Bounded batch recalibration (Baseline Eq. 3 & Refined Option A)
├── simulation/            # 30-run x 2,000-case discrete-event Monte Carlo benchmark suite
│   └── run_benchmarks.py  # Generates statistical tables & 4 publication-grade figures
├── backend/               # FastAPI REST API + SQLite persistence (medqueue.db)
├── frontend/              # Responsive React + Vite + Tailwind CSS web application
└── tests/                 # Automated Pytest unit & API integration suite