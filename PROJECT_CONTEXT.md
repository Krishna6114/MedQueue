# PROJECT_CONTEXT.md — MedQueue Master Engineering State

## 1. Project Overview
- **Title:** MedQueue: A Clinician-Feedback-Driven Recalibration System for Tele-Consultation Priority Queueing
- **Authors:** Ishi Jain, Ishan Agarwal, Kirtika Sharma, Krishna Bansal (Supervisor: Dr. Anuj Kumar, AKGEC Ghaziabad)
- **Domain:** Asynchronous Tele-Consultation Triage, Priority Queueing, Human-in-the-Loop ML, Explainable AI (XAI)
- **Strict Scope Constraint:** Queue prioritization and decision support ONLY. Never performs clinical diagnosis; clinician retains full override authority.

## 2. Finalized Technology Stack
- **Core Engine & Simulation:** Python 3.14, NumPy, SciPy, scikit-learn, Pandas, Matplotlib
- **Backend API & Storage:** FastAPI, Uvicorn, Pydantic v2, SQLAlchemy + SQLite (`medqueue.db`)
- **Frontend UI:** React (Vite) + Tailwind CSS v4 (Patient Intake, Doctor Dashboard, Research Lab)
- **Testing:** Pytest + HTTPX / FastAPI TestClient

## 3. Verified 30-Run x 2,000-Case Simulation Benchmarks (Mean +/- 95% CI)
- **FIFO:** High-urgency wait `11.44 +/- 0.81 min`, Inversion rate `0.2958 +/- 0.0119`, Low-urgency max wait `67.1 +/- 5.0 min`
- **Urgency-Only (w2=0):** High-urgency wait `3.01 +/- 0.05 min`, Inversion rate `0.0040 +/- 0.0006`, Low-urgency max wait `434.5 +/- 67.0 min` (severe starvation)
- **MedQueue (Fixed 0.60, 0.30, 0.10):** High-urgency wait `3.17 +/- 0.13 min` (-72.3% vs FIFO), Inversion rate `0.0495 +/- 0.0073`, Low-urgency max wait `126.8 +/- 6.4 min` (-70.8% vs Urgency-Only)
- **Baseline Recalibration (Paper Eq. 3, d=0.08):** `w1` drifted to `0.544`, High-urgency wait `3.50 +/- 0.47 min`, Inversion rate `0.0593 +/- 0.0142`
- **Refined Option A (Root-Cause-Aware, d=0.08):** `w1` adapted to `0.820`, High-urgency wait `3.02 +/- 0.06 min` (`-0.15 +/- 0.09 min` vs Fixed), Inversion rate `0.0239 +/- 0.0022` (`-0.0256 +/- 0.0055` vs Fixed), Low-urgency max wait bounded at `180.5 +/- 21.2 min`
- **Refined Option A + 15% Noise / 5% Bias:** `w1 = 0.645`, Inversion rate `0.0437 +/- 0.0038`, Low-urgency max wait `138.7 +/- 12.3 min`

## 4. Development Status
- **All 8 Implementation Stages Completed & Verified:**
  - Step 1: Workspace initialization & virtual environment
  - Step 2: ESI-inspired synthetic dataset generator (`core/dataset.py`)
  - Step 3: TF-IDF + 15-estimator Bagged Ridge scorer & token XAI (`core/scorer.py`)
  - Step 4: Adaptive Priority Queue & Option A Recalibration Engine (`core/queue_engine.py`, `core/recalibrator.py`)
  - Step 5: 30-run discrete-event simulation & publication figures (`simulation/run_benchmarks.py`)
  - Step 6: FastAPI REST backend & SQLite persistence (`backend/`)
  - Step 7: Responsive React + Tailwind web application (`frontend/`)
  - Step 8: Automated Pytest regression suite (`tests/test_medqueue.py`) & `README.md`