"""
MedQueue Core Package.

Shared algorithmic modules used identically by both the discrete-event
simulation harness and the live FastAPI tele-consultation backend:
- dataset: ESI-style synthetic symptom-severity reference generator
- scorer: TF-IDF + 15-estimator Bagged Ridge urgency/confidence engine + XAI
- queue_engine: Starvation-preventing composite priority queue
- recalibrator: Bounded clinician-feedback weight recalibration engine
"""

__version__ = "1.0.0"