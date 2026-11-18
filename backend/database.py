"""
SQLAlchemy SQLite Database Configuration and ORM Models for MedQueue.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medqueue.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PatientCaseModel(Base):
    """Persistent record of a tele-consultation case and its lifecycle."""
    __tablename__ = "patient_cases"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(32), unique=True, index=True, nullable=False)
    patient_name = Column(String(120), nullable=False, default="Anonymous Patient")
    patient_age = Column(Integer, nullable=True)
    symptom_text = Column(Text, nullable=False)

    # Scorer & XAI outputs
    urgency_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    prediction_std = Column(Float, nullable=False, default=0.0)
    explanation_summary = Column(Text, nullable=False, default="")
    top_contributors_json = Column(Text, nullable=False, default="[]")

    # Queue timing (measured in logical queue minutes for seamless real-time + demo support)
    arrival_minute = Column(Float, nullable=False)
    service_start_minute = Column(Float, nullable=True)
    resolution_minute = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Status: 'waiting' | 'in_consultation' | 'resolved'
    status = Column(String(24), index=True, nullable=False, default="waiting")
    clinician_override = Column(Boolean, nullable=False, default=False)

    # Clinician Feedback at resolution
    feedback_label = Column(String(32), nullable=True)
    over_flag_cause = Column(String(32), nullable=True)
    clinician_note = Column(Text, nullable=True)
    true_severity = Column(Float, nullable=True)  # Optional ground truth when seeded from synthetic benchmark


class WeightHistoryModel(Base):
    """Persistent audit log of batch recalibration updates to (w1, w2, w3)."""
    __tablename__ = "weight_history"

    id = Column(Integer, primary_key=True, index=True)
    cycle_index = Column(Integer, nullable=False)
    window_size = Column(Integer, nullable=False)
    recalibration_mode = Column(String(32), nullable=False)

    p_correct = Column(Float, nullable=False)
    p_over = Column(Float, nullable=False)
    p_under = Column(Float, nullable=False)
    p_over_wait = Column(Float, nullable=False)
    p_over_score = Column(Float, nullable=False)

    raw_delta_w1 = Column(Float, nullable=False)
    applied_delta_w1 = Column(Float, nullable=False)

    prev_w1 = Column(Float, nullable=False)
    prev_w2 = Column(Float, nullable=False)
    prev_w3 = Column(Float, nullable=False)

    new_w1 = Column(Float, nullable=False)
    new_w2 = Column(Float, nullable=False)
    new_w3 = Column(Float, nullable=False)

    queue_minute = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)


def init_db() -> None:
    """Create database tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)