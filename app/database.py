"""
app/database.py
SQLAlchemy models and database initialization for the Face Tracking System.
"""
import os
import json
import logging
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, Boolean, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

logger = logging.getLogger(__name__)

Base = declarative_base()


class Visitor(Base):
    """Represents a unique visitor detected by the face tracking system."""
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id = Column(String(64), unique=True, nullable=False, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    visit_count = Column(Integer, default=1)
    crop_image_path = Column(String(512), nullable=True)
    age = Column(Float, nullable=True)
    gender = Column(String(16), nullable=True)
    gender_confidence = Column(Float, nullable=True)
    recognition_confidence = Column(Float, nullable=True)
    embedding = Column(Text, nullable=True)  # JSON-serialized 512-d vector

    __table_args__ = (
        Index("idx_first_seen", "first_seen"),
        Index("idx_last_seen", "last_seen"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "visitor_id": self.visitor_id,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "visit_count": self.visit_count,
            "crop_image_path": self.crop_image_path,
            "age": self.age,
            "gender": self.gender,
            "gender_confidence": self.gender_confidence,
            "recognition_confidence": self.recognition_confidence,
        }


class EventLog(Base):
    """Logs every detection event (including returning visitors)."""
    __tablename__ = "event_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    visitor_id = Column(String(64), nullable=False, index=True)
    event_type = Column(String(32), nullable=False)  # 'new', 'returning', 'unknown'
    track_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    frame_number = Column(Integer, nullable=True)
    bbox_x = Column(Float, nullable=True)
    bbox_y = Column(Float, nullable=True)
    bbox_w = Column(Float, nullable=True)
    bbox_h = Column(Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "visitor_id": self.visitor_id,
            "event_type": self.event_type,
            "track_id": self.track_id,
            "confidence": self.confidence,
            "frame_number": self.frame_number,
        }


# ── Database Singleton ─────────────────────────────────────────────────────────

_engine = None
_Session = None


def init_db(config: dict = None) -> scoped_session:
    """Initialize the database engine and create all tables.

    Args:
        config: Parsed config.json dict. If None, uses defaults.

    Returns:
        A thread-safe scoped session factory.
    """
    global _engine, _Session

    if config is None:
        db_path = "database/visitors.db"
        echo = False
    else:
        db_path = config.get("database", {}).get("path", "database/visitors.db")
        echo = config.get("database", {}).get("echo", False)

    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    db_url = f"sqlite:///{db_path}"
    _engine = create_engine(
        db_url,
        echo=echo,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    Base.metadata.create_all(_engine)
    _Session = scoped_session(sessionmaker(bind=_engine))

    logger.info(f"Database initialized at {db_path}")
    return _Session


def get_session() -> scoped_session:
    """Return the active scoped session (init_db must be called first)."""
    if _Session is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _Session


def get_all_visitors(session):
    """Return all visitors ordered by first_seen descending."""
    return session.query(Visitor).order_by(Visitor.first_seen.desc()).all()


def get_visitor_by_id(session, visitor_id: str):
    """Return a single visitor by their unique ID."""
    return session.query(Visitor).filter_by(visitor_id=visitor_id).first()


def get_recent_events(session, limit: int = 200):
    """Return the most recent event log entries."""
    return (
        session.query(EventLog)
        .order_by(EventLog.timestamp.desc())
        .limit(limit)
        .all()
    )


def get_hourly_traffic(session):
    """Aggregate visitor counts by hour for chart data."""
    from sqlalchemy import func, extract

    results = (
        session.query(
            func.strftime("%H", EventLog.timestamp).label("hour"),
            func.count(EventLog.id).label("count"),
        )
        .filter(EventLog.event_type == "new")
        .group_by("hour")
        .all()
    )
    hourly = {str(i).zfill(2): 0 for i in range(24)}
    for row in results:
        if row.hour:
            hourly[row.hour] = row.count
    return hourly


def get_gender_distribution(session):
    """Return gender counts for pie chart."""
    from sqlalchemy import func

    results = (
        session.query(Visitor.gender, func.count(Visitor.id).label("count"))
        .filter(Visitor.gender.isnot(None))
        .group_by(Visitor.gender)
        .all()
    )
    return {row.gender: row.count for row in results}


def get_age_distribution(session):
    """Return age-bucket counts for bar chart."""
    from sqlalchemy import func

    buckets = {"0-18": 0, "19-30": 0, "31-45": 0, "46-60": 0, "60+": 0}
    visitors = session.query(Visitor.age).filter(Visitor.age.isnot(None)).all()
    for (age,) in visitors:
        if age <= 18:
            buckets["0-18"] += 1
        elif age <= 30:
            buckets["19-30"] += 1
        elif age <= 45:
            buckets["31-45"] += 1
        elif age <= 60:
            buckets["46-60"] += 1
        else:
            buckets["60+"] += 1
    return buckets
