"""
app/visitor_manager.py
Visitor re-identification using Cosine Similarity on ArcFace embeddings.

Manages an in-memory gallery of known visitors and persists them to SQLite.
"""
import os
import json
import uuid
import logging
from datetime import datetime

import numpy as np
import cv2

from .database import Visitor, EventLog, get_session

logger = logging.getLogger(__name__)


class VisitorManager:
    """
    Maintains a gallery of known visitor embeddings and performs
    cosine-similarity re-identification.

    Algorithm:
        1. Compare incoming embedding against all gallery embeddings.
        2. If max similarity > threshold → returning visitor.
        3. Else → new visitor; add to gallery and DB.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        rec_cfg = self.config.get("recognition", {})
        self.threshold = rec_cfg.get("similarity_threshold", 0.6)
        self.max_visitors = self.config.get("pipeline", {}).get("max_visitors", 10000)

        # Gallery: visitor_id → {"embedding": np.array, "track_id": int}
        self._gallery: dict[str, dict] = {}
        # Track ID → visitor_id (fast lookup for same-session tracks)
        self._track_map: dict[int, str] = {}

        self._load_gallery_from_db()

    # ── Gallery Management ────────────────────────────────────────────────────

    def _load_gallery_from_db(self):
        """Warm-start the gallery from persisted embeddings in SQLite."""
        try:
            session = get_session()
            visitors = session.query(Visitor).filter(
                Visitor.embedding.isnot(None)
            ).all()
            for v in visitors:
                emb = json.loads(v.embedding)
                self._gallery[v.visitor_id] = {
                    "embedding": np.array(emb, dtype=np.float32),
                    "track_id": None,
                }
            logger.info(f"Gallery loaded with {len(self._gallery)} visitors from DB.")
        except Exception as e:
            logger.warning(f"Could not load gallery from DB: {e}")

    def identify(
        self,
        embedding: np.ndarray,
        track_id: int,
        confidence: float,
        crop: np.ndarray,
        demographics: dict,
        frame_number: int,
        snapshot_dir: str = "logs/snapshots",
    ) -> dict:
        """
        Identify or register a visitor given a face embedding.

        Args:
            embedding:     512-d normalised face embedding.
            track_id:      ByteTrack persistent track ID.
            confidence:    Detection confidence score.
            crop:          BGR face crop (for saving snapshot).
            demographics:  Dict with age, gender, gender_confidence.
            frame_number:  Current frame counter.
            snapshot_dir:  Directory to save face crops.

        Returns:
            dict: {visitor_id, event_type, similarity, is_new}
        """
        if embedding is None:
            return {"visitor_id": None, "event_type": "unknown", "similarity": 0.0, "is_new": False}

        # Fast path: same track ID seen before this session
        if track_id in self._track_map:
            vid = self._track_map[track_id]
            self._update_last_seen(vid, frame_number, track_id, confidence)
            return {"visitor_id": vid, "event_type": "returning", "similarity": 1.0, "is_new": False}

        # Cosine similarity search
        best_id, best_sim = self._search_gallery(embedding)

        if best_sim >= self.threshold and best_id is not None:
            # Returning visitor
            self._track_map[track_id] = best_id
            self._update_returning_visitor(best_id, track_id, best_sim, frame_number, confidence)
            return {"visitor_id": best_id, "event_type": "returning", "similarity": best_sim, "is_new": False}
        else:
            # New visitor
            new_id = self._generate_visitor_id()
            crop_path = self._save_snapshot(crop, new_id, snapshot_dir)
            self._register_new_visitor(
                new_id, embedding, track_id, confidence, crop_path,
                demographics, frame_number
            )
            return {"visitor_id": new_id, "event_type": "new", "similarity": best_sim, "is_new": True}

    # ── Cosine Similarity ─────────────────────────────────────────────────────

    def _search_gallery(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Find the gallery entry with the highest cosine similarity."""
        if not self._gallery:
            return None, 0.0

        best_id = None
        best_sim = -1.0

        query = embedding / (np.linalg.norm(embedding) + 1e-6)

        for vid, data in self._gallery.items():
            ref = data["embedding"]
            sim = float(np.dot(query, ref))
            if sim > best_sim:
                best_sim = sim
                best_id = vid

        return best_id, best_sim

    # ── DB Operations ─────────────────────────────────────────────────────────

    def _register_new_visitor(
        self, visitor_id, embedding, track_id, confidence,
        crop_path, demographics, frame_number
    ):
        try:
            session = get_session()
            now = datetime.utcnow()
            visitor = Visitor(
                visitor_id=visitor_id,
                first_seen=now,
                last_seen=now,
                visit_count=1,
                crop_image_path=crop_path,
                age=demographics.get("age"),
                gender=demographics.get("gender"),
                gender_confidence=demographics.get("gender_confidence"),
                recognition_confidence=round(confidence, 4),
                embedding=json.dumps(embedding.tolist()),
            )
            session.add(visitor)

            event = EventLog(
                timestamp=now,
                visitor_id=visitor_id,
                event_type="new",
                track_id=track_id,
                confidence=confidence,
                frame_number=frame_number,
            )
            session.add(event)
            session.commit()

            # Add to in-memory gallery
            self._gallery[visitor_id] = {
                "embedding": embedding,
                "track_id": track_id,
            }
            self._track_map[track_id] = visitor_id
            logger.info(f"New visitor registered: {visitor_id}")
            logger.info(f"Face entry and registration: {visitor_id}")
        except Exception as e:
            logger.error(f"Failed to register visitor: {e}")
            try:
                session.rollback()
            except Exception:
                pass

    def _update_returning_visitor(self, visitor_id, track_id, similarity, frame_number, confidence):
        try:
            session = get_session()
            visitor = session.query(Visitor).filter_by(visitor_id=visitor_id).first()
            if visitor:
                visitor.last_seen = datetime.utcnow()
                visitor.visit_count = (visitor.visit_count or 0) + 1

            event = EventLog(
                timestamp=datetime.utcnow(),
                visitor_id=visitor_id,
                event_type="returning",
                track_id=track_id,
                confidence=similarity,
                frame_number=frame_number,
            )
            session.add(event)
            session.commit()
            logger.info(f"Face recognized and tracked: {visitor_id}")
        except Exception as e:
            logger.error(f"Failed to update returning visitor: {e}")
            try:
                session.rollback()
            except Exception:
                pass

    def log_exit(self, track_id: int, frame_number: int):
        try:
            visitor_id = self._track_map.get(track_id)
            if not visitor_id:
                return
            del self._track_map[track_id]  # Clean up track state

            session = get_session()
            event = EventLog(
                timestamp=datetime.utcnow(),
                visitor_id=visitor_id,
                event_type="exit",
                track_id=track_id,
                frame_number=frame_number,
            )
            session.add(event)
            session.commit()
            logger.info(f"Face exit: {visitor_id}")
        except Exception as e:
            logger.error(f"Failed to log exit: {e}")
            try:
                session.rollback()
            except Exception:
                pass

    def _update_last_seen(self, visitor_id, frame_number, track_id, confidence):
        """Lightweight update for already-tracked visitors in same session."""
        try:
            session = get_session()
            visitor = session.query(Visitor).filter_by(visitor_id=visitor_id).first()
            if visitor:
                visitor.last_seen = datetime.utcnow()
            session.commit()
        except Exception as e:
            logger.debug(f"last_seen update error: {e}")

    # ── Snapshot ───────────────────────────────────────────────────────────────

    def _save_snapshot(self, crop: np.ndarray, visitor_id: str, snapshot_dir: str) -> str | None:
        """Save face crop to disk and return relative path."""
        if crop is None or crop.size == 0:
            return None
        try:
            os.makedirs(snapshot_dir, exist_ok=True)
            filename = f"{visitor_id}.jpg"
            path = os.path.join(snapshot_dir, filename)
            cv2.imwrite(path, crop)
            return path
        except Exception as e:
            logger.debug(f"Snapshot save error: {e}")
            return None

    @staticmethod
    def _generate_visitor_id() -> str:
        return f"VIS-{uuid.uuid4().hex[:12].upper()}"

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)
