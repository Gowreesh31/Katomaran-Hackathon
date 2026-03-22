"""
app/recognizer.py
Face Detection (YOLOv8) + Biometric Encoding (InsightFace ArcFace buffalo_l).

Pipeline:
  frame → YOLOv8 detect+track → crop faces → ArcFace encode → 512-d embedding
"""
import os
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Combines YOLOv8 face detection with ByteTrack tracking and
    InsightFace ArcFace 512-d embedding extraction.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        det_cfg = self.config.get("detection", {})
        rec_cfg = self.config.get("recognition", {})

        # Detection params
        self.model_name = det_cfg.get("model", "yolov8n-face.pt")
        self.imgsz = det_cfg.get("imgsz", 320)
        self.conf = det_cfg.get("conf_threshold", 0.45)
        self.iou = det_cfg.get("iou_threshold", 0.5)
        self.tracker = det_cfg.get("tracker", "bytetrack.yaml")

        # Recognition params
        self.model_pack = rec_cfg.get("model_pack", "buffalo_l")
        self.embedding_dim = rec_cfg.get("embedding_dim", 512)

        # Internal state
        self.detector = None
        self.arcface = None
        self._frame_count = 0

        self._load_detector()
        self._load_arcface()

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_detector(self):
        """Load YOLOv8 face detection model."""
        try:
            from ultralytics import YOLO
            self.detector = YOLO(self.model_name)
            logger.info(f"YOLOv8 detector loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 detector: {e}")
            raise

    def _load_arcface(self):
        """Load InsightFace ArcFace buffalo_l model."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            # Use only recognition — detection handled by YOLOv8
            self.arcface = FaceAnalysis(
                name=self.model_pack,
                allowed_modules=["recognition"],
                providers=["CPUExecutionProvider"],
            )
            self.arcface.prepare(ctx_id=0, det_size=(160, 160))
            logger.info(f"InsightFace ArcFace loaded: {self.model_pack}")
        except Exception as e:
            logger.warning(
                f"InsightFace not available ({e}). "
                "Embeddings will use HOG fallback (lower accuracy)."
            )
            self.arcface = None

    # ── Face Detection ────────────────────────────────────────────────────────

    def detect_and_track(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 + ByteTrack on a single frame.

        Returns:
            List of dicts: {track_id, bbox (x1,y1,x2,y2), conf, crop}
        """
        if self.detector is None:
            return []

        results = self.detector.track(
            frame,
            persist=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    # Clamp to frame bounds
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    detections.append(
                        {
                            "track_id": track_id,
                            "bbox": (x1, y1, x2, y2),
                            "conf": conf,
                            "crop": crop,
                        }
                    )
                except Exception as e:
                    logger.debug(f"Box parse error: {e}")

        self._frame_count += 1
        return detections

    # ── Embedding Extraction ──────────────────────────────────────────────────

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        """
        Extract 512-d ArcFace embedding from a face crop.

        Args:
            face_crop: BGR numpy array of the face crop.

        Returns:
            Normalised 512-d float32 numpy array, or None on failure.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # ── ArcFace path ──────────────────────────────────────────────────────
        if self.arcface is not None:
            try:
                # Resize to a decent size for recognition
                resized = cv2.resize(face_crop, (112, 112))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                faces = self.arcface.get(rgb)
                if faces:
                    emb = faces[0].normed_embedding.astype(np.float32)
                    logger.info("Embedding generation successful")
                    return emb / (np.linalg.norm(emb) + 1e-6)
            except Exception as e:
                logger.debug(f"ArcFace embedding error: {e}")

        # ── HOG fallback ──────────────────────────────────────────────────────
        return self._hog_embedding(face_crop)

    def _hog_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        """Lightweight HOG-based fallback embedding (512-d via padding/truncation)."""
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            hog = cv2.HOGDescriptor(
                _winSize=(64, 64),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9,
            )
            descriptor = hog.compute(gray).flatten()
            # Pad or truncate to 512
            if len(descriptor) < 512:
                descriptor = np.pad(descriptor, (0, 512 - len(descriptor)))
            else:
                descriptor = descriptor[:512]
            norm = np.linalg.norm(descriptor)
            return descriptor / (norm + 1e-6)
        except Exception as e:
            logger.debug(f"HOG fallback error: {e}")
            return None

    @property
    def frame_count(self) -> int:
        return self._frame_count
