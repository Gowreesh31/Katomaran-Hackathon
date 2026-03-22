"""
main.py
Katomaran Intelligent Face Tracking System
Main pipeline: Capture → Detect → Encode → Identify → Log

Usage:
    python main.py                    # Normal mode (with display window)
    python main.py --headless         # Override to headless mode
    python main.py --source 0         # Use webcam (default)
    python main.py --source video.mp4 # Process a video file
"""
import os
import sys
import json
import time
import argparse
import logging
import signal
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.database import init_db
from app.recognizer import FaceRecognizer
from app.visitor_manager import VisitorManager
from app.demographics import DemographicsAnalyzer


# ── Logging Setup ──────────────────────────────────────────────────────────────
def setup_logging(config: dict):
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    log_file = config.get("logging", {}).get("event_log", "logs/events.log")
    level_str = config.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    os.makedirs(log_dir, exist_ok=True)

    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ── Config Loading ─────────────────────────────────────────────────────────────
def load_config(path: str = "config.json") -> dict:
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        logging.getLogger(__name__).info(f"Config loaded from '{path}'.")
        return cfg
    except FileNotFoundError:
        logging.getLogger(__name__).warning(f"'{path}' not found — using defaults.")
        return {}


# ── Drawing Utilities ──────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, detection: dict, result: dict, demographics: dict) -> np.ndarray:
    """Draw bounding box + HUD overlay on frame."""
    x1, y1, x2, y2 = detection["bbox"]
    is_new = result.get("is_new", False)
    color = (0, 255, 120) if not is_new else (0, 120, 255)

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Info tag
    vid = result.get("visitor_id", "Unknown")
    short_id = vid[-8:] if vid else "Unknown"
    event = "NEW" if is_new else "RETURNING"
    sim = result.get("similarity", 0.0)
    label = f"[{event}] {short_id} | sim:{sim:.2f}"

    age = demographics.get("age")
    gender = demographics.get("gender")
    demo_label = ""
    if age and gender:
        demo_label = f"Age:{age:.0f} | {gender}"

    # Background rect for text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if demo_label:
        cv2.putText(frame, demo_label, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return frame


# ── Main Pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(config: dict, headless: bool, source):
    logger = logging.getLogger("pipeline")

    # Init subsystems
    logger.info("Initialising subsystems…")
    init_db(config)
    recognizer = FaceRecognizer(config)
    visitor_mgr = VisitorManager(config)
    demographics = DemographicsAnalyzer(config)

    now_date = datetime.utcnow().strftime("%Y-%m-%d")
    snapshot_dir = os.path.join("logs", "entries", now_date)
    os.makedirs(snapshot_dir, exist_ok=True)

    skip_frames = config.get("pipeline", {}).get("skip_frames", 2)

    # Open capture source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open capture source: {source}")
        sys.exit(1)

    cam_cfg = config.get("camera", {})
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
        cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))

    logger.info(
        f"Capture started | source={source} | headless={headless} | "
        f"gallery={visitor_mgr.gallery_size} visitors pre-loaded"
    )

    # Graceful shutdown
    running = True
    def _shutdown(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received…")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    frame_num = 0
    fps_counter = 0
    fps_time = time.time()
    display_fps = 0.0
    active_track_ids = set()

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    logger.info("Video file ended.")
                    break
                logger.warning("Frame capture failed — retrying…")
                time.sleep(0.05)
                continue

            frame_num += 1
            fps_counter += 1

            # FPS calculation every second
            now = time.time()
            if now - fps_time >= 1.0:
                display_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # Skip frames for performance
            if frame_num % max(1, skip_frames) != 0:
                if not headless:
                    cv2.imshow("Katomaran Face Tracker", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            # ── Detect & Track ─────────────────────────────────────────────
            detections = recognizer.detect_and_track(frame)

            for det in detections:
                crop = det["crop"]
                track_id = det["track_id"]
                conf = det["conf"]

                # ── Embed ──────────────────────────────────────────────────
                embedding = recognizer.get_embedding(crop)

                # ── Demographics ───────────────────────────────────────────
                demo = demographics.predict(crop)

                # ── Identify / Register ────────────────────────────────────
                result = visitor_mgr.identify(
                    embedding=embedding,
                    track_id=track_id,
                    confidence=conf,
                    crop=crop,
                    demographics=demo,
                    frame_number=frame_num,
                    snapshot_dir=snapshot_dir,
                )

                # ── Draw overlay (non-headless) ────────────────────────────
                if not headless:
                    frame = draw_overlay(frame, det, result, demo)

            # Exit Tracking
            current_track_ids = {det["track_id"] for det in detections if det["track_id"] != -1}
            exited_tracks = active_track_ids - current_track_ids
            for track_id in exited_tracks:
                visitor_mgr.log_exit(track_id, frame_num)
            active_track_ids = current_track_ids

            # HUD
            if not headless:
                cv2.putText(
                    frame,
                    f"FPS: {display_fps:.1f} | Visitors: {visitor_mgr.gallery_size} | Frame: {frame_num}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow("Katomaran Face Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        logger.info(
            f"Pipeline stopped. Processed {frame_num} frames. "
            f"Total unique visitors: {visitor_mgr.gallery_size}"
        )


# ── Dashboard Thread ───────────────────────────────────────────────────────────
def start_dashboard(config: dict):
    """Start Flask dashboard in a daemon thread."""
    try:
        from web import create_app
        dash_cfg = config.get("dashboard", {})
        app = create_app(config)
        t = threading.Thread(
            target=lambda: app.run(
                host=dash_cfg.get("host", "0.0.0.0"),
                port=dash_cfg.get("port", 5000),
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
        )
        t.start()
        logging.getLogger(__name__).info(
            f"Dashboard running at http://localhost:{dash_cfg.get('port', 5000)}"
        )
    except Exception as e:
        logging.getLogger(__name__).warning(f"Dashboard failed to start: {e}")


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Katomaran Intelligent Face Tracker")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--source", default=None, help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--headless", action="store_true", help="Force headless mode")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable Flask dashboard")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    # Determine headless mode
    headless = args.headless or config.get("pipeline", {}).get("headless", False)

    # Determine source
    if args.source is None:
        raw_source = config.get("camera", {}).get("source", 0)
    else:
        raw_source = args.source

    # Convert to int if it's a digit string (webcam index)
    try:
        source = int(raw_source)
    except (ValueError, TypeError):
        source = str(raw_source)

    # Start dashboard thread (unless disabled)
    if not args.no_dashboard:
        start_dashboard(config)

    run_pipeline(config, headless, source)

    # Keep dashboard alive after video stream ends
    if not args.no_dashboard:
        dash_port = config.get('dashboard', {}).get('port', 5000)
        logging.getLogger(__name__).info(
            f"Video complete. Dashboard remains active at http://localhost:{dash_port}\nPress Ctrl+C to exit."
        )
        # Restore default Ctrl+C behavior since run_pipeline overwrites it
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Dashboard stopped.")
            pass


if __name__ == "__main__":
    main()
