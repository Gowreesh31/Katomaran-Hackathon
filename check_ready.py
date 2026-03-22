"""
check_ready.py
Katomaran Face Tracking System — Pre-flight Health Check

Verifies all dependencies, model files, and system resources are ready.
Run this before starting main.py.

Usage:
    python check_ready.py
"""
import sys
import os
import json
import platform
import importlib
import urllib.request
import shutil
from pathlib import Path


# ── ANSI Colors ────────────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def ok(msg):    print(f"  {C.GREEN}✓{C.RESET}  {msg}")
def warn(msg):  print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")
def fail(msg):  print(f"  {C.RED}✗{C.RESET}  {msg}")
def info(msg):  print(f"  {C.BLUE}ℹ{C.RESET}  {msg}")
def section(title):
    print(f"\n{C.BOLD}{C.BLUE}{'─'*50}{C.RESET}")
    print(f"{C.BOLD}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'─'*50}{C.RESET}")


# ── Checks ──────────────────────────────────────────────────────────────────────

def check_python():
    section("Python Environment")
    v = sys.version_info
    if v.major == 3 and v.minor >= 9:
        ok(f"Python {v.major}.{v.minor}.{v.micro} (>= 3.9)")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro} — need 3.9+")
    info(f"Platform: {platform.system()} {platform.machine()}")


def check_dependencies():
    section("Core Dependencies")

    packages = {
        "cv2":         ("opencv-python", True),
        "numpy":       ("numpy", True),
        "ultralytics": ("ultralytics (YOLOv8)", True),
        "insightface": ("insightface (ArcFace)", True),
        "onnxruntime": ("onnxruntime", True),
        "sqlalchemy":  ("SQLAlchemy", True),
        "flask":       ("Flask", True),
        "flask_cors":  ("Flask-CORS", True),
        "scipy":       ("scipy", False),
        "sklearn":     ("scikit-learn", False),
        "PIL":         ("Pillow", False),
        "tqdm":        ("tqdm", False),
    }

    all_critical_ok = True
    for mod, (label, critical) in packages.items():
        try:
            importlib.import_module(mod)
            ok(label)
        except ImportError:
            if critical:
                fail(f"{label}  ← REQUIRED  (pip install {label.split()[0].lower()})")
                all_critical_ok = False
            else:
                warn(f"{label}  ← optional (pip install {label.split()[0].lower()})")

    return all_critical_ok


def check_config():
    section("Configuration")
    if not os.path.exists("config.json"):
        fail("config.json not found")
        return {}
    try:
        with open("config.json") as f:
            cfg = json.load(f)
        ok("config.json loaded")
        headless = cfg.get("pipeline", {}).get("headless", False)
        info(f"Headless mode: {'ON' if headless else 'OFF'}")
        info(f"DB path: {cfg.get('database', {}).get('path', 'unknown')}")
        info(f"Dashboard port: {cfg.get('dashboard', {}).get('port', 5000)}")
        return cfg
    except json.JSONDecodeError as e:
        fail(f"config.json parse error: {e}")
        return {}


def check_folders():
    section("Directory Structure")
    required = ["app", "web", "models", "logs", "logs/snapshots", "database"]
    all_ok = True
    for d in required:
        if os.path.isdir(d):
            ok(f"./{d}/")
        else:
            fail(f"./{d}/  ← MISSING  (run: mkdir -p {d})")
            all_ok = False
    return all_ok


def check_models():
    section("Model Files")

    # YOLOv8 face model — auto-downloaded by ultralytics
    yolo_paths = ["yolov8n-face.pt", "models/yolov8n-face.pt"]
    yolo_found = any(os.path.exists(p) for p in yolo_paths)
    if yolo_found:
        ok("yolov8n-face.pt found")
    else:
        warn("yolov8n-face.pt not found — ultralytics will auto-download on first run")

    # InsightFace buffalo_l — downloaded by insightface
    insightface_cache = os.path.expanduser("~/.insightface/models/buffalo_l")
    if os.path.isdir(insightface_cache):
        ok(f"InsightFace buffalo_l model found at {insightface_cache}")
    else:
        warn("InsightFace buffalo_l not cached — will auto-download on first run")
        info("To pre-download: python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)\"")

    # Age/Gender ONNX model
    ag_path = "models/age_gender.onnx"
    if os.path.exists(ag_path):
        size_mb = os.path.getsize(ag_path) / 1e6
        ok(f"age_gender.onnx found ({size_mb:.1f} MB)")
    else:
        warn(f"age_gender.onnx not found at '{ag_path}'")
        info("Demographics will use heuristic fallback.")
        info("To download: python check_ready.py --download-ag-model")


def check_camera():
    section("Camera / Capture")
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ok("Default camera (index 0) accessible")
            cap.release()
        else:
            warn("Default camera not accessible — use --source to specify a video file")
    except Exception as e:
        warn(f"Camera check failed: {e}")


def check_database(cfg: dict):
    section("Database")
    db_path = cfg.get("database", {}).get("path", "database/visitors.db")
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.isdir(db_dir):
        try:
            os.makedirs(str(db_dir), exist_ok=True)
            ok(f"Created database directory: {db_dir}/")
        except Exception as e:
            fail(f"Cannot create database directory: {e}")
            return

    if os.path.exists(db_path):
        size_kb = os.path.getsize(db_path) / 1024
        ok(f"Database file exists ({size_kb:.1f} KB): {db_path}")
    else:
        info(f"Database will be created at first run: {db_path}")

    # Test SQLAlchemy round-trip
    try:
        from sqlalchemy import create_engine, text  # type: ignore
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        ok("SQLAlchemy ↔ SQLite connection OK")
    except Exception as e:
        fail(f"Database connectivity error: {e}")


def download_ag_model():
    """Attempt to download a community age/gender ONNX model."""
    section("Downloading Age/Gender Model")
    url = "https://huggingface.co/Xenova/age-gender-estimation/resolve/main/onnx/model.onnx"
    dest = "models/age_gender.onnx"
    os.makedirs("models", exist_ok=True)
    info(f"Downloading from: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 8192
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Downloading... {pct:.1f}%", end="", flush=True)
        print()
        ok(f"Saved to {dest}")
    except Exception as e:
        fail(f"Download failed: {e}")
        info("Manually download an age/gender ONNX model and place it at models/age_gender.onnx")


def summary(passed: bool):
    section("Summary")
    if passed:
        print(f"\n  {C.GREEN}{C.BOLD}✓ System ready! Run:  python main.py{C.RESET}")
        print(f"  {C.BLUE}  Dashboard:         http://localhost:5000{C.RESET}\n")
    else:
        print(f"\n  {C.RED}{C.BOLD}✗ Some checks failed. Fix the issues above and re-run.{C.RESET}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Katomaran Face Tracker — Health Check")
    parser.add_argument("--download-ag-model", action="store_true", help="Download age/gender ONNX model")
    args = parser.parse_args()

    if args.download_ag_model:
        download_ag_model()
        sys.exit(0)

    print(f"\n{C.BOLD}{C.BLUE}  ◉ Katomaran Intelligent Face Tracking System{C.RESET}")
    print(f"  {C.BLUE}  Pre-flight Health Check{C.RESET}")

    check_python()
    deps_ok = check_dependencies()
    cfg = check_config()
    folders_ok = check_folders()
    check_models()
    check_camera()
    check_database(cfg)

    summary(deps_ok and folders_ok)
