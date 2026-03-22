"""
web/routes.py
Flask routes for the Katomaran Face Tracking Analytics Dashboard.
"""
import os
import json
import glob
import logging
from datetime import datetime

from flask import Blueprint, render_template, jsonify, send_file, abort, current_app, request  # type: ignore

from app.database import (  # type: ignore
    init_db, get_session, get_all_visitors, get_visitor_by_id,
    get_recent_events, get_hourly_traffic, get_gender_distribution,
    get_age_distribution, EventLog
)

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _ensure_db():
    """Lazily init DB if not already done."""
    config = current_app.config.get("TRACKER_CONFIG", {})
    try:
        get_session()
    except RuntimeError:
        init_db(config)


# ── Pages ──────────────────────────────────────────────────────────────────────

@dashboard_bp.route("/")
def index():
    """Main analytics dashboard."""
    _ensure_db()
    session = get_session()
    visitors = get_all_visitors(session)
    total = len(visitors)
    new_today = sum(
        1 for v in visitors
        if v.first_seen and v.first_seen.date() == datetime.utcnow().date()
    )
    returning = sum(1 for v in visitors if (v.visit_count or 1) > 1)
    avg_age_list = [v.age for v in visitors if v.age is not None]
    avg_age = float(f"{sum(avg_age_list) / len(avg_age_list):.1f}") if avg_age_list else None

    gender_dist = get_gender_distribution(session)
    age_dist = get_age_distribution(session)
    hourly = get_hourly_traffic(session)

    stats = {
        "total": total,
        "new_today": new_today,
        "returning": returning,
        "avg_age": avg_age,
    }

    return render_template(
        "index.html",
        stats=stats,
        visitors=visitors[:50],
        gender_dist=json.dumps(gender_dist),
        age_dist=json.dumps(age_dist),
        hourly=json.dumps(hourly),
    )


@dashboard_bp.route("/visitor/<visitor_id>")
def visitor_detail(visitor_id: str):
    """Individual visitor dossier page."""
    _ensure_db()
    session = get_session()
    visitor = get_visitor_by_id(session, visitor_id)
    if visitor is None:
        abort(404)
    events = (
        session.query(EventLog)
        .filter_by(visitor_id=visitor_id)
        .order_by(EventLog.timestamp.desc())
        .limit(50)
        .all()
    )
    return render_template("visitor.html", visitor=visitor, events=events)


# ── API ────────────────────────────────────────────────────────────────────────

@dashboard_bp.route("/api/stats")
def api_stats():
    """Live statistics JSON for auto-refresh."""
    _ensure_db()
    session = get_session()
    visitors = get_all_visitors(session)
    total = len(visitors)
    new_today = sum(
        1 for v in visitors
        if v.first_seen and v.first_seen.date() == datetime.utcnow().date()
    )
    return jsonify({
        "total_visitors": total,
        "new_today": new_today,
        "returning": sum(1 for v in visitors if (v.visit_count or 1) > 1),
        "hourly": get_hourly_traffic(session),
        "gender": get_gender_distribution(session),
        "age_buckets": get_age_distribution(session),
    })


@dashboard_bp.route("/api/visitors")
def api_visitors():
    """Paginated visitor list."""
    _ensure_db()
    session = get_session()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    all_visitors = get_all_visitors(session)
    start = (page - 1) * per_page
    subset = all_visitors[start: start + per_page]
    return jsonify({
        "total": len(all_visitors),
        "page": page,
        "per_page": per_page,
        "visitors": [v.to_dict() for v in subset],
    })


@dashboard_bp.route("/api/events")
def api_events():
    """Recent event log."""
    _ensure_db()
    session = get_session()
    limit = int(request.args.get("limit", 100))
    events = get_recent_events(session, limit=limit)
    return jsonify([e.to_dict() for e in events])


@dashboard_bp.route("/snapshots/<path:filename>")
def serve_snapshot(filename: str):
    """Serve face crop snapshots."""
    # Search the new Hackathon-required date folders first!
    matches = glob.glob(f"logs/entries/*/{filename}")
    if matches:
        return send_file(os.path.abspath(matches[-1]), mimetype="image/jpeg")

    # Fallback to default
    snapshot_dir = os.path.abspath("logs/snapshots")
    filepath = os.path.join(snapshot_dir, filename)
    if not os.path.isfile(filepath):
        abort(404)
    return send_file(filepath, mimetype="image/jpeg")


@dashboard_bp.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})
