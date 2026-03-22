"""
web/__init__.py
Flask application factory for the Analytics Dashboard.
"""
from flask import Flask
from flask_cors import CORS


def create_app(config: dict = None) -> Flask:
    """Create and configure the Flask dashboard application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "katomaran-face-tracker-2024"
    CORS(app)

    # Store config in app
    if config:
        app.config["TRACKER_CONFIG"] = config

    # Register blueprint
    from .routes import dashboard_bp
    app.register_blueprint(dashboard_bp)

    return app
