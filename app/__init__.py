"""
app/__init__.py
Katomaran Intelligent Face Tracking System - Core Package
"""
from .database import init_db
from .recognizer import FaceRecognizer
from .visitor_manager import VisitorManager
from .demographics import DemographicsAnalyzer

__all__ = ["init_db", "FaceRecognizer", "VisitorManager", "DemographicsAnalyzer"]
__version__ = "1.0.0"
