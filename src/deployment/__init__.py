"""
Deployment module for the movie sentiment analysis API.
"""

from .api import app
from .model_loader import ModelLoader, get_model_loader, initialize_model

__all__ = ["app", "ModelLoader", "get_model_loader", "initialize_model"]
