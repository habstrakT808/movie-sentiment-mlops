"""
Models package for movie sentiment analysis.
"""

from src.models.base_trainer import BaseTrainer
from src.models.model_registry import ModelRegistry

__all__ = [
    "BaseTrainer",
    "ModelRegistry",
]
