"""
Model registry for managing trained models.
Integrates with MLflow Model Registry.
"""

from pathlib import Path
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import Config
from src.utils.helpers import load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Model registry for managing trained models.
    """

    def __init__(self, tracking_uri: str = None):
        """
        Initialize model registry.

        Args:
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri is None:
            tracking_uri = Config.MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

        self.registry_file = Config.MODELS_DIR / "registry.json"
        self.registry = self._load_registry()

        logger.info("ModelRegistry initialized")

    def _load_registry(self) -> Dict:
        """Load registry from file."""
        if self.registry_file.exists():
            return load_json(self.registry_file)
        return {"models": {}}

    def _save_registry(self):
        """Save registry to file."""
        save_json(self.registry, self.registry_file)

    def register_model(
        self,
        model_name: str,
        run_id: str,
        metrics: Dict[str, float],
        model_path: Path,
        metadata: Optional[Dict] = None,
    ):
        """
        Register a trained model.

        Args:
            model_name: Name of the model
            run_id: MLflow run ID
            metrics: Model metrics
            model_path: Path to saved model
            metadata: Additional metadata
        """
        logger.info(f"Registering model: {model_name}")

        # Create model entry
        model_entry = {
            "model_name": model_name,
            "run_id": run_id,
            "metrics": metrics,
            "model_path": str(model_path),
            "metadata": metadata or {},
        }

        # Add to registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = []

        self.registry["models"][model_name].append(model_entry)

        # Save registry
        self._save_registry()

        logger.info(f"Model {model_name} registered successfully")

    def get_best_model(
        self, metric: str = "test_f1", model_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get best model based on a metric.

        Args:
            metric: Metric to use for comparison
            model_type: Filter by model type (optional)

        Returns:
            Best model entry
        """
        logger.info(f"Finding best model by {metric}...")

        best_model = None
        best_score = -float("inf")

        for model_name, versions in self.registry["models"].items():
            # Filter by model type if specified
            if model_type and model_name != model_type:
                continue

            for version in versions:
                if metric in version["metrics"]:
                    score = version["metrics"][metric]
                    if score > best_score:
                        best_score = score
                        best_model = version

        if best_model:
            logger.info(
                f"Best model: {best_model['model_name']} "
                f"({metric}={best_score:.4f})"
            )
        else:
            logger.warning("No models found in registry")

        return best_model

    def get_all_models(self) -> Dict[str, List[Dict]]:
        """
        Get all registered models.

        Returns:
            Dictionary of all models
        """
        return self.registry["models"]

    def compare_models(self, metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare all models on specified metrics.

        Args:
            metrics: List of metrics to compare

        Returns:
            Dictionary of model comparisons
        """
        if metrics is None:
            metrics = [
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
            ]

        logger.info("Comparing all models...")

        comparison = {}

        for model_name, versions in self.registry["models"].items():
            # Get latest version
            if versions:
                latest_version = versions[-1]
                comparison[model_name] = {
                    metric: latest_version["metrics"].get(metric, 0.0)
                    for metric in metrics
                }

        return comparison

    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """
        Transition model to a different stage in MLflow.

        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage (Staging/Production/Archived)
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )
            logger.info(f"Model {model_name} v{version} transitioned to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {str(e)}")

    def get_model_info(self, model_name: str) -> Optional[List[Dict]]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of model versions
        """
        return self.registry["models"].get(model_name)

    def print_registry_summary(self):
        """Print summary of registered models."""
        print("\n" + "=" * 100)
        print("MODEL REGISTRY SUMMARY")
        print("=" * 100)

        if not self.registry["models"]:
            print("No models registered yet.")
        else:
            for model_name, versions in self.registry["models"].items():
                print(f"\n📦 {model_name}")
                print(f"   Versions: {len(versions)}")
                if versions:
                    latest = versions[-1]
                    print("   Latest metrics:")
                    for metric, value in latest["metrics"].items():
                        if metric.startswith("test_"):
                            print(f"      {metric}: {value:.4f}")

        print("=" * 100 + "\n")
