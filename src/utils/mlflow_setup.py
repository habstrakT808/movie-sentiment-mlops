"""
MLflow setup and configuration utilities.
Provides centralized MLflow tracking and experiment management.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowManager:
    """Manager class for MLflow operations."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize MLflow Manager.

        Args:
            tracking_uri: MLflow tracking URI (default from config)
            experiment_name: Experiment name (default from config)
        """
        self.tracking_uri = tracking_uri or Config.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or Config.MLFLOW_EXPERIMENT_NAME

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")

        # Create or get experiment
        self.experiment = self._setup_experiment()
        self.client = MlflowClient()

    def _setup_experiment(self) -> mlflow.entities.Experiment:
        """Create or get MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=str(Config.PROJECT_ROOT / "mlruns"),
                )
                experiment = mlflow.get_experiment(experiment_id)
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                logger.info(f"Using existing experiment: {self.experiment_name}")

            mlflow.set_experiment(self.experiment_name)
            return experiment

        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise

    def start_run(
        self, run_name: str, tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Optional tags for the run

        Returns:
            Active MLflow run
        """
        tags = tags or {}
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        return run

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def log_artifact(self, artifact_path: Path):
        """Log artifact to MLflow."""
        try:
            mlflow.log_artifact(str(artifact_path))
            logger.debug(f"Logged artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")

    def log_model(
        self, model: Any, artifact_path: str, **kwargs
    ) -> mlflow.models.model.ModelInfo:
        """
        Log model to MLflow.

        Args:
            model: Model object to log
            artifact_path: Path within run's artifact directory
            **kwargs: Additional arguments for model logging

        Returns:
            ModelInfo object
        """
        try:
            # Detect model type and log accordingly
            if hasattr(model, "sklearn"):
                model_info = mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            elif hasattr(model, "pytorch"):
                model_info = mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            else:
                model_info = mlflow.pyfunc.log_model(
                    artifact_path, python_model=model, **kwargs
                )

            logger.info(f"Logged model to: {artifact_path}")
            return model_info

        except Exception as e:
            logger.error(f"Error logging model: {e}")
            raise

    def register_model(
        self, run_id: str, model_name: str, model_path: str = "model"
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Register model in MLflow Model Registry.

        Args:
            run_id: Run ID containing the model
            model_name: Name for the registered model
            model_path: Path to model within run artifacts

        Returns:
            ModelVersion object
        """
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            model_version = mlflow.register_model(model_uri, model_name)
            logger.info(
                f"Registered model '{model_name}' version {model_version.version}"
            )
            return model_version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def transition_model_stage(
        self, model_name: str, version: int, stage: str
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Transition model to a different stage.

        Args:
            model_name: Name of the registered model
            version: Model version number
            stage: Target stage (Staging, Production, Archived)

        Returns:
            Updated ModelVersion object
        """
        try:
            model_version = self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return model_version

        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise

    def get_best_run(
        self, metric_name: str, ascending: bool = False
    ) -> mlflow.entities.Run:
        """
        Get the best run based on a metric.

        Args:
            metric_name: Name of the metric to optimize
            ascending: If True, lower is better

        Returns:
            Best run object
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
                max_results=1,
            )

            if len(runs) == 0:
                raise ValueError("No runs found in experiment")

            best_run_id = runs.iloc[0]["run_id"]
            best_run = self.client.get_run(best_run_id)

            logger.info(
                f"Best run: {best_run_id} with {metric_name}="
                f"{best_run.data.metrics.get(metric_name)}"
            )
            return best_run

        except Exception as e:
            logger.error(f"Error getting best run: {e}")
            raise

    def compare_runs(self, run_ids: list, metrics: list) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple runs across specified metrics.

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metric names to compare

        Returns:
            Dictionary of run comparisons
        """
        comparison = {}

        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
                "metrics": {
                    metric: run.data.metrics.get(metric, None) for metric in metrics
                },
            }

        return comparison

    def cleanup_old_runs(self, keep_last_n: int = 10):
        """
        Delete old runs, keeping only the most recent N runs.

        Args:
            keep_last_n: Number of recent runs to keep
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=["start_time DESC"],
            )

            if len(runs) <= keep_last_n:
                logger.info(f"Only {len(runs)} runs exist, nothing to cleanup")
                return

            runs_to_delete = runs.iloc[keep_last_n:]

            for _, run in runs_to_delete.iterrows():
                self.client.delete_run(run["run_id"])
                logger.info(f"Deleted run: {run['run_id']}")

            logger.info(f"Cleaned up {len(runs_to_delete)} old runs")

        except Exception as e:
            logger.error(f"Error cleaning up runs: {e}")


# Convenience function for quick setup
def setup_mlflow(
    tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None
) -> MLflowManager:
    """
    Quick setup function for MLflow.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name

    Returns:
        Configured MLflowManager instance
    """
    return MLflowManager(tracking_uri=tracking_uri, experiment_name=experiment_name)
