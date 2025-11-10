"""
Base trainer class for all models.
Provides common functionality for training, evaluation, and MLflow logging.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import Config
from src.utils.helpers import ensure_directory, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    All model trainers should inherit from this class and implement
    the abstract methods.
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        experiment_name: str = "movie_sentiment_analysis",
    ):
        """
        Initialize base trainer.

        Args:
            model_name: Name of the model (e.g., 'logistic_regression')
            config: Model configuration dictionary
            experiment_name: MLflow experiment name
        """
        self.model_name = model_name
        self.config = config
        self.experiment_name = experiment_name

        # Model and results
        self.model = None
        self.best_params = None
        self.metrics = {}

        # Paths
        self.model_dir = Config.MODELS_DIR / model_name
        self.metrics_dir = Path("metrics")

        # Create directories
        ensure_directory(self.model_dir)
        ensure_directory(self.metrics_dir)
        ensure_directory(self.metrics_dir / "confusion_matrices")
        ensure_directory(self.metrics_dir / "roc_curves")
        ensure_directory(self.metrics_dir / "classification_reports")

        # Setup MLflow
        self._setup_mlflow()

        logger.info(f"Initialized {model_name} trainer")

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow experiment set to: {self.experiment_name}")

    @abstractmethod
    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> Any:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Trained model
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        pass

    def evaluate(
        self, X: Any, y_true: np.ndarray, split_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Input features
            y_true: True labels
            split_name: Name of the split (train/val/test)

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating model on {split_name} set...")

        # Get predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        # Calculate metrics
        metrics = {
            f"{split_name}_accuracy": accuracy_score(y_true, y_pred),
            f"{split_name}_precision": precision_score(
                y_true, y_pred, average="binary"
            ),
            f"{split_name}_recall": recall_score(y_true, y_pred, average="binary"),
            f"{split_name}_f1": f1_score(y_true, y_pred, average="binary"),
        }

        # ROC AUC (need probability of positive class)
        if y_pred_proba.ndim > 1:
            y_pred_proba_pos = y_pred_proba[:, 1]
        else:
            y_pred_proba_pos = y_pred_proba

        metrics[f"{split_name}_roc_auc"] = roc_auc_score(y_true, y_pred_proba_pos)

        # Log metrics
        logger.info(f"{split_name.capitalize()} Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Store metrics
        self.metrics.update(metrics)

        return metrics

    def generate_confusion_matrix(
        self,
        X: Any,
        y_true: np.ndarray,
        split_name: str = "test",
        save_plot: bool = True,
    ) -> np.ndarray:
        """
        Generate confusion matrix.

        Args:
            X: Input features
            y_true: True labels
            split_name: Name of the split
            save_plot: Whether to save plot

        Returns:
            Confusion matrix
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        if save_plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )
            plt.title(f"Confusion Matrix - {self.model_name} ({split_name})")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # Save plot
            plot_path = (
                self.metrics_dir
                / "confusion_matrices"
                / f"{self.model_name}_{split_name}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Confusion matrix saved to {plot_path}")

        return cm

    def generate_roc_curve(
        self,
        X: Any,
        y_true: np.ndarray,
        split_name: str = "test",
        save_plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate ROC curve.

        Args:
            X: Input features
            y_true: True labels
            split_name: Name of the split
            save_plot: Whether to save plot

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        import matplotlib.pyplot as plt

        y_pred_proba = self.predict_proba(X)

        # Get probability of positive class
        if y_pred_proba.ndim > 1:
            y_pred_proba_pos = y_pred_proba[:, 1]
        else:
            y_pred_proba_pos = y_pred_proba

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba_pos)
        roc_auc = roc_auc_score(y_true, y_pred_proba_pos)

        if save_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.4f})",
            )
            plt.plot(
                [0, 1],
                [0, 1],
                color="navy",
                lw=2,
                linestyle="--",
                label="Random Classifier",
            )
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.model_name} ({split_name})")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)

            # Save plot
            plot_path = (
                self.metrics_dir / "roc_curves" / f"{self.model_name}_{split_name}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"ROC curve saved to {plot_path}")

        return fpr, tpr, thresholds

    def generate_classification_report(
        self,
        X: Any,
        y_true: np.ndarray,
        split_name: str = "test",
        save_report: bool = True,
    ) -> str:
        """
        Generate classification report.

        Args:
            X: Input features
            y_true: True labels
            split_name: Name of the split
            save_report: Whether to save report

        Returns:
            Classification report as string
        """
        y_pred = self.predict(X)
        report = classification_report(
            y_true, y_pred, target_names=["Negative", "Positive"], digits=4
        )

        if save_report:
            report_path = (
                self.metrics_dir
                / "classification_reports"
                / f"{self.model_name}_{split_name}.txt"
            )
            with open(report_path, "w") as f:
                f.write(f"Classification Report - {self.model_name} ({split_name})\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)

            logger.info(f"Classification report saved to {report_path}")

        return report

    def save_model(self):
        """Save trained model and metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Save model (implementation in subclasses)
        self._save_model_impl()

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "config": self.config,
            "best_params": self.best_params,
            "metrics": self.metrics,
        }

        metadata_path = self.model_dir / "metadata.json"
        save_json(metadata, metadata_path)

        logger.info(f"Model and metadata saved to {self.model_dir}")

    @abstractmethod
    def _save_model_impl(self):
        """Implementation-specific model saving."""
        pass

    def log_to_mlflow(self, run_name: Optional[str] = None):
        """
        Log model, parameters, and metrics to MLflow.

        Args:
            run_name: Name for the MLflow run
        """
        if run_name is None:
            run_name = f"{self.model_name}_run"

        try:
            with mlflow.start_run(run_name=run_name) as run:
                # Log parameters
                if self.config:
                    mlflow.log_params(self.config)

                if self.best_params:
                    mlflow.log_params(
                        {f"best_{k}": v for k, v in self.best_params.items()}
                    )

                # Log metrics
                if self.metrics:
                    mlflow.log_metrics(self.metrics)

                # Log model (implementation in subclasses)
                self._log_model_to_mlflow()

                # Log artifacts (normalize path for Windows compatibility)
                try:
                    mlflow.log_artifacts(
                        str((self.metrics_dir / "confusion_matrices").resolve())
                    )
                    mlflow.log_artifacts(
                        str((self.metrics_dir / "roc_curves").resolve())
                    )
                    mlflow.log_artifacts(
                        str((self.metrics_dir / "classification_reports").resolve())
                    )
                except Exception as e:
                    logger.warning(f"Failed to log some artifacts to MLflow: {e}")

                # Set tags
                mlflow.set_tag("model_type", self.model_name)
                mlflow.set_tag("framework", self._get_framework_name())

                logger.info(f"Logged to MLflow. Run ID: {run.info.run_id}")

                return run.info.run_id
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}. Model is still saved to disk.")
            return None

    @abstractmethod
    def _log_model_to_mlflow(self):
        """Implementation-specific MLflow model logging."""
        pass

    @abstractmethod
    def _get_framework_name(self) -> str:
        """Get framework name for tagging."""
        pass

    def check_performance_gates(self, min_metrics: Dict[str, float]) -> bool:
        """
        Check if model meets minimum performance requirements.

        Args:
            min_metrics: Dictionary of minimum required metrics

        Returns:
            True if all gates pass, False otherwise
        """
        logger.info("Checking performance gates...")

        all_passed = True
        for metric_name, min_value in min_metrics.items():
            # Check test metrics
            test_metric_name = f"test_{metric_name}"
            if test_metric_name in self.metrics:
                actual_value = self.metrics[test_metric_name]
                passed = actual_value >= min_value

                status = "[PASS]" if passed else "[FAIL]"
                logger.info(
                    f"{status} - {metric_name}: {actual_value:.4f} "
                    f"(required: {min_value:.4f})"
                )

                if not passed:
                    all_passed = False

        if all_passed:
            logger.info("[SUCCESS] All performance gates passed!")
        else:
            logger.warning("[FAIL] Some performance gates failed!")

        return all_passed
