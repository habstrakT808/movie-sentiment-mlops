"""
Continuous Learning Pipeline for Movie Sentiment Analysis.

Implements automatic retraining pipeline that:
1. Collects feedback from predictions database
2. Combines feedback with original training data
3. Retrains model when threshold is reached
4. Evaluates and compares with production model
5. Deploys new model if performance improves
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dashboard.components.database import PredictionDatabase
from src.models.model_registry import ModelRegistry
from src.models.train_transformer import TransformerTrainer
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configuration from environment variables
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "1000"))
MIN_IMPROVEMENT = float(os.getenv("MIN_IMPROVEMENT", "0.01"))  # 1%
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "1"))


class ContinuousLearner:
    """
    Continuous learning pipeline for automatic model retraining.

    Features:
    - Collects feedback from predictions database
    - Combines feedback with original training data
    - Retrains model when sufficient feedback is available
    - Evaluates new model and compares with production
    - Deploys new model if performance improves
    """

    def __init__(
        self,
        retrain_threshold: int = RETRAIN_THRESHOLD,
        min_improvement: float = MIN_IMPROVEMENT,
        model_name: str = "distilbert",
        db_path: Optional[str] = None,
    ):
        """
        Initialize continuous learner.

        Args:
            retrain_threshold: Minimum feedback samples for retraining
            min_improvement: Minimum improvement (0.01 = 1%) for deployment
            model_name: Model to retrain (default: distilbert)
            db_path: Path to predictions database
        """
        self.retrain_threshold = retrain_threshold
        self.min_improvement = min_improvement
        self.model_name = model_name

        # Initialize database
        self.db = PredictionDatabase(db_path) if db_path else PredictionDatabase()

        # Initialize model registry
        self.registry = ModelRegistry()

        # Paths
        self.train_data_path = Config.PROCESSED_DATA_DIR / "train.csv"
        self.val_data_path = Config.PROCESSED_DATA_DIR / "validation.csv"
        self.test_data_path = Config.PROCESSED_DATA_DIR / "test.csv"

        logger.info(
            f"ContinuousLearner initialized - "
            f"Threshold: {retrain_threshold}, "
            f"Min improvement: {min_improvement:.2%}, "
            f"Model: {model_name}"
        )

    def collect_feedback_from_db(self) -> pd.DataFrame:
        """
        Collect feedback from database that hasn't been used for training.

        Returns:
            DataFrame with columns: id, text, sentiment, timestamp
        """
        try:
            logger.info("Collecting feedback from database...")

            feedback_df = self.db.get_feedback_for_training()

            if feedback_df.empty:
                logger.info("No new feedback available")
                return pd.DataFrame()

            # Map feedback to sentiment
            feedback_df["sentiment"] = feedback_df["feedback"].map(
                {1: "positive", -1: "negative"}
            )

            # Filter out any invalid mappings
            feedback_df = feedback_df.dropna(subset=["sentiment"])

            logger.info(f"Collected {len(feedback_df)} feedback samples")

            # Log sentiment distribution
            sentiment_dist = feedback_df["sentiment"].value_counts()
            logger.info(f"Feedback distribution: {sentiment_dist.to_dict()}")

            # Check for imbalance
            if len(sentiment_dist) > 0:
                max_ratio = sentiment_dist.max() / len(feedback_df)
                if max_ratio > 0.7:
                    logger.warning(
                        f"⚠️ Feedback data imbalance detected: "
                        f"max ratio: {max_ratio:.2%}"
                    )

            return feedback_df[["id", "text", "sentiment", "timestamp"]]

        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def has_sufficient_feedback(self, min_samples: Optional[int] = None) -> bool:
        """
        Check if there's sufficient feedback for retraining.

        Args:
            min_samples: Minimum samples required (default: self.retrain_threshold)

        Returns:
            True if sufficient feedback available
        """
        if min_samples is None:
            min_samples = self.retrain_threshold

        try:
            feedback_df = self.collect_feedback_from_db()
            feedback_count = len(feedback_df)

            sufficient = feedback_count >= min_samples

            if sufficient:
                logger.info(
                    f"✅ Sufficient feedback: {feedback_count} samples "
                    f"(threshold: {min_samples})"
                )
            else:
                logger.info(
                    f"⏳ Insufficient feedback: {feedback_count} samples "
                    f"(need: {min_samples})"
                )

            return sufficient

        except Exception as e:
            logger.error(f"Error checking feedback sufficiency: {e}")
            return False

    def prepare_training_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int]]:
        """
        Prepare training data by combining original data with feedback.

        Strategy:
        - Training set = original train + feedback data
        - Validation set = original validation (unchanged for fair comparison)
        - Test set = original test (unchanged)

        Returns:
            Tuple of (train_df, val_df, test_df, feedback_ids)
        """
        try:
            logger.info("Preparing training data...")

            # Load original training data
            if not self.train_data_path.exists():
                raise FileNotFoundError(
                    f"Training data not found: {self.train_data_path}"
                )

            original_train = pd.read_csv(self.train_data_path)
            logger.info(f"Loaded original training data: {len(original_train)} samples")

            # Load validation data (unchanged)
            if self.val_data_path.exists():
                val_df = pd.read_csv(self.val_data_path)
                logger.info(f"Loaded validation data: {len(val_df)} samples")
            else:
                logger.warning(
                    "Validation data not found, will use original train split"
                )
                val_df = None

            # Load test data (unchanged)
            if self.test_data_path.exists():
                test_df = pd.read_csv(self.test_data_path)
                logger.info(f"Loaded test data: {len(test_df)} samples")
            else:
                test_df = None

            # Collect feedback data
            feedback_df = self.collect_feedback_from_db()

            if feedback_df.empty:
                logger.warning("No feedback data available")
                return original_train, val_df, test_df, []

            # Store feedback IDs for marking as used
            feedback_ids = feedback_df["id"].tolist()

            # Prepare feedback data for training
            feedback_train = feedback_df[["text", "sentiment"]].copy()
            feedback_train = feedback_train.rename(columns={"text": "text_cleaned"})

            logger.info(f"Feedback data prepared: {len(feedback_train)} samples")

            # Combine original training data with feedback
            combined_train = pd.concat(
                [original_train, feedback_train], ignore_index=True
            )

            # Shuffle combined data
            combined_train = combined_train.sample(frac=1, random_state=42).reset_index(
                drop=True
            )

            logger.info(
                f"✅ Combined training data prepared: "
                f"{len(combined_train)} samples "
                f"(original: {len(original_train)}, feedback: {len(feedback_train)})"
            )

            # Log sentiment distribution
            sentiment_dist = combined_train["sentiment"].value_counts(normalize=True)
            logger.info(
                f"Combined data sentiment distribution: {sentiment_dist.to_dict()}"
            )

            return combined_train, val_df, test_df, feedback_ids

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

    def trigger_retraining(self) -> Dict:
        """
        Trigger model retraining with combined dataset.

        Returns:
            Dict with retraining results including metrics and deployment decision
        """
        try:
            logger.info("=" * 80)
            logger.info("🔄 STARTING CONTINUOUS LEARNING RETRAINING")
            logger.info("=" * 80)

            start_time = time.time()

            # Check sufficient feedback
            if not self.has_sufficient_feedback():
                return {
                    "status": "insufficient_feedback",
                    "message": f"Need at least {self.retrain_threshold} feedback samples",
                    "current_feedback": len(self.collect_feedback_from_db()),
                }

            # Prepare training data
            train_df, val_df, test_df, feedback_ids = self.prepare_training_data()

            if train_df is None or val_df is None:
                raise ValueError("Failed to prepare training data")

            # Prepare data for training
            train_texts = train_df["text_cleaned"].tolist()
            train_labels = (
                train_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()
            )

            val_texts = val_df["text_cleaned"].tolist()
            val_labels = (
                val_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()
            )

            if test_df is not None:
                test_texts = test_df["text_cleaned"].tolist()
                test_labels = (
                    test_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()
                )
            else:
                test_texts, test_labels = val_texts, val_labels

            # Load hyperparameters from params.yaml
            import yaml

            with open("params.yaml", "r") as f:
                params = yaml.safe_load(f)
            config = params["model"]["transformer"]

            logger.info("🚀 Training new model with combined dataset...")

            # Initialize trainer
            trainer = TransformerTrainer(model_name=self.model_name, config=config)

            # Train model
            trainer.train(train_texts, train_labels, val_texts, val_labels)

            # Evaluate on test set
            logger.info("📊 Evaluating new model on test set...")
            new_metrics = trainer.evaluate(
                test_texts, np.array(test_labels), split_name="test"
            )

            logger.info(f"New model test metrics: {new_metrics}")

            # Get production model metrics
            production_model = self.registry.get_best_model(
                metric="test_f1", model_type=self.model_name
            )

            if production_model is None:
                logger.warning(
                    "No production model found, deploying new model by default"
                )
                production_metrics = {"test_f1": 0.0, "test_accuracy": 0.0}
                should_deploy = True
            else:
                production_metrics = production_model["metrics"]
                logger.info(f"Production model metrics: {production_metrics}")

                # Compare models
                should_deploy = self.deploy_model_if_better(
                    new_metrics, production_metrics
                )

            # Save new model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_version_dir = Config.MODELS_DIR / f"{self.model_name}_{timestamp}"

            logger.info(f"💾 Saving new model to {model_version_dir}...")
            trainer.model_dir = model_version_dir
            trainer.save_model()

            # Log to MLflow
            run_id = trainer.log_to_mlflow(
                run_name=f"{self.model_name}_continuous_learning_{timestamp}"
            )

            # Register model
            self.registry.register_model(
                model_name=f"{self.model_name}_retrained",
                run_id=run_id,
                metrics=new_metrics,
                model_path=model_version_dir,
                metadata={
                    "training_time": trainer.training_time,
                    "feedback_samples": len(feedback_ids),
                    "total_training_samples": len(train_df),
                    "timestamp": timestamp,
                    "continuous_learning": True,
                },
            )

            # Mark feedback as used
            if should_deploy and feedback_ids:
                logger.info(
                    f"✅ Marking {len(feedback_ids)} feedback samples as used..."
                )
                self.db.mark_feedback_as_used(feedback_ids)

            # Calculate improvement
            improvement = self._calculate_improvement(new_metrics, production_metrics)

            total_time = time.time() - start_time

            result = {
                "status": "success",
                "new_model_metrics": new_metrics,
                "production_metrics": production_metrics,
                "improvement": improvement,
                "should_deploy": should_deploy,
                "model_version": timestamp,
                "model_path": str(model_version_dir),
                "feedback_samples": len(feedback_ids),
                "total_training_samples": len(train_df),
                "training_time": total_time,
                "run_id": run_id,
            }

            logger.info("=" * 80)
            if should_deploy:
                logger.info("✅ RETRAINING COMPLETE - NEW MODEL DEPLOYED")
            else:
                logger.info("⚠️ RETRAINING COMPLETE - NEW MODEL NOT DEPLOYED")
            logger.info("=" * 80)

            return result

        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

    def deploy_model_if_better(
        self, new_model_metrics: Dict, production_metrics: Dict
    ) -> bool:
        """
        Compare new model with production and decide deployment.

        Args:
            new_model_metrics: Metrics from new model
            production_metrics: Metrics from production model

        Returns:
            True if new model should be deployed
        """
        try:
            # Primary metric: test_f1
            new_f1 = new_model_metrics.get("test_f1", 0.0)
            production_f1 = production_metrics.get("test_f1", 0.0)

            # Calculate improvement
            if production_f1 > 0:
                improvement = (new_f1 - production_f1) / production_f1
            else:
                improvement = 1.0  # Deploy if no production model

            logger.info("📊 Model Comparison:")
            logger.info(f"  Production F1: {production_f1:.4f}")
            logger.info(f"  New Model F1:  {new_f1:.4f}")
            logger.info(f"  Improvement:   {improvement:.2%}")
            logger.info(f"  Threshold:     {self.min_improvement:.2%}")

            if improvement >= self.min_improvement:
                logger.info(
                    f"✅ Improvement {improvement:.2%} >= threshold {self.min_improvement:.2%}"
                )
                logger.info("🚀 Deploying new model...")
                return True
            elif improvement > 0:
                logger.info(
                    f"⚠️ Improvement {improvement:.2%} < threshold {self.min_improvement:.2%}"
                )
                logger.info("❌ Skipping deployment (improvement too small)")
                return False
            else:
                logger.warning(f"❌ Model performance degraded: {improvement:.2%}")
                logger.info("❌ Skipping deployment (performance worse)")
                return False

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return False

    def _calculate_improvement(
        self, new_metrics: Dict, production_metrics: Dict
    ) -> Dict:
        """Calculate improvement for all metrics."""
        improvement = {}

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            test_metric = f"test_{metric}"
            if test_metric in new_metrics and test_metric in production_metrics:
                new_val = new_metrics[test_metric]
                prod_val = production_metrics[test_metric]

                if prod_val > 0:
                    imp = (new_val - prod_val) / prod_val
                else:
                    imp = 1.0

                improvement[metric] = imp

        return improvement

    def trigger_data_collection_retraining(
        self,
        new_data_path: Optional[Path] = None,
    ) -> Dict:
        """
        Trigger retraining dengan data baru yang dikumpulkan secara berkala.

        Flow:
        1. Load training data yang sudah digabung (dari incremental preprocessor)
        2. Retrain model dengan data yang lebih lengkap
        3. Evaluate dan compare dengan production model
        4. Deploy jika lebih baik

        Args:
            new_data_path: Path ke data baru yang sudah di-preprocess (optional)
                          Jika None, akan load dari processed data directory

        Returns:
            Dict dengan hasil retraining (format sama dengan trigger_retraining)
        """
        try:
            logger.info("=" * 80)
            logger.info("🔄 STARTING DATA COLLECTION-BASED RETRAINING")
            logger.info("=" * 80)

            start_time = time.time()

            # Load training data (from processed directory, already merged by preprocessor)
            if new_data_path and Path(new_data_path).exists():
                logger.info(f"Loading training data from: {new_data_path}")
                train_df = pd.read_csv(new_data_path)
            else:
                # Load from default processed directory
                if not self.train_data_path.exists():
                    raise FileNotFoundError(
                        f"Training data not found: {self.train_data_path}. "
                        f"Please run incremental preprocessing first."
                    )
                logger.info(f"Loading training data from: {self.train_data_path}")
                train_df = pd.read_csv(self.train_data_path)

            # Load validation and test data
            if self.val_data_path.exists():
                val_df = pd.read_csv(self.val_data_path)
                logger.info(f"Loaded validation data: {len(val_df)} samples")
            else:
                raise FileNotFoundError(
                    f"Validation data not found: {self.val_data_path}"
                )

            if self.test_data_path.exists():
                test_df = pd.read_csv(self.test_data_path)
                logger.info(f"Loaded test data: {len(test_df)} samples")
            else:
                logger.warning("Test data not found, using validation data for testing")
                test_df = val_df

            logger.info(f"✅ Loaded training data: {len(train_df)} samples")
            logger.info(
                f"   Sentiment distribution: {train_df['sentiment'].value_counts().to_dict()}"
            )

            # Prepare data for training
            train_texts = train_df["text_cleaned"].tolist()
            train_labels = (
                train_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()
            )

            val_texts = val_df["text_cleaned"].tolist()
            val_labels = (
                val_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()
            )

            test_texts = test_df["text_cleaned"].tolist()
            test_labels = (
                test_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()
            )

            # Load hyperparameters from params.yaml
            import yaml

            with open("params.yaml", "r") as f:
                params = yaml.safe_load(f)
            config = params["model"]["transformer"]

            logger.info("🚀 Training new model with collected data...")

            # Initialize trainer
            trainer = TransformerTrainer(model_name=self.model_name, config=config)

            # Train model
            trainer.train(train_texts, train_labels, val_texts, val_labels)

            # Evaluate on test set
            logger.info("📊 Evaluating new model on test set...")
            new_metrics = trainer.evaluate(
                test_texts, np.array(test_labels), split_name="test"
            )

            logger.info(f"New model test metrics: {new_metrics}")

            # Get production model metrics
            production_model = self.registry.get_best_model(
                metric="test_f1", model_type=self.model_name
            )

            if production_model is None:
                logger.warning(
                    "No production model found, deploying new model by default"
                )
                production_metrics = {"test_f1": 0.0, "test_accuracy": 0.0}
                should_deploy = True
            else:
                production_metrics = production_model["metrics"]
                logger.info(f"Production model metrics: {production_metrics}")

                # Compare models
                should_deploy = self.deploy_model_if_better(
                    new_metrics, production_metrics
                )

            # Save new model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_version_dir = (
                Config.MODELS_DIR / f"{self.model_name}_data_collection_{timestamp}"
            )

            logger.info(f"💾 Saving new model to {model_version_dir}...")
            trainer.model_dir = model_version_dir
            trainer.save_model()

            # Log to MLflow
            run_id = trainer.log_to_mlflow(
                run_name=f"{self.model_name}_data_collection_{timestamp}"
            )

            # Register model
            self.registry.register_model(
                model_name=f"{self.model_name}_data_collection",
                run_id=run_id,
                metrics=new_metrics,
                model_path=model_version_dir,
                metadata={
                    "training_time": trainer.training_time,
                    "total_training_samples": len(train_df),
                    "timestamp": timestamp,
                    "data_collection_based": True,
                },
            )

            # Calculate improvement
            improvement = self._calculate_improvement(new_metrics, production_metrics)

            total_time = time.time() - start_time

            result = {
                "status": "success",
                "new_model_metrics": new_metrics,
                "production_metrics": production_metrics,
                "improvement": improvement,
                "should_deploy": should_deploy,
                "model_version": timestamp,
                "model_path": str(model_version_dir),
                "total_training_samples": len(train_df),
                "training_time": total_time,
                "run_id": run_id,
            }

            logger.info("=" * 80)
            if should_deploy:
                logger.info("✅ RETRAINING COMPLETE - NEW MODEL DEPLOYED")
            else:
                logger.info("⚠️ RETRAINING COMPLETE - NEW MODEL NOT DEPLOYED")
            logger.info("=" * 80)

            return result

        except Exception as e:
            logger.error(f"❌ Data collection retraining failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
