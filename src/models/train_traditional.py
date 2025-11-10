"""
Traditional ML models training with hyperparameter tuning.
Trains Logistic Regression, Random Forest, and SVM.
"""

import time
from pathlib import Path
from typing import Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.models.base_trainer import BaseTrainer
from src.models.model_registry import ModelRegistry
from src.models.utils import (
    load_traditional_ml_data,
    prepare_features_and_labels,
    save_model_pickle,
)
from src.utils.helpers import save_json, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TraditionalMLTrainer(BaseTrainer):
    """
    Trainer for traditional ML models.
    """

    def __init__(
        self,
        model_name: str,
        model_class,
        param_grid: Dict,
        config: Dict,
        cv_folds: int = 5,
    ):
        """
        Initialize traditional ML trainer.

        Args:
            model_name: Name of the model
            model_class: Sklearn model class
            param_grid: Hyperparameter grid for GridSearchCV
            config: Model configuration
            cv_folds: Number of cross-validation folds
        """
        super().__init__(model_name, config)

        self.model_class = model_class
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.grid_search = None
        self.training_time = 0

        logger.info(f"Initialized {model_name} trainer with {cv_folds}-fold CV")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ):
        """
        Train model with GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used for traditional ML)
            y_val: Validation labels (not used for traditional ML)

        Returns:
            Best trained model
        """
        logger.info(f"Starting training for {self.model_name}...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Hyperparameter grid size: {self._get_grid_size()}")

        start_time = time.time()

        # Initialize base model
        # For SVM, enable probability to allow predict_proba
        if self.model_name == "svm":
            base_model = self.model_class(
                random_state=self.config.get("random_state", 42), probability=True
            )
        else:
            base_model = self.model_class(
                random_state=self.config.get("random_state", 42)
            )

        # Setup GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            cv=self.cv_folds,
            scoring="f1",
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
        )

        # Train
        logger.info("Running GridSearchCV...")
        self.grid_search.fit(X_train, y_train)

        # Get best model
        self.model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_

        self.training_time = time.time() - start_time

        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV F1 score: {self.grid_search.best_score_:.4f}")

        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)

    def _save_model_impl(self):
        """Save model to disk."""
        model_path = self.model_dir / "model.pkl"
        save_model_pickle(self.model, model_path)

        # Save GridSearchCV results
        cv_results_path = self.model_dir / "cv_results.json"
        cv_results = {
            "best_params": self.best_params,
            "best_score": float(self.grid_search.best_score_),
            "cv_results": {
                "mean_test_score": self.grid_search.cv_results_[
                    "mean_test_score"
                ].tolist(),
                "std_test_score": self.grid_search.cv_results_[
                    "std_test_score"
                ].tolist(),
                "params": [str(p) for p in self.grid_search.cv_results_["params"]],
            },
        }
        save_json(cv_results, cv_results_path)

        # Save feature importance if available
        if hasattr(self.model, "feature_importances_"):
            self._save_feature_importance()
        elif hasattr(self.model, "coef_"):
            self._save_coefficients()

    def _save_feature_importance(self):
        """Save feature importance for tree-based models."""
        # This will be implemented when we have feature names
        logger.info("Model has feature_importances_ attribute")

    def _save_coefficients(self):
        """Save coefficients for linear models."""
        logger.info("Model has coef_ attribute")

    def _log_model_to_mlflow(self):
        """Log model to MLflow."""
        mlflow.sklearn.log_model(self.model, "model")

        # Log GridSearchCV results
        if self.grid_search:
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_metric("best_cv_f1", self.grid_search.best_score_)
            mlflow.log_metric("training_time_seconds", self.training_time)

    def _get_framework_name(self) -> str:
        """Get framework name."""
        return "scikit-learn"

    def _get_grid_size(self) -> int:
        """Calculate total number of parameter combinations."""
        size = 1
        for values in self.param_grid.values():
            size *= len(values)
        return size


@timer
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict,
) -> TraditionalMLTrainer:
    """
    Train Logistic Regression model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        config: Model configuration

    Returns:
        Trained LogisticRegression trainer
    """
    logger.info("=" * 80)
    logger.info("TRAINING LOGISTIC REGRESSION")
    logger.info("=" * 80)

    # Setup trainer
    trainer = TraditionalMLTrainer(
        model_name="logistic_regression",
        model_class=LogisticRegression,
        param_grid=config["logistic_regression"],
        config=config,
        cv_folds=config["cv_folds"],
    )

    # Train
    trainer.train(X_train, y_train)

    # Evaluate on all splits
    trainer.evaluate(X_train, y_train, split_name="train")
    trainer.evaluate(X_val, y_val, split_name="val")
    trainer.evaluate(X_test, y_test, split_name="test")

    # Generate visualizations
    trainer.generate_confusion_matrix(X_test, y_test, split_name="test")
    trainer.generate_roc_curve(X_test, y_test, split_name="test")
    trainer.generate_classification_report(X_test, y_test, split_name="test")

    # Save model
    trainer.save_model()

    # Log to MLflow
    trainer.log_to_mlflow(run_name="logistic_regression_gridsearch")

    logger.info("[SUCCESS] Logistic Regression training complete!")

    return trainer


@timer
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict,
) -> TraditionalMLTrainer:
    """
    Train Random Forest model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        config: Model configuration

    Returns:
        Trained RandomForest trainer
    """
    logger.info("=" * 80)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("=" * 80)

    # Setup trainer
    trainer = TraditionalMLTrainer(
        model_name="random_forest",
        model_class=RandomForestClassifier,
        param_grid=config["random_forest"],
        config=config,
        cv_folds=config["cv_folds"],
    )

    # Train
    trainer.train(X_train, y_train)

    # Evaluate on all splits
    trainer.evaluate(X_train, y_train, split_name="train")
    trainer.evaluate(X_val, y_val, split_name="val")
    trainer.evaluate(X_test, y_test, split_name="test")

    # Generate visualizations
    trainer.generate_confusion_matrix(X_test, y_test, split_name="test")
    trainer.generate_roc_curve(X_test, y_test, split_name="test")
    trainer.generate_classification_report(X_test, y_test, split_name="test")

    # Save model
    trainer.save_model()

    # Log to MLflow
    trainer.log_to_mlflow(run_name="random_forest_gridsearch")

    logger.info("[SUCCESS] Random Forest training complete!")

    return trainer


@timer
def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict,
) -> TraditionalMLTrainer:
    """
    Train SVM model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        config: Model configuration

    Returns:
        Trained SVM trainer
    """
    logger.info("=" * 80)
    logger.info("TRAINING SUPPORT VECTOR MACHINE")
    logger.info("=" * 80)

    # Setup trainer
    trainer = TraditionalMLTrainer(
        model_name="svm",
        model_class=SVC,
        param_grid=config["svm"],
        config=config,
        cv_folds=config["cv_folds"],
    )

    # Train
    trainer.train(X_train, y_train)

    # Evaluate on all splits
    trainer.evaluate(X_train, y_train, split_name="train")
    trainer.evaluate(X_val, y_val, split_name="val")
    trainer.evaluate(X_test, y_test, split_name="test")

    # Generate visualizations
    trainer.generate_confusion_matrix(X_test, y_test, split_name="test")
    trainer.generate_roc_curve(X_test, y_test, split_name="test")
    trainer.generate_classification_report(X_test, y_test, split_name="test")

    # Save model
    trainer.save_model()

    # Log to MLflow
    trainer.log_to_mlflow(run_name="svm_gridsearch")

    logger.info("[SUCCESS] SVM training complete!")

    return trainer


def main():
    """Main training pipeline for traditional ML models."""
    logger.info("=" * 80)
    logger.info("TRADITIONAL ML TRAINING PIPELINE")
    logger.info("=" * 80)

    # Load configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["model"]["traditional"]
    eval_config = params["evaluation"]

    # Load data
    logger.info("Loading preprocessed data...")
    train_df, val_df, test_df = load_traditional_ml_data()

    # Prepare features and labels
    X_train, y_train = prepare_features_and_labels(train_df)
    X_val, y_val = prepare_features_and_labels(val_df)
    X_test, y_test = prepare_features_and_labels(test_df)

    logger.info(
        f"Data loaded - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # Initialize model registry
    registry = ModelRegistry()

    # Store all trainers
    trainers = {}

    # Train Logistic Regression
    try:
        lr_trainer = train_logistic_regression(
            X_train, y_train, X_val, y_val, X_test, y_test, config
        )
        trainers["logistic_regression"] = lr_trainer

        # Register model
        registry.register_model(
            model_name="logistic_regression",
            run_id=lr_trainer.log_to_mlflow(run_name="lr_final"),
            metrics=lr_trainer.metrics,
            model_path=lr_trainer.model_dir / "model.pkl",
            metadata={"best_params": lr_trainer.best_params},
        )
    except Exception as e:
        logger.error(f"Logistic Regression training failed: {str(e)}")

    # Train Random Forest
    try:
        rf_trainer = train_random_forest(
            X_train, y_train, X_val, y_val, X_test, y_test, config
        )
        trainers["random_forest"] = rf_trainer

        # Register model
        registry.register_model(
            model_name="random_forest",
            run_id=rf_trainer.log_to_mlflow(run_name="rf_final"),
            metrics=rf_trainer.metrics,
            model_path=rf_trainer.model_dir / "model.pkl",
            metadata={"best_params": rf_trainer.best_params},
        )
    except Exception as e:
        logger.error(f"Random Forest training failed: {str(e)}")

    # Train SVM
    try:
        svm_trainer = train_svm(X_train, y_train, X_val, y_val, X_test, y_test, config)
        trainers["svm"] = svm_trainer

        # Register model
        registry.register_model(
            model_name="svm",
            run_id=svm_trainer.log_to_mlflow(run_name="svm_final"),
            metrics=svm_trainer.metrics,
            model_path=svm_trainer.model_dir / "model.pkl",
            metadata={"best_params": svm_trainer.best_params},
        )
    except Exception as e:
        logger.error(f"SVM training failed: {str(e)}")

    # Compare models
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    comparison = {}
    for name, trainer in trainers.items():
        comparison[name] = trainer.metrics

    # Save comparison
    comparison_path = Path("metrics/traditional_ml_results.json")
    save_json(comparison, comparison_path)

    # Print comparison table
    comparison_df = pd.DataFrame(comparison).T
    print("\n", comparison_df.to_string())

    # Check performance gates
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE GATES CHECK")
    logger.info("=" * 80)

    min_metrics = {
        "accuracy": eval_config["min_accuracy"],
        "f1": eval_config["min_f1"],
        "precision": eval_config["min_precision"],
    }

    for name, trainer in trainers.items():
        logger.info(f"\n{name.upper()}:")
        trainer.check_performance_gates(min_metrics)

    # Find best model
    best_model = registry.get_best_model(metric="test_f1")
    if best_model:
        logger.info(f"\n[WINNER] Best model: {best_model['model_name']}")
        logger.info(f"   Test F1: {best_model['metrics']['test_f1']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("[SUCCESS] TRADITIONAL ML TRAINING COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
