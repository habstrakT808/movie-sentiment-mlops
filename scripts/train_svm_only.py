"""
Script to train only SVM model.
Use this when Logistic Regression and Random Forest are already trained.
"""

import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_registry import ModelRegistry
from src.models.train_traditional import train_svm
from src.models.utils import load_traditional_ml_data, prepare_features_and_labels
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Train only SVM model."""
    logger.info("=" * 80)
    logger.info("SVM TRAINING ONLY")
    logger.info("=" * 80)

    # Load configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["model"]["traditional"]

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

    # Train SVM
    try:
        svm_trainer = train_svm(X_train, y_train, X_val, y_val, X_test, y_test, config)

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

    logger.info("\n" + "=" * 80)
    logger.info("[SUCCESS] SVM TRAINING COMPLETE!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
