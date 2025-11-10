"""
Utility functions for model training and evaluation.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_traditional_ml_data(
    data_dir: Path = Path("data/processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed data for traditional ML models.

    Args:
        data_dir: Directory containing processed data

    Returns:
        Tuple of (train_df, val_df, test_df) with features and labels
    """
    logger.info("Loading traditional ML data...")

    train_df = pd.read_csv(data_dir / "train_features.csv")
    val_df = pd.read_csv(data_dir / "validation_features.csv")
    test_df = pd.read_csv(data_dir / "test_features.csv")

    logger.info(
        f"Loaded data - Train: {len(train_df)}, "
        f"Val: {len(val_df)}, Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def load_transformer_data(
    data_dir: Path = Path("data/processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed data for transformer models.

    Args:
        data_dir: Directory containing processed data

    Returns:
        Tuple of (train_df, val_df, test_df) with text and labels
    """
    logger.info("Loading transformer data...")

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    logger.info(
        f"Loaded data - Train: {len(train_df)}, "
        f"Val: {len(val_df)}, Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def prepare_features_and_labels(
    df: pd.DataFrame, target_column: str = "sentiment"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and labels from DataFrame.

    Args:
        df: Input DataFrame
        target_column: Name of target column

    Returns:
        Tuple of (X, y)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be pd.DataFrame, got {type(df)}")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert sentiment labels to binary (if they're strings)
    if y.dtype == "object":
        y = y.map({"negative": 0, "positive": 1})

    return X, y


def load_label_encoder(encoder_path: Path = Path("data/processed/label_encoder.pkl")):
    """
    Load label encoder.

    Args:
        encoder_path: Path to label encoder

    Returns:
        Label encoder
    """
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    logger.info(f"Loaded label encoder from {encoder_path}")
    return encoder


def save_model_pickle(model: Any, filepath: Path):
    """
    Save model using pickle.

    Args:
        model: Model to save
        filepath: Path to save model
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {filepath}")


def load_model_pickle(filepath: Path) -> Any:
    """
    Load model from pickle.

    Args:
        filepath: Path to model file

    Returns:
        Loaded model
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Model loaded from {filepath}")
    return model


def plot_training_history(
    history: Dict[str, List[float]], save_path: Path, title: str = "Training History"
):
    """
    Plot training history.

    Args:
        history: Dictionary with metrics history
        save_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    if "train_loss" in history and "val_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # Plot accuracy
    if "train_accuracy" in history and "val_accuracy" in history:
        axes[1].plot(history["train_accuracy"], label="Train Accuracy")
        axes[1].plot(history["val_accuracy"], label="Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Training history plot saved to {save_path}")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: Path,
    top_n: int = 20,
    title: str = "Feature Importance",
):
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importances: Feature importance values
        save_path: Path to save plot
        top_n: Number of top features to show
        title: Plot title
    """
    # Get top N features
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Feature importance plot saved to {save_path}")


def compare_models_roc(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path,
    title: str = "ROC Curve Comparison",
):
    """
    Plot ROC curves for multiple models.

    Args:
        models_data: Dictionary mapping model names to (y_true, y_pred_proba)
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    for model_name, (y_true, y_pred_proba) in models_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC comparison plot saved to {save_path}")


def create_model_comparison_table(
    models_metrics: Dict[str, Dict[str, float]], save_path: Path
) -> pd.DataFrame:
    """
    Create model comparison table.

    Args:
        models_metrics: Dictionary mapping model names to metrics
        save_path: Path to save table

    Returns:
        Comparison DataFrame
    """
    # Create DataFrame
    comparison_df = pd.DataFrame(models_metrics).T

    # Sort by F1 score (descending)
    if "test_f1" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("test_f1", ascending=False)

    # Save to CSV
    comparison_df.to_csv(save_path)

    logger.info(f"Model comparison table saved to {save_path}")

    return comparison_df


def print_model_comparison(comparison_df: pd.DataFrame):
    """
    Print formatted model comparison table.

    Args:
        comparison_df: Comparison DataFrame
    """
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string())
    print("=" * 100 + "\n")


def get_misclassified_samples(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: pd.Series = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get misclassified samples for error analysis.

    Args:
        X: Features
        y_true: True labels
        y_pred: Predicted labels
        texts: Original texts (optional)
        top_n: Number of samples to return

    Returns:
        DataFrame with misclassified samples
    """
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]

    if len(misclassified_indices) == 0:
        logger.info("No misclassified samples found!")
        return pd.DataFrame()

    # Sample random misclassified examples
    sample_size = min(top_n, len(misclassified_indices))
    sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)

    # Create DataFrame
    error_df = pd.DataFrame(
        {
            "true_label": y_true[sample_indices],
            "predicted_label": y_pred[sample_indices],
        }
    )

    if texts is not None:
        error_df["text"] = texts.iloc[sample_indices].values

    logger.info(f"Found {len(misclassified_indices)} misclassified samples")

    return error_df
