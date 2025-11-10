#!/usr/bin/env python
"""Script to re-log existing trained models to MLflow without retraining."""

import json
import pickle
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("movie_sentiment_analysis")


def log_traditional_ml_model(model_name: str, model_path: Path, metadata_path: Path):
    """Log traditional ML model to MLflow."""
    print(f"\n[LOG] Logging {model_name} to MLflow...")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name}_final"):
        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log parameters
        if "best_params" in metadata and metadata["best_params"]:
            mlflow.log_params(metadata["best_params"])

        # Log metrics
        if "metrics" in metadata:
            for metric_name, metric_value in metadata["metrics"].items():
                if metric_name.startswith("test_"):
                    mlflow.log_metric(metric_name, metric_value)

        # Log model path
        mlflow.log_param("model_path", str(model_path))

        # Log metadata
        mlflow.log_dict(metadata, "metadata.json")

        run_id = mlflow.active_run().info.run_id
        print(f"[SUCCESS] {model_name} logged to MLflow (run_id: {run_id})")
        return run_id


def log_transformer_model(model_dir: Path, metadata_path: Path):
    """Log transformer model to MLflow."""
    print("\n[LOG] Logging distilbert to MLflow...")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    # Start MLflow run
    with mlflow.start_run(run_name="distilbert_finetuning"):
        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Log tokenizer
        tokenizer.save_pretrained("tokenizer")
        mlflow.log_artifacts("tokenizer", "tokenizer")

        # Log config
        if "config" in metadata:
            mlflow.log_params(metadata["config"])

        # Log metrics
        if "metrics" in metadata:
            for metric_name, metric_value in metadata["metrics"].items():
                if metric_name.startswith("test_"):
                    mlflow.log_metric(metric_name, metric_value)

        # Log training history if available
        history_path = model_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)

            # Log training metrics over epochs
            if "train_loss" in history and "val_loss" in history:
                for epoch, (train_loss, val_loss) in enumerate(
                    zip(history["train_loss"], history["val_loss"])
                ):
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    if "train_accuracy" in history:
                        mlflow.log_metric(
                            "train_accuracy",
                            history["train_accuracy"][epoch],
                            step=epoch,
                        )
                    if "val_accuracy" in history:
                        mlflow.log_metric(
                            "val_accuracy", history["val_accuracy"][epoch], step=epoch
                        )

        # Log metadata
        mlflow.log_dict(metadata, "metadata.json")

        run_id = mlflow.active_run().info.run_id
        print(f"[SUCCESS] distilbert logged to MLflow (run_id: {run_id})")
        return run_id


def main():
    """Main function to re-log all models to MLflow."""
    print("=" * 80)
    print("RE-LOGGING MODELS TO MLFLOW")
    print("=" * 80)

    # Traditional ML models
    traditional_models = ["logistic_regression", "random_forest", "svm"]

    for model_name in traditional_models:
        model_path = Path(f"models/{model_name}/model.pkl")
        metadata_path = Path(f"models/{model_name}/metadata.json")

        if model_path.exists() and metadata_path.exists():
            try:
                log_traditional_ml_model(model_name, model_path, metadata_path)
            except Exception as e:
                print(f"[ERROR] Failed to log {model_name}: {e}")
        else:
            print(f"[SKIP] {model_name} not found")

    # Transformer model
    distilbert_dir = Path("models/distilbert")
    metadata_path = distilbert_dir / "metadata.json"

    if distilbert_dir.exists() and metadata_path.exists():
        try:
            log_transformer_model(distilbert_dir, metadata_path)
        except Exception as e:
            print(f"[ERROR] Failed to log distilbert: {e}")
    else:
        print("[SKIP] distilbert not found")

    print("\n" + "=" * 80)
    print("[SUCCESS] All models re-logged to MLflow!")
    print("=" * 80)

    # Verify
    print("\n[VERIFY] Checking MLflow runs...")
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name("movie_sentiment_analysis")
    if experiment:
        runs = client.search_runs(experiment.experiment_id, max_results=10)
        print(f"Total runs: {len(runs)}")
        for run in runs:
            f1 = run.data.metrics.get("test_f1", 0)
            print(f"  - {run.info.run_name}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
