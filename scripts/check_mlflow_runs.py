#!/usr/bin/env python
"""Script to check MLflow runs summary."""

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

try:
    experiment = client.get_experiment_by_name("movie_sentiment_analysis")
    if experiment is None:
        print("Experiment 'movie_sentiment_analysis' not found!")
        exit(1)

    runs = client.search_runs(
        experiment.experiment_id, order_by=["metrics.test_f1 DESC"], max_results=10
    )

    print("=" * 60)
    print("MLFLOW RUNS SUMMARY")
    print("=" * 60)
    print(f"Total runs: {len(runs)}\n")

    for i, run in enumerate(runs[:10], 1):
        f1 = run.data.metrics.get("test_f1", 0)
        accuracy = run.data.metrics.get("test_accuracy", 0)
        print(f"{i}. {run.info.run_name}")
        print(f"   F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        print(f"   Run ID: {run.info.run_id[:8]}...")
        print()

    print("=" * 60)

except Exception as e:
    print(f"Error: {e}")
