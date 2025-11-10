#!/usr/bin/env python
"""Script to verify MLflow runs and show details."""

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

print("=" * 80)
print("MLFLOW DATABASE VERIFICATION")
print("=" * 80)

# Check database file
db_path = Path("mlflow.db")
if db_path.exists():
    print(f"\n[OK] Database file exists: {db_path}")
    print(f"     Size: {db_path.stat().st_size / 1024:.2f} KB")
else:
    print(f"\n[ERROR] Database file not found: {db_path}")
    exit(1)

# Get experiment
try:
    experiment = client.get_experiment_by_name("movie_sentiment_analysis")
    if experiment is None:
        print("\n[ERROR] Experiment 'movie_sentiment_analysis' not found!")
        exit(1)

    print("\n[OK] Experiment found:")
    print(f"     Name: {experiment.name}")
    print(f"     ID: {experiment.experiment_id}")
    print(f"     Artifact Location: {experiment.artifact_location}")

    # Get runs
    runs = client.search_runs(
        experiment.experiment_id, order_by=["metrics.test_f1 DESC"], max_results=10
    )

    print(f"\n[OK] Total runs: {len(runs)}")

    if len(runs) == 0:
        print("\n[WARNING] No runs found in experiment!")
        print("          This is why MLflow UI is empty.")
        print("\n          Solution: Run 'python scripts/relog_models_to_mlflow.py'")
    else:
        print("\nRuns in database:")
        for i, run in enumerate(runs, 1):
            f1 = run.data.metrics.get("test_f1", 0)
            accuracy = run.data.metrics.get("test_accuracy", 0)
            print(f"\n  {i}. {run.info.run_name}")
            print(f"     Run ID: {run.info.run_id}")
            print(f"     Status: {run.info.status}")
            print(f"     F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
            print(f"     Created: {run.info.start_time}")

        print("\n" + "=" * 80)
        print("SOLUTION")
        print("=" * 80)
        print("\nIf MLflow UI is still empty, try:")
        print("1. Refresh browser (F5 or Ctrl+R)")
        print("2. Clear browser cache")
        print("3. Restart MLflow UI:")
        print("   - Stop current UI (Ctrl+C)")
        print("   - Run: mlflow ui --backend-store-uri sqlite:///mlflow.db")
        print("4. Check URL: http://localhost:5000/#/experiments/1/runs")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback

    traceback.print_exc()
