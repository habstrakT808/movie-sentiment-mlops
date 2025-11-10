#!/usr/bin/env python
"""Script to check and compare ALL models (Traditional ML + Transformer)."""

import json
from pathlib import Path

import pandas as pd

print("=" * 80)
print("ALL MODELS - RESULTS SUMMARY")
print("=" * 80)

# Traditional ML models
traditional_models = ["logistic_regression", "random_forest", "svm"]
# Transformer model
transformer_models = ["distilbert"]

all_results = {}

# Load traditional ML models
for model_name in traditional_models:
    metadata_path = Path(f"models/{model_name}/metadata.json")

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if "metrics" in metadata:
            metrics = metadata["metrics"]
            all_results[model_name] = {
                "test_accuracy": metrics.get("test_accuracy", 0),
                "test_precision": metrics.get("test_precision", 0),
                "test_recall": metrics.get("test_recall", 0),
                "test_f1": metrics.get("test_f1", 0),
                "test_roc_auc": metrics.get("test_roc_auc", 0),
            }

# Load transformer model
for model_name in transformer_models:
    metadata_path = Path(f"models/{model_name}/metadata.json")

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if "metrics" in metadata:
            metrics = metadata["metrics"]
            all_results[model_name] = {
                "test_accuracy": metrics.get("test_accuracy", 0),
                "test_precision": metrics.get("test_precision", 0),
                "test_recall": metrics.get("test_recall", 0),
                "test_f1": metrics.get("test_f1", 0),
                "test_roc_auc": metrics.get("test_roc_auc", 0),
            }

# Create comparison DataFrame
if all_results:
    df = pd.DataFrame(all_results).T
    df = df.round(4)

    # Sort by F1 score
    df = df.sort_values("test_f1", ascending=False)

    print("\n", df.to_string())

    # Find best model
    best_model = df["test_f1"].idxmax()
    best_f1 = df.loc[best_model, "test_f1"]

    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Model: {best_model}")
    print(f"Test F1: {best_f1:.4f}")
    print(f"Test Accuracy: {df.loc[best_model, 'test_accuracy']:.4f}")
    print(f"Test ROC AUC: {df.loc[best_model, 'test_roc_auc']:.4f}")

    # Performance gates check
    print("\n" + "=" * 80)
    print("PERFORMANCE GATES CHECK")
    print("=" * 80)

    try:
        import yaml

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        min_metrics = {
            "test_accuracy": params["evaluation"]["min_accuracy"],
            "test_f1": params["evaluation"]["min_f1"],
            "test_precision": params["evaluation"]["min_precision"],
        }

        print("\nMinimum Requirements:")
        for metric, threshold in min_metrics.items():
            print(f"  {metric}: {threshold:.4f}")

        print("\nModel Performance vs Gates:")
        for model_name in df.index:
            print(f"\n[{model_name.upper().replace('_', ' ')}]")
            all_passed = True
            for metric, threshold in min_metrics.items():
                if metric in df.columns:
                    value = df.loc[model_name, metric]
                    passed = value >= threshold
                    status = "[PASS]" if passed else "[FAIL]"
                    print(
                        f"  {status} {metric}: {value:.4f} (required: {threshold:.4f})"
                    )
                    if not passed:
                        all_passed = False

            if all_passed:
                print("  -> All gates PASSED")
            else:
                print("  -> Some gates FAILED")

    except Exception as e:
        print(f"Could not load performance gates: {e}")

print("\n" + "=" * 80)
