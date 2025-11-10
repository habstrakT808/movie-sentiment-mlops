#!/usr/bin/env python
"""Script to check and compare traditional ML model results."""

import json
from pathlib import Path

print("=" * 80)
print("TRADITIONAL ML MODELS - RESULTS SUMMARY")
print("=" * 80)

models = ["logistic_regression", "random_forest", "svm"]

for model_name in models:
    metadata_path = Path(f"models/{model_name}/metadata.json")

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print(f"\n[{model_name.upper().replace('_', ' ')}]")
        print(f"   Best Params: {metadata.get('best_params', 'N/A')}")

        if "metrics" in metadata:
            metrics = metadata["metrics"]
            print(f"   Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
            print(f"   Test Precision: {metrics.get('test_precision', 0):.4f}")
            print(f"   Test Recall: {metrics.get('test_recall', 0):.4f}")
            print(f"   Test F1: {metrics.get('test_f1', 0):.4f}")
            print(f"   Test ROC AUC: {metrics.get('test_roc_auc', 0):.4f}")
        else:
            print("   Metrics: Not available")
    else:
        print(f"\n[{model_name.upper().replace('_', ' ')}] - metadata not found")

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

# Create comparison table
comparison = {}
for model_name in models:
    metadata_path = Path(f"models/{model_name}/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if "metrics" in metadata:
            metrics = metadata["metrics"]
            comparison[model_name] = {
                "test_accuracy": metrics.get("test_accuracy", 0),
                "test_precision": metrics.get("test_precision", 0),
                "test_recall": metrics.get("test_recall", 0),
                "test_f1": metrics.get("test_f1", 0),
                "test_roc_auc": metrics.get("test_roc_auc", 0),
            }

if comparison:
    import pandas as pd

    df = pd.DataFrame(comparison).T
    df = df.round(4)
    print("\n", df.to_string())

    # Find best model by F1
    best_model = max(comparison.items(), key=lambda x: x[1]["test_f1"])
    print(f"\n[WINNER] Best Model (by F1): {best_model[0]}")
    print(f"   Test F1: {best_model[1]['test_f1']:.4f}")

print("\n" + "=" * 80)
