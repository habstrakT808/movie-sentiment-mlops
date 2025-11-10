#!/usr/bin/env python
"""Script to create detailed comparison of traditional ML models."""

import json
from pathlib import Path

import pandas as pd

# Load results
results_path = Path("metrics/traditional_ml_results.json")
if not results_path.exists():
    print("ERROR: metrics/traditional_ml_results.json not found!")
    print("Please run training first to generate results.")
    exit(1)

with open(results_path, "r") as f:
    results = json.load(f)

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T

# Select test metrics only
test_metrics = [col for col in comparison_df.columns if col.startswith("test_")]
comparison_df = comparison_df[test_metrics]

# Sort by F1 score
if "test_f1" in comparison_df.columns:
    comparison_df = comparison_df.sort_values("test_f1", ascending=False)

# Print formatted table
print("\n" + "=" * 100)
print("DETAILED MODEL COMPARISON - TEST SET PERFORMANCE")
print("=" * 100)

# Format the DataFrame for better display
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

print("\n", comparison_df.to_string())

# Performance gates from params.yaml
print("\n" + "=" * 100)
print("PERFORMANCE GATES CHECK")
print("=" * 100)

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
    for model_name in comparison_df.index:
        print(f"\n[{model_name.upper().replace('_', ' ')}]")
        all_passed = True
        for metric, threshold in min_metrics.items():
            if metric in comparison_df.columns:
                value = comparison_df.loc[model_name, metric]
                passed = value >= threshold
                status = "[PASS]" if passed else "[FAIL]"
                print(f"  {status} {metric}: {value:.4f} (required: {threshold:.4f})")
                if not passed:
                    all_passed = False

        if all_passed:
            print("  -> All gates PASSED")
        else:
            print("  -> Some gates FAILED")

except Exception as e:
    print(f"Could not load performance gates: {e}")

# Summary statistics
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

if "test_f1" in comparison_df.columns:
    best_f1 = comparison_df["test_f1"].max()
    best_model = comparison_df["test_f1"].idxmax()
    print(f"\nBest Model (by F1): {best_model}")
    print(f"  Test F1: {best_f1:.4f}")

if "test_accuracy" in comparison_df.columns:
    best_acc = comparison_df["test_accuracy"].max()
    best_model_acc = comparison_df["test_accuracy"].idxmax()
    print(f"\nBest Model (by Accuracy): {best_model_acc}")
    print(f"  Test Accuracy: {best_acc:.4f}")

if "test_roc_auc" in comparison_df.columns:
    best_auc = comparison_df["test_roc_auc"].max()
    best_model_auc = comparison_df["test_roc_auc"].idxmax()
    print(f"\nBest Model (by ROC AUC): {best_model_auc}")
    print(f"  Test ROC AUC: {best_auc:.4f}")

print("\n" + "=" * 100)
