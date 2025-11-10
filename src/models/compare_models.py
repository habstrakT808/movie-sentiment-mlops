"""
Comprehensive model comparison and analysis.
Generates comparison plots and detailed reports.
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.models.utils import (
    load_traditional_ml_data,
    load_transformer_data,
    prepare_features_and_labels,
)
from src.utils.helpers import ensure_directory, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_all_model_results() -> Dict:
    """Load results from all trained models."""
    logger.info("Loading all model results...")

    results = {}
    models = ["logistic_regression", "random_forest", "svm", "distilbert"]

    for model_name in models:
        metadata_path = Path(f"models/{model_name}/metadata.json")

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            if "metrics" in metadata:
                results[model_name] = metadata["metrics"]
                logger.info(f"✅ {model_name}: loaded")
        else:
            logger.warning(f"❌ {model_name}: metadata not found")

    return results


def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create formatted comparison table."""
    # Extract test metrics
    test_metrics = {}
    for model, metrics in results.items():
        test_metrics[model] = {
            k.replace("test_", ""): v
            for k, v in metrics.items()
            if k.startswith("test_")
        }

    df = pd.DataFrame(test_metrics).T
    df = df.round(4)

    # Sort by F1 score
    if "f1" in df.columns:
        df = df.sort_values("f1", ascending=False)

    return df


def plot_metrics_comparison(results: Dict, save_dir: Path):
    """Plot comparison of all metrics across models."""
    ensure_directory(save_dir)

    # Prepare data
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    data = {metric: [] for metric in metrics}

    for model in models:
        for metric in metrics:
            test_metric = f"test_{metric}"
            data[metric].append(results[model].get(test_metric, 0))

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(models, data[metric], color=colors)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Rotate x labels
        ax.set_xticklabels(models, rotation=45, ha="right")

    # Remove extra subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()
    save_path = save_dir / "metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Metrics comparison plot saved to {save_path}")


def plot_roc_curves_comparison(save_dir: Path):
    """Plot ROC curves for all models on same plot."""
    ensure_directory(save_dir)

    logger.info("Generating combined ROC curve comparison...")

    # Load test data
    _, _, test_df = load_traditional_ml_data()
    X_test, y_test = prepare_features_and_labels(test_df)

    # Load transformer test data
    _, _, test_df_transformer = load_transformer_data()
    test_texts = test_df_transformer["text_cleaned"].tolist()
    y_test_transformer = (
        test_df_transformer["sentiment"].map({"negative": 0, "positive": 1}).values
    )

    plt.figure(figsize=(10, 8))

    # Traditional ML models
    traditional_models = ["logistic_regression", "random_forest", "svm"]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    for idx, model_name in enumerate(traditional_models):
        model_path = Path(f"models/{model_name}/model.pkl")

        if model_path.exists():
            import pickle

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Get predictions
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                plt.plot(
                    fpr,
                    tpr,
                    lw=2,
                    color=colors[idx],
                    label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.4f})',
                )
            except Exception as e:
                logger.warning(f"Could not plot ROC for {model_name}: {e}")

    # DistilBERT
    distilbert_path = Path("models/distilbert")
    if distilbert_path.exists():
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizer,
            )

            # Load model
            tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path)
            model = DistilBertForSequenceClassification.from_pretrained(distilbert_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            # Get predictions
            from torch.utils.data import DataLoader

            from src.models.train_transformer import SentimentDataset

            dataset = SentimentDataset(
                test_texts, [0] * len(test_texts), tokenizer, 512
            )
            loader = DataLoader(dataset, batch_size=16, shuffle=False)

            all_probs = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                    all_probs.extend(probs[:, 1].cpu().numpy())

            y_pred_proba = np.array(all_probs)
            fpr, tpr, _ = roc_curve(y_test_transformer, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                lw=2,
                color=colors[3],
                label=f"DistilBERT (AUC = {roc_auc:.4f})",
            )
        except Exception as e:
            logger.warning(f"Could not plot ROC for DistilBERT: {e}")

    # Random classifier
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - All Models Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    save_path = save_dir / "roc_curves_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC curves comparison saved to {save_path}")


def generate_performance_report(results: Dict, save_path: Path):
    """Generate detailed performance report."""
    import yaml

    # Load performance gates
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    min_metrics = params["evaluation"]

    report = {
        "summary": {
            "total_models": len(results),
            "best_model": None,
            "best_f1": 0,
            "models_passed_gates": 0,
        },
        "models": {},
    }

    # Analyze each model
    for model_name, metrics in results.items():
        test_metrics = {k: v for k, v in metrics.items() if k.startswith("test_")}

        # Check gates
        gates_passed = True
        gate_results = {}

        for gate, threshold in min_metrics.items():
            if gate.startswith("min_"):
                metric_name = gate.replace("min_", "")
                test_metric = f"test_{metric_name}"

                if test_metric in metrics:
                    value = metrics[test_metric]
                    passed = value >= threshold
                    gate_results[metric_name] = {
                        "value": value,
                        "threshold": threshold,
                        "passed": passed,
                    }

                    if not passed:
                        gates_passed = False

        report["models"][model_name] = {
            "metrics": test_metrics,
            "gates": gate_results,
            "all_gates_passed": gates_passed,
        }

        # Update summary
        if gates_passed:
            report["summary"]["models_passed_gates"] += 1

        # Track best model
        if metrics.get("test_f1", 0) > report["summary"]["best_f1"]:
            report["summary"]["best_f1"] = metrics["test_f1"]
            report["summary"]["best_model"] = model_name

    # Save report
    save_json(report, save_path)
    logger.info(f"Performance report saved to {save_path}")

    return report


def print_final_summary(report: Dict):
    """Print final summary of all models."""
    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 100)

    print(f"\n📊 Total Models Trained: {report['summary']['total_models']}")
    print(f"✅ Models Passed All Gates: {report['summary']['models_passed_gates']}")
    print(f"🏆 Best Model: {report['summary']['best_model'].upper().replace('_', ' ')}")
    print(f"   Best F1 Score: {report['summary']['best_f1']:.4f}")

    print("\n" + "-" * 100)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("-" * 100)

    for model_name, data in report["models"].items():
        print(f"\n📦 {model_name.upper().replace('_', ' ')}")
        print("   Test Metrics:")
        for metric, value in data["metrics"].items():
            print(f"      {metric}: {value:.4f}")

        print("   Performance Gates:")
        all_passed = True
        for gate, info in data["gates"].items():
            status = "✅ PASS" if info["passed"] else "❌ FAIL"
            print(
                f"      {status} {gate}: {info['value']:.4f} (required: {info['threshold']:.4f})"
            )
            if not info["passed"]:
                all_passed = False

        if all_passed:
            print("   🎉 ALL GATES PASSED!")
        else:
            print("   ⚠️  SOME GATES FAILED")

    print("\n" + "=" * 100)


def main():
    """Main comparison pipeline."""
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON & ANALYSIS")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path("metrics/comparison")
    ensure_directory(output_dir)

    # Load all results
    results = load_all_model_results()

    if not results:
        logger.error("No model results found!")
        return

    # Create comparison table
    logger.info("Creating comparison table...")
    comparison_df = create_comparison_table(results)
    comparison_df.to_csv(output_dir / "model_comparison.csv")
    print("\n", comparison_df.to_string())

    # Plot metrics comparison
    logger.info("Generating metrics comparison plots...")
    plot_metrics_comparison(results, output_dir)

    # Plot ROC curves comparison
    logger.info("Generating ROC curves comparison...")
    try:
        plot_roc_curves_comparison(output_dir)
    except Exception as e:
        logger.warning(f"Could not generate ROC comparison: {e}")

    # Generate performance report
    logger.info("Generating performance report...")
    report = generate_performance_report(
        results, output_dir / "performance_report.json"
    )

    # Print final summary
    print_final_summary(report)

    logger.info(f"\n✅ Model comparison complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
