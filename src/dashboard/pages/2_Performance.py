"""
Model performance dashboard page.
"""

import json

import pandas as pd
import streamlit as st

from src.dashboard.components.visualizations import create_model_comparison_bar
from src.dashboard.config import AVAILABLE_MODELS, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")

# Title
st.title("📊 Model Performance Dashboard")
st.markdown(
    """
Compare model performance metrics and analyze their strengths and weaknesses.
"""
)


# Load model metrics
@st.cache_data
def load_model_metrics():
    """Load metrics for all models."""
    metrics = {}

    for model_name, model_info in AVAILABLE_MODELS.items():
        model_path = model_info["path"]
        metadata_file = model_path / "metadata.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)

                    # Extract test metrics
                    model_metrics = data.get("metrics", {})

                    # Get test metrics (preferred) or val metrics
                    metrics[model_name] = {
                        "accuracy": model_metrics.get(
                            "test_accuracy", model_metrics.get("val_accuracy", 0)
                        ),
                        "precision": model_metrics.get(
                            "test_precision", model_metrics.get("val_precision", 0)
                        ),
                        "recall": model_metrics.get(
                            "test_recall", model_metrics.get("val_recall", 0)
                        ),
                        "f1": model_metrics.get(
                            "test_f1", model_metrics.get("val_f1", 0)
                        ),
                        "roc_auc": model_metrics.get(
                            "test_roc_auc", model_metrics.get("val_roc_auc", 0)
                        ),
                    }

                    logger.info(f"Loaded metrics for {model_name}")

            except Exception as e:
                logger.error(f"Failed to load metrics for {model_name}: {str(e)}")
                metrics[model_name] = {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "roc_auc": 0,
                }
        else:
            logger.warning(f"Metadata file not found for {model_name}")
            metrics[model_name] = {
                "accuracy": model_info["accuracy"],  # Use config value as fallback
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "roc_auc": 0,
            }

    return metrics


# Load metrics
try:
    model_metrics = load_model_metrics()
except Exception as e:
    st.error(f"❌ Failed to load model metrics: {str(e)}")
    st.stop()

# Model comparison overview
st.markdown("---")
st.subheader("🏆 Model Comparison")

# Create metrics cards
cols = st.columns(len(model_metrics))

for i, (model_name, metrics) in enumerate(model_metrics.items()):
    with cols[i]:
        accuracy = metrics["accuracy"]
        f1 = metrics["f1"]

        # Determine badge color
        if accuracy >= 0.90:
            badge_color = "#00D26A"  # Green
            badge_text = "Excellent"
        elif accuracy >= 0.85:
            badge_color = "#FFA500"  # Orange
            badge_text = "Good"
        else:
            badge_color = "#FF4B4B"  # Red
            badge_text = "Fair"

        st.markdown(
            f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            color: white;
        ">
            <h3 style="margin: 0 0 10px 0;">{model_name}</h3>
            <div style="
                display: inline-block;
                background-color: {badge_color};
                padding: 5px 15px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                margin-bottom: 15px;
            ">
                {badge_text}
            </div>
            <div style="font-size: 36px; font-weight: bold; margin: 10px 0;">
                {accuracy:.1%}
            </div>
            <div style="font-size: 14px; opacity: 0.9;">
                Accuracy
            </div>
            <hr style="border: 1px solid rgba(255,255,255,0.2); margin: 15px 0;">
            <div style="font-size: 12px; opacity: 0.8;">
                F1 Score: {f1:.1%}<br>
                Precision: {metrics['precision']:.1%}<br>
                Recall: {metrics['recall']:.1%}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Detailed metrics comparison
st.markdown("---")
st.subheader("📈 Detailed Metrics Comparison")

# Bar chart
fig = create_model_comparison_bar(model_metrics)
st.plotly_chart(fig, use_container_width=True)

# Metrics table
st.markdown("---")
st.subheader("📋 Metrics Table")

metrics_df = pd.DataFrame(model_metrics).T
metrics_df = metrics_df.reset_index().rename(columns={"index": "Model"})

# Format percentages
for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}")

# Rename columns
metrics_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# Model details
st.markdown("---")
st.subheader("🔍 Model Details")

tab1, tab2 = st.tabs(["DistilBERT", "Logistic Regression"])

with tab1:
    st.markdown(
        """
    ### DistilBERT (Transformer Model)

    **Architecture:**
    - Distilled version of BERT
    - 6 transformer layers
    - 66M parameters
    - 512 max sequence length

    **Training Configuration:**
    - Epochs: 3
    - Batch size: 16
    - Learning rate: 2e-5
    - Optimizer: AdamW
    - Warmup steps: 500

    **Strengths:**
    - ✅ Excellent accuracy (92.50%)
    - ✅ Understands context and nuance
    - ✅ Handles complex language patterns
    - ✅ Good with sarcasm and irony

    **Weaknesses:**
    - ⚠️ Slower inference time (~400ms)
    - ⚠️ Requires more memory
    - ⚠️ Computationally expensive

    **Best For:**
    - Complex, nuanced reviews
    - Long-form content
    - High-accuracy requirements
    """
    )

    # Load DistilBERT metadata
    distilbert_metadata_file = MODELS_DIR / "distilbert" / "metadata.json"
    if distilbert_metadata_file.exists():
        with open(distilbert_metadata_file, "r") as f:
            distilbert_data = json.load(f)

            st.markdown("#### Training Metrics")

            train_metrics = distilbert_data.get("metrics", {})

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Train Accuracy", f"{train_metrics.get('train_accuracy', 0):.2%}"
                )
                st.metric(
                    "Validation Accuracy", f"{train_metrics.get('val_accuracy', 0):.2%}"
                )

            with col2:
                st.metric("Train F1", f"{train_metrics.get('train_f1', 0):.2%}")
                st.metric("Validation F1", f"{train_metrics.get('val_f1', 0):.2%}")

            with col3:
                st.metric(
                    "Train ROC AUC", f"{train_metrics.get('train_roc_auc', 0):.2%}"
                )
                st.metric(
                    "Validation ROC AUC", f"{train_metrics.get('val_roc_auc', 0):.2%}"
                )

with tab2:
    st.markdown(
        """
    ### Logistic Regression (Traditional ML)

    **Algorithm:**
    - Linear classification model
    - TF-IDF feature extraction
    - 5,016 features
    - L2 regularization

    **Training Configuration:**
    - Cross-validation: 5-fold
    - Regularization (C): Optimized via GridSearch
    - Solver: liblinear/saga
    - Max iterations: 2000

    **Strengths:**
    - ✅ Fast inference (~50ms)
    - ✅ Low memory footprint
    - ✅ Interpretable results
    - ✅ Good baseline performance (87.40%)

    **Weaknesses:**
    - ⚠️ Lower accuracy than DistilBERT
    - ⚠️ Limited context understanding
    - ⚠️ Struggles with complex patterns
    - ⚠️ Less effective with sarcasm

    **Best For:**
    - Quick sentiment checks
    - Resource-constrained environments
    - Batch processing
    - Interpretability requirements
    """
    )

    # Load Logistic Regression metadata
    lr_metadata_file = MODELS_DIR / "logistic_regression" / "metadata.json"
    if lr_metadata_file.exists():
        with open(lr_metadata_file, "r") as f:
            lr_data = json.load(f)

            st.markdown("#### Best Hyperparameters")

            best_params = lr_data.get("best_params", {})

            if best_params:
                st.json(best_params)

            st.markdown("#### Training Metrics")

            lr_metrics = lr_data.get("metrics", {})

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Train Accuracy", f"{lr_metrics.get('train_accuracy', 0):.2%}"
                )
                st.metric(
                    "Validation Accuracy", f"{lr_metrics.get('val_accuracy', 0):.2%}"
                )

            with col2:
                st.metric("Train F1", f"{lr_metrics.get('train_f1', 0):.2%}")
                st.metric("Validation F1", f"{lr_metrics.get('val_f1', 0):.2%}")

            with col3:
                st.metric(
                    "Train Precision", f"{lr_metrics.get('train_precision', 0):.2%}"
                )
                st.metric("Train Recall", f"{lr_metrics.get('train_recall', 0):.2%}")

# Performance insights
st.markdown("---")
st.subheader("💡 Performance Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    ### Key Findings

    1. **DistilBERT outperforms** traditional ML by ~5 percentage points
    2. **Both models exceed** the minimum accuracy threshold (85%)
    3. **High ROC AUC scores** indicate excellent class separation
    4. **Balanced precision and recall** across both models
    5. **DistilBERT shows minimal overfitting** (train vs. test gap < 7%)
    """
    )

with col2:
    st.markdown(
        """
    ### Recommendations

    ✅ **Use DistilBERT when:**
    - Accuracy is critical
    - Processing complex reviews
    - Resources are available
    - Latency is acceptable

    ✅ **Use Logistic Regression when:**
    - Speed is priority
    - Processing large batches
    - Resources are limited
    - Good-enough accuracy is sufficient
    """
    )

# Confusion matrix comparison
st.markdown("---")
st.subheader("🎯 Model Behavior Analysis")

st.info(
    """
**Performance Characteristics:**

- **DistilBERT**: Excellent at detecting nuanced sentiment, especially in complex sentences with mixed emotions
- **Logistic Regression**: Strong baseline performance, particularly effective with clear sentiment indicators
- **Both models**: Show consistent performance across positive and negative classes with minimal bias

**Error Analysis:**
- Most errors occur on neutral or ambiguous reviews
- Sarcastic content remains challenging for both models
- Context-dependent sentiment (e.g., "so bad it's good") requires careful interpretation
"""
)

# Export report
st.markdown("---")

if st.button("📥 Download Performance Report"):
    report_data = {
        "models": model_metrics,
        "summary": {
            "best_model": max(model_metrics.items(), key=lambda x: x[1]["accuracy"])[0],
            "avg_accuracy": sum(m["accuracy"] for m in model_metrics.values())
            / len(model_metrics),
        },
    }

    report_json = json.dumps(report_data, indent=2)

    st.download_button(
        label="Download JSON Report",
        data=report_json,
        file_name="model_performance_report.json",
        mime="application/json",
    )
