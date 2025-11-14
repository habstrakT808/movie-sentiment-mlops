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

# Enhanced Custom CSS with Animations
st.markdown(
    """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main theme */
    .main {
        background: linear-gradient(135deg, #0E1117 0%, #1a1d29 100%);
    }

    /* Headers with gradient text */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        padding-bottom: 1rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        animation: fadeInDown 0.8s ease-out;
    }

    h2 {
        color: #FAFAFA;
        font-weight: 700;
        margin-top: 2rem;
        animation: fadeIn 1s ease-out;
    }

    h3 {
        color: #FAFAFA;
        font-weight: 600;
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
        animation: fadeInUp 1s ease-out;
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        animation: gradientShift 3s ease infinite;
        background-size: 200% auto;
    }

    .hero-subtitle {
        font-size: 1.3rem;
        color: #B0B0B0;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(38, 39, 48, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }

    .glass-card:hover::before {
        left: 100%;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Sidebar Enhancements */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0E1117 100%) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* Disable scroll only for navigation links section (Predict, Performance, Insights) */
    [data-testid="stSidebar"] nav,
    [data-testid="stSidebar"] > div > nav,
    [data-testid="stSidebar"] [class*="css-"] nav,
    [data-testid="stSidebar"] section[data-testid="stSidebarNav"] {
        overflow-y: hidden !important;
        overflow-x: hidden !important;
        max-height: fit-content !important;
        height: auto !important;
    }

    [data-testid="stSidebar"] nav > div,
    [data-testid="stSidebar"] nav > ul,
    [data-testid="stSidebar"] nav > section {
        overflow-y: hidden !important;
        overflow-x: hidden !important;
    }

    [data-testid="stSidebar"] nav::-webkit-scrollbar,
    [data-testid="stSidebar"] [class*="css-"] nav::-webkit-scrollbar,
    [data-testid="stSidebar"] section[data-testid="stSidebarNav"]::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }

    [data-testid="stSidebar"] nav,
    [data-testid="stSidebar"] [class*="css-"] nav,
    [data-testid="stSidebar"] section[data-testid="stSidebarNav"] {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }

    [data-testid="stSidebar"] .element-container {
        animation: fadeInLeft 0.5s ease-out;
    }

    /* Sidebar Navigation Links */
    [data-testid="stSidebar"] a {
        color: #B0B0B0 !important;
        text-decoration: none;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        display: block;
        transition: all 0.3s ease;
        margin: 0.25rem 0;
        font-weight: 500;
    }

    [data-testid="stSidebar"] a:hover {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea !important;
        transform: translateX(5px);
    }

    /* Active sidebar link */
    [data-testid="stSidebar"] a[aria-current="page"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        color: #667eea !important;
        border-left: 3px solid #667eea;
        font-weight: 700;
    }

    /* Sidebar Headers */
    [data-testid="stSidebar"] h3 {
        color: #FAFAFA;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* Sidebar Dividers */
    [data-testid="stSidebar"] hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Sidebar Stats Cards */
    [data-testid="stSidebar"] .stat-card {
        background: rgba(38, 39, 48, 0.5);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stat-card:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateX(3px);
    }

    /* Success/Error/Info Boxes */
    .stSuccess {
        background: rgba(0, 210, 106, 0.1);
        border-left: 4px solid #00D26A;
        border-radius: 8px;
        animation: slideInRight 0.5s ease-out;
    }

    .stError {
        background: rgba(255, 75, 75, 0.1);
        border-left: 4px solid #FF4B4B;
        border-radius: 8px;
        animation: shake 0.5s ease-out;
    }

    .stInfo {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        animation: slideInRight 0.5s ease-out;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1d29;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
<div class="hero-section">
    <div class="hero-title">📊 Model Performance Dashboard</div>
    <div class="hero-subtitle">
        Compare model performance metrics and analyze their strengths and weaknesses with detailed insights.
    </div>
</div>
""",
    unsafe_allow_html=True,
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
st.markdown(
    "<div style='margin-top: 2rem; margin-bottom: 2rem;'></div>", unsafe_allow_html=True
)

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
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            text-align: center;
            color: white;
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        " onmouseover="this.style.transform='translateY(-5px)';
            this.style.boxShadow='0 12px 48px rgba(102, 126, 234, 0.4)';"
            onmouseout="this.style.transform='translateY(0)';
            this.style.boxShadow='0 8px 32px rgba(102, 126, 234, 0.3)';">
            <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 700;">
                {model_name}</h3>
            <div style="
                display: inline-block;
                background-color: {badge_color};
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            ">
                {badge_text}
            </div>
            <div style="font-size: 3rem; font-weight: 800; margin: 1rem 0; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);">
                {accuracy:.1%}
            </div>
            <div style="font-size: 1rem; opacity: 0.9; font-weight: 600; margin-bottom: 1.5rem;">
                Accuracy
            </div>
            <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1.5rem 0;">
            <div style="font-size: 0.9rem; opacity: 0.9; line-height: 1.8;">
                <div style="margin-bottom: 0.5rem;"><strong>F1 Score:</strong> {f1:.1%}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Precision:</strong> {metrics['precision']:.1%}</div>
                <div><strong>Recall:</strong> {metrics['recall']:.1%}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<div style='margin-top: 2rem; margin-bottom: 2rem;'></div>", unsafe_allow_html=True
)

# Detailed metrics comparison
st.markdown(
    """
<div class="glass-card">
    <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">📈 Detailed Metrics Comparison</h2>
""",
    unsafe_allow_html=True,
)

# Bar chart
fig = create_model_comparison_bar(model_metrics)
st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='margin-top: 2rem; margin-bottom: 2rem;'></div>", unsafe_allow_html=True
)

# Metrics table
st.markdown(
    """
<div class="glass-card">
    <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">📋 Metrics Table</h2>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

metrics_df = pd.DataFrame(model_metrics).T
metrics_df = metrics_df.reset_index().rename(columns={"index": "Model"})

# Format percentages
for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}")

# Rename columns
metrics_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='margin-top: 3rem; margin-bottom: 3rem;'></div>", unsafe_allow_html=True
)

# Model details
st.markdown(
    """
<div class="glass-card">
    <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">🔍 Model Details</h2>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["DistilBERT", "Logistic Regression"])

with tab1:
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
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

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Load DistilBERT metadata
    distilbert_metadata_file = MODELS_DIR / "distilbert" / "metadata.json"
    if distilbert_metadata_file.exists():
        with open(distilbert_metadata_file, "r") as f:
            distilbert_data = json.load(f)

            st.markdown("#### Training Metrics")
            st.markdown(
                "<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True
            )

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
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
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

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

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
            st.markdown(
                "<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True
            )

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

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='margin-top: 2rem; margin-bottom: 2rem;'></div>", unsafe_allow_html=True
)

# Performance insights
st.markdown(
    """
<div class="glass-card">
    <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">💡 Performance Insights</h2>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='margin-top: 2rem; margin-bottom: 2rem;'></div>", unsafe_allow_html=True
)

# Confusion matrix comparison
st.markdown(
    """
<div class="glass-card">
    <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">🎯 Model Behavior Analysis</h2>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='margin-top: 3rem; margin-bottom: 2rem;'></div>", unsafe_allow_html=True
)

# Export report
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
