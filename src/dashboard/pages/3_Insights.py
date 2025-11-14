"""
Data insights and visualizations page.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from src.dashboard.components.database import get_database
from src.dashboard.components.visualizations import (
    create_confidence_histogram,
    create_predictions_timeline,
    create_sentiment_pie,
    create_text_length_distribution,
)
from src.dashboard.config import DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Data Insights", page_icon="📈", layout="wide")

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

    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
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
    <div class="hero-title">📈 Data Insights & Analytics</div>
    <div class="hero-subtitle">
        Explore dataset statistics, sentiment distributions, and prediction patterns with interactive visualizations.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# Load data
@st.cache_data
def load_training_data():
    """Load training data for analysis."""
    try:
        train_file = DATA_DIR / "processed" / "train.csv"
        if train_file.exists():
            df = pd.read_csv(train_file)
            logger.info(f"Loaded training data: {len(df)} samples")
            return df
        else:
            logger.warning("Training data not found")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load training data: {str(e)}")
        return pd.DataFrame()


# Load prediction history
db = get_database()
prediction_stats = db.get_statistics()
predictions_df = db.get_recent_predictions(limit=1000)

# Overview metrics
st.markdown("---")
st.subheader("📊 Overview Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    ">
        <div style="font-size: 14px; opacity: 0.8;">Total Predictions</div>
        <div style="font-size: 36px; font-weight: bold;">{prediction_stats['total']}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    positive_count = prediction_stats["by_sentiment"].get("positive", 0)
    positive_pct = (
        (positive_count / prediction_stats["total"] * 100)
        if prediction_stats["total"] > 0
        else 0
    )
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #00D26A 0%, #00A854 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    ">
        <div style="font-size: 14px; opacity: 0.8;">Positive Reviews</div>
        <div style="font-size: 36px; font-weight: bold;">{positive_count}</div>
        <div style="font-size: 12px; opacity: 0.8;">{positive_pct:.1f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    negative_count = prediction_stats["by_sentiment"].get("negative", 0)
    negative_pct = (
        (negative_count / prediction_stats["total"] * 100)
        if prediction_stats["total"] > 0
        else 0
    )
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #FF4B4B 0%, #CC3939 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    ">
        <div style="font-size: 14px; opacity: 0.8;">Negative Reviews</div>
        <div style="font-size: 36px; font-weight: bold;">{negative_count}</div>
        <div style="font-size: 12px; opacity: 0.8;">{negative_pct:.1f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #FFA500 0%, #CC8400 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    ">
        <div style="font-size: 14px; opacity: 0.8;">Avg Confidence</div>
        <div style="font-size: 36px; font-weight: bold;">{prediction_stats['avg_confidence']:.1%}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Sentiment distribution
if prediction_stats["total"] > 0:
    st.markdown("---")
    st.subheader("🎭 Sentiment Distribution")

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_pie = create_sentiment_pie(prediction_stats["by_sentiment"])
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown(
            """
        ### Distribution Analysis

        The sentiment distribution shows the balance between positive and negative predictions.

        **Key Insights:**
        """
        )

        if positive_count > negative_count:
            ratio = (
                positive_count / negative_count if negative_count > 0 else float("inf")
            )
            st.success(
                f"✅ More positive reviews ({ratio:.1f}x more positive than negative)"
            )
        elif negative_count > positive_count:
            ratio = (
                negative_count / positive_count if positive_count > 0 else float("inf")
            )
            st.error(
                f"⚠️ More negative reviews ({ratio:.1f}x more negative than positive)"
            )
        else:
            st.info("⚖️ Balanced distribution of positive and negative reviews")

        st.markdown(
            f"""
        - **Positive**: {positive_pct:.1f}%
        - **Negative**: {negative_pct:.1f}%
        - **Total**: {prediction_stats['total']} predictions
        """
        )

# Predictions over time
if not predictions_df.empty and len(predictions_df) > 5:
    st.markdown("---")
    st.subheader("📅 Predictions Over Time")

    fig_timeline = create_predictions_timeline(predictions_df)
    st.plotly_chart(fig_timeline, use_container_width=True)

# Confidence analysis
if not predictions_df.empty:
    st.markdown("---")
    st.subheader("🎯 Confidence Score Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_conf = create_confidence_histogram(predictions_df)
        st.plotly_chart(fig_conf, use_container_width=True)

    with col2:
        st.markdown("### Confidence Statistics")

        avg_conf = predictions_df["confidence"].mean()
        median_conf = predictions_df["confidence"].median()
        min_conf = predictions_df["confidence"].min()
        max_conf = predictions_df["confidence"].max()

        st.metric("Average Confidence", f"{avg_conf:.1%}")
        st.metric("Median Confidence", f"{median_conf:.1%}")
        st.metric("Min Confidence", f"{min_conf:.1%}")
        st.metric("Max Confidence", f"{max_conf:.1%}")

        # Confidence ranges
        high_conf = len(predictions_df[predictions_df["confidence"] >= 0.9])
        med_conf = len(
            predictions_df[
                (predictions_df["confidence"] >= 0.7)
                & (predictions_df["confidence"] < 0.9)
            ]
        )
        low_conf = len(predictions_df[predictions_df["confidence"] < 0.7])

        st.markdown("### Confidence Ranges")
        st.markdown(
            f"""
        - **High (≥90%)**: {high_conf} ({high_conf/len(predictions_df)*100:.1f}%)
        - **Medium (70-90%)**: {med_conf} ({med_conf/len(predictions_df)*100:.1f}%)
        - **Low (<70%)**: {low_conf} ({low_conf/len(predictions_df)*100:.1f}%)
        """
        )

# Text length analysis
if not predictions_df.empty:
    st.markdown("---")
    st.subheader("📏 Text Length Analysis")

    fig_length = create_text_length_distribution(predictions_df)
    st.plotly_chart(fig_length, use_container_width=True)

    # Length statistics
    predictions_df["text_length"] = predictions_df["text"].str.len()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Average Length", f"{predictions_df['text_length'].mean():.0f} chars")
    with col2:
        st.metric(
            "Median Length", f"{predictions_df['text_length'].median():.0f} chars"
        )
    with col3:
        st.metric("Min Length", f"{predictions_df['text_length'].min():.0f} chars")
    with col4:
        st.metric("Max Length", f"{predictions_df['text_length'].max():.0f} chars")

# Word clouds
st.markdown("---")
st.subheader("☁️ Word Clouds")

st.info(
    """
Word clouds visualize the most common words in positive and negative reviews.
Larger words appear more frequently in the reviews.
"""
)

if not predictions_df.empty and len(predictions_df) >= 10:
    col1, col2 = st.columns(2)

    # Positive word cloud
    with col1:
        st.markdown("### 🟢 Positive Reviews")

        positive_texts = predictions_df[predictions_df["sentiment"] == "positive"][
            "text"
        ]

        if len(positive_texts) > 0:
            try:
                positive_text = " ".join(positive_texts.astype(str))

                wordcloud_pos = WordCloud(
                    width=800,
                    height=400,
                    background_color="#0E1117",
                    colormap="Greens",
                    max_words=100,
                    relative_scaling=0.5,
                    min_font_size=10,
                ).generate(positive_text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_pos, interpolation="bilinear")
                ax.axis("off")
                fig.patch.set_facecolor("#0E1117")
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.warning(f"Could not generate word cloud: {str(e)}")
        else:
            st.info("No positive reviews yet")

    # Negative word cloud
    with col2:
        st.markdown("### 🔴 Negative Reviews")

        negative_texts = predictions_df[predictions_df["sentiment"] == "negative"][
            "text"
        ]

        if len(negative_texts) > 0:
            try:
                negative_text = " ".join(negative_texts.astype(str))

                wordcloud_neg = WordCloud(
                    width=800,
                    height=400,
                    background_color="#0E1117",
                    colormap="Reds",
                    max_words=100,
                    relative_scaling=0.5,
                    min_font_size=10,
                ).generate(negative_text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_neg, interpolation="bilinear")
                ax.axis("off")
                fig.patch.set_facecolor("#0E1117")
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.warning(f"Could not generate word cloud: {str(e)}")
        else:
            st.info("No negative reviews yet")
else:
    st.info(
        "Need at least 10 predictions to generate word clouds. Make some predictions first!"
    )

# Model usage statistics
st.markdown("---")
st.subheader("🤖 Model Usage Statistics")

if prediction_stats["by_model"]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Distribution")

        model_df = pd.DataFrame(
            list(prediction_stats["by_model"].items()), columns=["Model", "Count"]
        )

        st.dataframe(model_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Usage Percentage")

        total = sum(prediction_stats["by_model"].values())

        for model, count in prediction_stats["by_model"].items():
            percentage = (count / total * 100) if total > 0 else 0
            st.progress(percentage / 100, text=f"{model}: {percentage:.1f}%")

# Sample reviews
st.markdown("---")
st.subheader("📝 Sample Predictions")

if not predictions_df.empty:
    # Show recent samples
    sample_df = predictions_df.head(10).copy()

    # Format for display
    sample_df["text"] = sample_df["text"].apply(
        lambda x: x[:100] + "..." if len(x) > 100 else x
    )
    sample_df["confidence"] = sample_df["confidence"].apply(lambda x: f"{x:.1%}")
    sample_df["timestamp"] = sample_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    display_df = sample_df[
        ["timestamp", "text", "model_name", "sentiment", "confidence"]
    ].rename(
        columns={
            "timestamp": "Time",
            "text": "Review",
            "model_name": "Model",
            "sentiment": "Sentiment",
            "confidence": "Confidence",
        }
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Export insights
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Export Prediction Data", use_container_width=True):
        if not predictions_df.empty:
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="predictions_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning("No data to export")

with col2:
    if st.button("📊 Export Statistics", use_container_width=True):
        import json

        stats_json = json.dumps(prediction_stats, indent=2)
        st.download_button(
            label="Download JSON",
            data=stats_json,
            file_name="prediction_statistics.json",
            mime="application/json",
            use_container_width=True,
        )
