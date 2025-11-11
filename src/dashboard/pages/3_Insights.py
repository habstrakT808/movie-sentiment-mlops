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

# Title
st.title("📈 Data Insights & Analytics")
st.markdown(
    """
Explore dataset statistics, sentiment distributions, and prediction patterns.
"""
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
