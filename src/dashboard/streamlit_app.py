"""
Main Streamlit application for Movie Sentiment Analysis.
"""

import uuid

import streamlit as st

from src.dashboard.components.database import get_database
from src.dashboard.components.model_loader import get_model_loader, initialize_models
from src.dashboard.config import (
    APP_ICON,
    APP_TITLE,
    AVAILABLE_MODELS,
    DISTILBERT_PATH,
    LAYOUT,
    LOGISTIC_REGRESSION_PATH,
    PAGE_TITLE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/habstrakT808/movie-sentiment-mlops",
        "Report a bug": "https://github.com/habstrakT808/movie-sentiment-mlops/issues",
        "About": """
        # Movie Sentiment Analysis

        **Version:** 1.0.0

        State-of-the-art sentiment analysis for movie reviews using DistilBERT.

        **Models:**
        - DistilBERT: 92.50% accuracy
        - Logistic Regression: 87.40% accuracy

        **GitHub:** https://github.com/habstrakT808/movie-sentiment-mlops
        """,
    },
)

# Custom CSS
st.markdown(
    """
<style>
    /* Main theme */
    .main {
        background-color: #0E1117;
    }

    /* Headers */
    h1 {
        color: #FAFAFA;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 2px solid #FF4B4B;
    }

    h2, h3 {
        color: #FAFAFA;
        font-weight: 600;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        color: white;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Text input */
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #464646;
        border-radius: 5px;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 5px;
    }

    /* Dataframe */
    .dataframe {
        background-color: #262730;
        color: #FAFAFA;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #262730;
    }

    /* Success/Error boxes */
    .stSuccess {
        background-color: rgba(0, 210, 106, 0.1);
        border-left: 4px solid #00D26A;
    }

    .stError {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 4px solid #FF4B4B;
    }

    .stInfo {
        background-color: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
    }

    .stWarning {
        background-color: rgba(255, 165, 0, 0.1);
        border-left: 4px solid #FFA500;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"New session started: {st.session_state.session_id}")

    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False

    if "prediction_count" not in st.session_state:
        st.session_state.prediction_count = 0

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None


def load_models():
    """Load models if not already loaded."""
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading models... This may take a moment..."):
            try:
                distilbert_ok, logistic_ok = initialize_models(
                    DISTILBERT_PATH, LOGISTIC_REGRESSION_PATH
                )

                if distilbert_ok:
                    st.session_state.models_loaded = True
                    logger.info("Models loaded successfully")
                    return True
                else:
                    st.error("❌ Failed to load models. Please check the logs.")
                    logger.error("Failed to load models")
                    return False

            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                logger.error(f"Error loading models: {str(e)}")
                return False

    return True


def display_sidebar():
    """Display sidebar with app information."""
    with st.sidebar:
        st.title(APP_TITLE)
        st.markdown("---")

        # Model status
        st.subheader("🤖 Model Status")

        loader = get_model_loader()
        loaded_models = loader.get_loaded_models()

        for model_name, model_info in AVAILABLE_MODELS.items():
            if model_name in loaded_models:
                st.success(f"✅ {model_name}")
                st.caption(f"Accuracy: {model_info['accuracy']:.1%}")
            else:
                st.warning(f"⚠️ {model_name} (Not loaded)")

        st.markdown("---")

        # Statistics
        st.subheader("📊 Statistics")

        db = get_database()
        stats = db.get_statistics()

        st.metric("Total Predictions", stats["total"])
        st.metric("Today's Predictions", stats["today"])

        if stats["total"] > 0:
            st.metric("Average Confidence", f"{stats['avg_confidence']:.1%}")

        st.markdown("---")

        # Session info
        st.subheader("ℹ️ Session Info")
        st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
        st.caption(f"Predictions this session: {st.session_state.prediction_count}")

        st.markdown("---")

        # Links
        st.subheader("🔗 Links")
        st.markdown(
            """
        - [GitHub Repository](https://github.com/habstrakT808/movie-sentiment-mlops)
        - [Documentation](https://github.com/habstrakT808/movie-sentiment-mlops/blob/main/README.md)
        - [Report Issue](https://github.com/habstrakT808/movie-sentiment-mlops/issues)
        """
        )

        st.markdown("---")

        # Footer
        st.caption("Made with ❤️ using Streamlit")
        st.caption("Version 1.0.0")


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Display sidebar
    display_sidebar()

    # Load models
    if not load_models():
        st.stop()

    # Main content
    st.title("🎬 Movie Sentiment Analysis")
    st.markdown(
        """
    Welcome to the **Movie Sentiment Analysis Dashboard**! This application uses
    state-of-the-art machine learning models to analyze the sentiment of movie reviews.
    """
    )

    st.info(
        """
    👈 **Navigate using the sidebar** to access different features:
    - **🎯 Predict**: Analyze sentiment of movie reviews
    - **📊 Performance**: View model performance metrics
    - **📈 Insights**: Explore data insights and statistics
    """
    )

    # Quick stats
    st.markdown("---")
    st.subheader("📊 Quick Statistics")

    db = get_database()
    stats = db.get_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="font-size: 14px; opacity: 0.8;">Total Predictions</div>
            <div style="font-size: 32px; font-weight: bold;">{stats['total']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        positive_count = stats["by_sentiment"].get("positive", 0)
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="font-size: 14px; opacity: 0.8;">Positive Reviews</div>
            <div style="font-size: 32px; font-weight: bold;">{positive_count}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        negative_count = stats["by_sentiment"].get("negative", 0)
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="font-size: 14px; opacity: 0.8;">Negative Reviews</div>
            <div style="font-size: 32px; font-weight: bold;">{negative_count}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="font-size: 14px; opacity: 0.8;">Avg. Confidence</div>
            <div style="font-size: 32px; font-weight: bold;">{stats['avg_confidence']:.1%}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Recent predictions
    if stats["total"] > 0:
        st.markdown("---")
        st.subheader("🕐 Recent Predictions")

        recent_df = db.get_recent_predictions(limit=5)

        if not recent_df.empty:
            # Format dataframe for display
            display_df = recent_df.copy()
            display_df["text"] = display_df["text"].apply(
                lambda x: x[:50] + "..." if len(x) > 50 else x
            )
            display_df["confidence"] = display_df["confidence"].apply(
                lambda x: f"{x:.1%}"
            )
            display_df["timestamp"] = display_df["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )

            # Rename columns
            display_df = display_df[
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
        else:
            st.info(
                "No predictions yet. Go to the Predict page to make your first prediction!"
            )

    # Getting started
    st.markdown("---")
    st.subheader("🚀 Getting Started")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### How to Use

        1. **Navigate to Predict page** using the sidebar
        2. **Enter a movie review** in the text area
        3. **Select a model** (DistilBERT or Logistic Regression)
        4. **Click "Predict Sentiment"** to analyze
        5. **View results** with confidence scores
        6. **Provide feedback** to help improve the model
        """
        )

    with col2:
        st.markdown(
            """
        ### Available Models

        **🤖 DistilBERT**
        - State-of-the-art transformer model
        - 92.50% accuracy
        - Best for complex reviews

        **📊 Logistic Regression**
        - Fast and efficient baseline
        - 87.40% accuracy
        - Good for quick predictions
        """
        )

    # Tips
    st.markdown("---")
    st.subheader("💡 Tips")

    st.info(
        """
    - **Longer reviews** generally produce more accurate predictions
    - **DistilBERT** performs better on nuanced or sarcastic reviews
    - **Logistic Regression** is faster for quick sentiment checks
    - Check the **Performance page** to compare model metrics
    - Explore the **Insights page** for data visualizations
    """
    )


if __name__ == "__main__":
    main()
