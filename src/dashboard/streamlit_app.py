"""
Main Streamlit application for Movie Sentiment Analysis.
Enhanced with stunning visuals and animations.
"""

import uuid

import streamlit as st

from src.dashboard.components.database import get_database
from src.dashboard.components.model_loader import get_model_loader, initialize_models
from src.dashboard.config import (
    APP_ICON,
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

    /* Metric Cards with Gradient */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin: 10px 0;
        color: white;
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s;
    }

    .metric-card:hover::after {
        opacity: 1;
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(102, 126, 234, 0.4);
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-icon {
        font-size: 2.5rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }

    /* Feature Cards */
    .feature-card {
        background: rgba(38, 39, 48, 0.5);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
    }

    .feature-card:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
    }

    .feature-description {
        font-size: 0.9rem;
        color: #B0B0B0;
        line-height: 1.6;
    }

    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        animation: fadeIn 0.5s ease-out;
    }

    .status-success {
        background: rgba(0, 210, 106, 0.2);
        color: #00D26A;
        border: 1px solid rgba(0, 210, 106, 0.3);
    }

    .status-warning {
        background: rgba(255, 165, 0, 0.2);
        color: #FFA500;
        border: 1px solid rgba(255, 165, 0, 0.3);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2s ease-in-out infinite;
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

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Navigation Cards */
    .nav-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        padding: 2rem;
        border: 2px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
        height: 100%;
    }

    .nav-card:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-color: rgba(102, 126, 234, 0.6);
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    }

    .nav-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 2s ease-in-out infinite;
    }

    .nav-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
    }

    .nav-description {
        font-size: 1rem;
        color: #B0B0B0;
    }

    /* Dataframe Styling */
    .dataframe {
        background-color: rgba(38, 39, 48, 0.5) !important;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar Enhancements */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0E1117 100%);
    }

    [data-testid="stSidebar"] .element-container {
        animation: fadeInLeft 0.5s ease-out;
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

    .stWarning {
        background: rgba(255, 165, 0, 0.1);
        border-left: 4px solid #FFA500;
        border-radius: 8px;
        animation: slideInRight 0.5s ease-out;
    }

    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
    }

    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
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
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }

    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }

    @keyframes shake {
        0%, 100% {
            transform: translateX(0);
        }
        25% {
            transform: translateX(-10px);
        }
        75% {
            transform: translateX(10px);
        }
    }

    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
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

    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }

        .hero-subtitle {
            font-size: 1rem;
        }

        .metric-value {
            font-size: 2rem;
        }
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
        with st.spinner("🔄 Loading AI models... This may take a moment..."):
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
    """Display enhanced sidebar with app information."""
    with st.sidebar:
        # Logo/Title Section
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎬</div>
            <h2 style="margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Movie Sentiment
            </h2>
            <p style="color: #B0B0B0; font-size: 0.9rem; margin-top: 0.5rem;">
                AI-Powered Analysis
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Model Status Section
        st.markdown("### 🤖 Model Status")

        loader = get_model_loader()
        loaded_models = loader.get_loaded_models()

        for model_name, model_info in AVAILABLE_MODELS.items():
            if model_name in loaded_models:
                st.markdown(
                    f"""
                <div class="status-indicator status-success">
                    <span class="status-dot"></span>
                    <span>{model_name}</span>
                </div>
                <div style="margin-left: 1.5rem; margin-bottom: 1rem; font-size: 0.85rem; color: #B0B0B0;">
                    Accuracy: <strong style="color: #00D26A;">{model_info['accuracy']:.1%}</strong>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="status-indicator status-warning">
                    <span class="status-dot"></span>
                    <span>{model_name}</span>
                </div>
                <div style="margin-left: 1.5rem; margin-bottom: 1rem; font-size: 0.85rem; color: #B0B0B0;">
                    Not loaded
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Statistics Section
        st.markdown("### 📊 Session Stats")

        db = get_database()
        stats = db.get_statistics()

        # Total Predictions
        st.markdown(
            f"""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem;">
            <div style="font-size: 0.8rem; color: #B0B0B0; text-transform: uppercase; letter-spacing: 1px;">
                Total Predictions
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: #667eea;">
                {stats['total']}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Today's Predictions
        st.markdown(
            f"""
        <div style="background: rgba(0, 210, 106, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem;">
            <div style="font-size: 0.8rem; color: #B0B0B0; text-transform: uppercase; letter-spacing: 1px;">
                Today
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: #00D26A;">
                {stats['today']}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if stats["total"] > 0:
            st.markdown(
                f"""
            <div style="background: rgba(255, 165, 0, 0.1); padding: 1rem; border-radius: 10px;">
                <div style="font-size: 0.8rem; color: #B0B0B0; text-transform: uppercase; letter-spacing: 1px;">
                    Avg Confidence
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #FFA500;">
                    {stats['avg_confidence']:.0%}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Session Info
        st.markdown("### ℹ️ Session Info")
        st.caption(f"🆔 Session: {st.session_state.session_id[:8]}...")
        st.caption(f"🎯 Predictions: {st.session_state.prediction_count}")

        st.markdown("---")

        # Quick Links
        st.markdown("### 🔗 Quick Links")
        st.markdown(
            """
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <a href="https://github.com/habstrakT808/movie-sentiment-mlops" target="_blank"
               style="color: #667eea; text-decoration: none; padding: 0.5rem; background: rgba(102, 126, 234, 0.1);
               border-radius: 8px; transition: all 0.3s;">
                📦 GitHub Repository
            </a>
            <a href="https://github.com/habstrakT808/movie-sentiment-mlops/blob/main/README.md" target="_blank"
               style="color: #667eea; text-decoration: none; padding: 0.5rem; background: rgba(102, 126, 234, 0.1);
               border-radius: 8px; transition: all 0.3s;">
                📚 Documentation
            </a>
            <a href="https://github.com/habstrakT808/movie-sentiment-mlops/issues" target="_blank"
               style="color: #667eea; text-decoration: none; padding: 0.5rem; background: rgba(102, 126, 234, 0.1);
               border-radius: 8px; transition: all 0.3s;">
                🐛 Report Issue
            </a>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Footer
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0; color: #666;">
            <p style="font-size: 0.8rem; margin: 0;">Made with ❤️ using</p>
            <p style="font-size: 0.9rem; font-weight: 600; margin: 0.25rem 0;">Streamlit & PyTorch</p>
            <p style="font-size: 0.75rem; color: #555; margin-top: 0.5rem;">Version 1.0.0</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Display sidebar
    display_sidebar()

    # Load models
    if not load_models():
        st.stop()

    # Hero Section
    st.markdown(
        """
    <div class="hero-section">
        <div class="hero-title">🎬 Movie Sentiment Analysis</div>
        <div class="hero-subtitle">
            Harness the power of AI to analyze movie reviews with state-of-the-art deep learning models
        </div>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; position: relative; z-index: 1;">
            <div style="background: rgba(0, 210, 106, 0.2); padding: 0.5rem 1.5rem; border-radius: 20px; border: 1px solid rgba(0, 210, 106, 0.3);">  # noqa: E501
                <span style="color: #00D26A; font-weight: 700;">✓ 92.50% Accuracy</span>
            </div>
            <div style="background: rgba(102, 126, 234, 0.2); padding: 0.5rem 1.5rem; border-radius: 20px; border: 1px solid rgba(102, 126, 234, 0.3);">  # noqa: E501
                <span style="color: #667eea; font-weight: 700;">⚡ Real-time Analysis</span>
            </div>
            <div style="background: rgba(255, 165, 0, 0.2); padding: 0.5rem 1.5rem; border-radius: 20px; border: 1px solid rgba(255, 165, 0, 0.3);">  # noqa: E501
                <span style="color: #FFA500; font-weight: 700;">🚀 2 AI Models</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Quick Navigation
    st.markdown("## 🚀 Quick Start")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="nav-card">
            <div class="nav-icon">🎯</div>
            <div class="nav-title">Predict</div>
            <div class="nav-description">Analyze movie review sentiment instantly</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="nav-card">
            <div class="nav-icon">📊</div>
            <div class="nav-title">Performance</div>
            <div class="nav-description">Compare model metrics & accuracy</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="nav-card">
            <div class="nav-icon">📈</div>
            <div class="nav-title">Insights</div>
            <div class="nav-description">Explore data analytics & trends</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.info(
        "👈 **Use the sidebar** to navigate between different features of the dashboard"
    )

    # Live Statistics
    st.markdown("---")
    st.markdown("## 📊 Live Statistics")

    db = get_database()
    stats = db.get_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">{stats['total']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        positive_count = stats["by_sentiment"].get("positive", 0)
        st.markdown(
            f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #00D26A 0%, #00A854 100%);">
            <div class="metric-icon">😊</div>
            <div class="metric-label">Positive Reviews</div>
            <div class="metric-value">{positive_count}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        negative_count = stats["by_sentiment"].get("negative", 0)
        st.markdown(
            f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #FF4B4B 0%, #CC3939 100%);">
            <div class="metric-icon">😞</div>
            <div class="metric-label">Negative Reviews</div>
            <div class="metric-value">{negative_count}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #FFA500 0%, #CC8400 100%);">
            <div class="metric-icon">🎓</div>
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-value">{stats['avg_confidence']:.0%}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Recent Activity
    if stats["total"] > 0:
        st.markdown("---")
        st.markdown("## 🕐 Recent Activity")

        recent_df = db.get_recent_predictions(limit=5)

        if not recent_df.empty:
            # Format dataframe for display
            display_df = recent_df.copy()
            display_df["text"] = display_df["text"].apply(
                lambda x: x[:60] + "..." if len(x) > 60 else x
            )
            display_df["confidence"] = display_df["confidence"].apply(
                lambda x: f"{x:.1%}"
            )
            display_df["timestamp"] = display_df["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )

            # Add emoji to sentiment
            display_df["sentiment"] = display_df["sentiment"].apply(
                lambda x: f"{'😊' if x == 'positive' else '😞'} {x.capitalize()}"
            )

            # Rename columns
            display_df = display_df[
                ["timestamp", "text", "model_name", "sentiment", "confidence"]
            ].rename(
                columns={
                    "timestamp": "⏰ Time",
                    "text": "📝 Review",
                    "model_name": "🤖 Model",
                    "sentiment": "🎭 Sentiment",
                    "confidence": "📊 Confidence",
                }
            )

            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent activity. Start making predictions!")

    # Features Section
    st.markdown("---")
    st.markdown("## ✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">Dual AI Models</div>
            <div class="feature-description">
                Choose between DistilBERT (92.50% accuracy) for complex analysis
                or Logistic Regression (87.40%) for quick predictions
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Real-time Analysis</div>
            <div class="feature-description">
                Get instant sentiment predictions with confidence scores
                and detailed probability distributions
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Performance Metrics</div>
            <div class="feature-description">
                Compare models with detailed metrics including accuracy,
                precision, recall, F1-score, and ROC AUC
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <span class="feature-icon">📈</span>
            <div class="feature-title">Data Insights</div>
            <div class="feature-description">
                Explore sentiment distributions, confidence analysis,
                and word clouds with interactive visualizations
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="feature-card">
            <span class="feature-icon">💾</span>
            <div class="feature-title">Prediction History</div>
            <div class="feature-description">
                Track all your predictions with timestamps, export to CSV,
                and analyze patterns over time
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="feature-card">
            <span class="feature-icon">🔒</span>
            <div class="feature-title">Privacy First</div>
            <div class="feature-description">
                All processing happens locally. No data sent to external servers.
                Your reviews stay private and secure
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # How It Works
    st.markdown("---")
    st.markdown("## 🔬 How It Works")

    st.markdown(
        """
    <div class="glass-card">
        <h3 style="margin-top: 0;">Simple 3-Step Process</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 2rem;">
            <div>
                <div style="font-size: 3rem; margin-bottom: 1rem;">1️⃣</div>
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">Enter Review</h4>
                <p style="color: #B0B0B0; line-height: 1.6;">
                    Type or paste any movie review (10-5000 characters).
                    The longer the review, the better the analysis.
                </p>
            </div>
            <div>
                <div style="font-size: 3rem; margin-bottom: 1rem;">2️⃣</div>
                <h4 style="color: #764ba2; margin-bottom: 0.5rem;">Select Model</h4>
                <p style="color: #B0B0B0; line-height: 1.6;">
                    Choose DistilBERT for accuracy or Logistic Regression for speed.
                    Each model has its strengths.
                </p>
            </div>
            <div>
                <div style="font-size: 3rem; margin-bottom: 1rem;">3️⃣</div>
                <h4 style="color: #f093fb; margin-bottom: 0.5rem;">Get Results</h4>
                <p style="color: #B0B0B0; line-height: 1.6;">
                    Receive instant sentiment analysis with confidence scores,
                    probability distribution, and detailed insights.
                </p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Model Comparison
    st.markdown("---")
    st.markdown("## 🤖 Our AI Models")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="margin-top: 0; color: #667eea;">🧠 DistilBERT</h3>
            <div style="margin: 1.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #B0B0B0;">Accuracy</span>
                    <span style="color: #00D26A; font-weight: 700;">92.50%</span>
                </div>
                <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden;">  # noqa: E501
                    <div style="width: 92.5%; height: 100%; background: linear-gradient(90deg, #00D26A, #00A854);"></div>
                </div>
            </div>
            <p style="color: #B0B0B0; line-height: 1.6; margin-bottom: 1rem;">
                State-of-the-art transformer model with 66M parameters.
                Excellent at understanding context, nuance, and complex language patterns.
            </p>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <span style="background: rgba(0, 210, 106, 0.2); color: #00D26A; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem;">  # noqa: E501
                    ✓ Best Accuracy
                </span>
                <span style="background: rgba(0, 210, 106, 0.2); color: #00D26A; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem;">  # noqa: E501
                    ✓ Context Aware
                </span>
                <span style="background: rgba(0, 210, 106, 0.2); color: #00D26A; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem;">  # noqa: E501
                    ✓ Handles Sarcasm
                </span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="margin-top: 0; color: #764ba2;">📊 Logistic Regression</h3>
            <div style="margin: 1.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #B0B0B0;">Accuracy</span>
                    <span style="color: #FFA500; font-weight: 700;">87.40%</span>
                </div>
                <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden;">  # noqa: E501
                    <div style="width: 87.4%; height: 100%; background: linear-gradient(90deg, #FFA500, #CC8400);"></div>
                </div>
            </div>
            <p style="color: #B0B0B0; line-height: 1.6; margin-bottom: 1rem;">
                Fast and efficient baseline model using TF-IDF features.
                Perfect for quick sentiment checks and batch processing.
            </p>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <span style="background: rgba(255, 165, 0, 0.2); color: #FFA500; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem;">  # noqa: E501
                    ⚡ Fast
                </span>
                <span style="background: rgba(255, 165, 0, 0.2); color: #FFA500; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem;">  # noqa: E501
                    💾 Lightweight
                </span>
                <span style="background: rgba(255, 165, 0, 0.2); color: #FFA500; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem;">  # noqa: E501
                    📈 Reliable
                </span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Call to Action
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 20px; border: 2px solid rgba(102, 126, 234, 0.3);">  # noqa: E501
        <h2 style="margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">  # noqa: E501
            Ready to Analyze Movie Reviews?
        </h2>
        <p style="color: #B0B0B0; font-size: 1.1rem; margin-bottom: 2rem;">
            Start making predictions now and explore the power of AI sentiment analysis
        </p>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
            <a href="/Predict" target="_self" style="text-decoration: none;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; border-radius: 12px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: inline-block;">  # noqa: E501
                    🎯 Start Predicting
                </div>
            </a>
            <a href="/Performance" target="_self" style="text-decoration: none;">
                <div style="background: rgba(102, 126, 234, 0.2); color: #667eea; padding: 1rem 2rem; border-radius: 12px; font-weight: 700; cursor: pointer; transition: all 0.3s; display: inline-block; border: 2px solid #667eea;">  # noqa: E501
                    📊 View Performance
                </div>
            </a>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem 0; color: #666;">
        <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">
            Built with 🔥 using <strong>Streamlit</strong>, <strong>PyTorch</strong>, and <strong>Transformers</strong>
        </p>
        <p style="font-size: 0.85rem; color: #555;">
            © 2024 Movie Sentiment Analysis |
            <a href="https://github.com/habstrakT808/movie-sentiment-mlops" target="_blank" style="color: #667eea; text-decoration: none;">  # noqa: E501
                GitHub
            </a> |
            <a href="mailto:jhodywiraputra@gmail.com" style="color: #667eea; text-decoration: none;">
                Contact
            </a>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
