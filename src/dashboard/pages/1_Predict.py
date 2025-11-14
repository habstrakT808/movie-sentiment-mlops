"""
Prediction page for sentiment analysis.
"""

import time
import uuid

import streamlit as st

from src.dashboard.components.database import get_database
from src.dashboard.components.model_loader import get_model_loader
from src.dashboard.components.visualizations import (
    create_confidence_gauge,
    create_probability_bars,
)
from src.dashboard.config import (
    AVAILABLE_MODELS,
    MAX_TEXT_LENGTH,
    MIN_TEXT_LENGTH,
    SENTIMENT_COLORS,
    SENTIMENT_EMOJIS,
)
from src.dashboard.utils.helpers import (
    clean_text_for_display,
    format_confidence,
    validate_text_input,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Predict Sentiment", page_icon="🎯", layout="wide")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New session started: {st.session_state.session_id}")

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# Handle example text loading
if "example_text" in st.session_state:
    st.session_state.review_text = st.session_state.example_text
    del st.session_state.example_text

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
        border-color: rgba(102, 126,234, 0.5);
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
    <div class="hero-title">🎯 Predict Movie Review Sentiment</div>
    <div class="hero-subtitle">
        Enter a movie review below and let our AI models analyze its sentiment with state-of-the-art deep learning!
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Main prediction interface
st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

# Input section with glass cards
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
    <div class="glass-card">
        <h3 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">📝 Enter Movie Review</h3>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # Text input
    review_text = st.text_area(
        "Review Text",
        value=st.session_state.review_text,
        height=200,
        placeholder=(
            "Example: This movie was absolutely amazing! The acting was superb and "
            "the plot kept me engaged throughout..."
        ),
        help="Enter a movie review (minimum 10 characters, maximum 5000 characters)",
        label_visibility="collapsed",
        key="review_text_area",
    )

    # Update session state when text changes
    st.session_state.review_text = review_text

    # Character counter
    char_count = len(review_text)
    if char_count > 0:
        color = (
            "#00D26A" if MIN_TEXT_LENGTH <= char_count <= MAX_TEXT_LENGTH else "#FF4B4B"
        )
        st.markdown(
            f"<p style='color: {color}; font-size: 14px; font-weight: 600;'>Characters: {char_count} / {MAX_TEXT_LENGTH}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='color: #B0B0B0; font-size: 14px;'>Characters: 0 / {MAX_TEXT_LENGTH}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
    <div class="glass-card">
        <h3 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">⚙️ Settings</h3>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Choose which model to use for prediction",
    )

    st.markdown(
        "<div style='margin-top: 1rem; margin-bottom: 1rem;'></div>",
        unsafe_allow_html=True,
    )

    # Display model info
    model_info = AVAILABLE_MODELS[model_name]
    st.markdown(
        f"""
    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px;
        border-left: 4px solid #667eea; margin: 1rem 0;">
        <h4 style="color: #667eea; margin-top: 0;">{model_name}</h4>
        <p style="color: #B0B0B0; margin-bottom: 0.5rem; font-size: 0.9rem;">
            {model_info['description']}</p>
        <p style="color: #00D26A; font-weight: 700; margin-bottom: 0;">Accuracy: {model_info['accuracy']:.1%}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Predict button
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    predict_button = st.button(
        "🎯 Predict Sentiment", use_container_width=True, type="primary"
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

# Prediction logic
if predict_button:
    # Validate input
    is_valid, error_msg = validate_text_input(
        review_text, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
    )

    if not is_valid:
        st.error(error_msg)
    else:
        try:
            # Get model loader
            loader = get_model_loader()

            # Check if model is loaded
            if not loader.is_loaded(model_name):
                st.error(f"❌ {model_name} is not loaded. Please check the logs.")
                st.stop()

            # Make prediction
            with st.spinner(f"🔄 Analyzing with {model_name}..."):
                start_time = time.time()
                result = loader.predict(review_text, model_name)
                prediction_time = time.time() - start_time

            # Save to database
            db = get_database()
            prediction_id = db.add_prediction(
                text=review_text,
                model_name=model_name,
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                session_id=st.session_state.session_id,
            )

            # Update session state
            st.session_state.prediction_count += 1
            st.session_state.last_prediction = result

            logger.info(
                f"Prediction {prediction_id} completed: {result['sentiment']} "
                f"({result['confidence']:.2%})"
            )

            # Display results
            st.markdown(
                "<div style='margin-top: 3rem; margin-bottom: 2rem;'></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
            <div style="text-align: center; margin: 2rem 0;">
                <h2 style="color: #FAFAFA; font-weight: 700;">🎉 Prediction Results</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Sentiment badge
            sentiment = result["sentiment"]
            confidence = result["confidence"]

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Large sentiment display
                emoji = SENTIMENT_EMOJIS.get(sentiment, "")
                color = SENTIMENT_COLORS.get(sentiment, "#808080")

                st.markdown(
                    f"""
                <div style="
                    text-align: center;
                    padding: 40px;
                    background: linear-gradient(135deg, {color}40 0%, {color}20 100%);
                    border-radius: 20px;
                    border: 2px solid {color};
                    margin: 20px 0;
                ">
                    <div style="font-size: 80px; margin-bottom: 10px;">{emoji}</div>
                    <div style="font-size: 36px; font-weight: bold; color: {color}; text-transform: uppercase;">
                        {sentiment}
                    </div>
                    <div style="font-size: 24px; color: #FAFAFA; margin-top: 10px;">
                        Confidence: {format_confidence(confidence)}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Detailed results
            st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
            <div class="glass-card">
            """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    "<h3 style='color: #FAFAFA; margin-bottom: 1.5rem;'>📊 Confidence Score</h3>",
                    unsafe_allow_html=True,
                )
                fig_gauge = create_confidence_gauge(confidence, sentiment)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.markdown(
                    "<h3 style='color: #FAFAFA; margin-bottom: 1.5rem;'>📈 Probability Distribution</h3>",
                    unsafe_allow_html=True,
                )
                fig_bars = create_probability_bars(result["probabilities"])
                st.plotly_chart(fig_bars, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='margin-top: 2rem; margin-bottom: 2rem;'></div>",
                unsafe_allow_html=True,
            )

            # Metadata
            st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
            <div class="glass-card">
                <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">ℹ️ Prediction Details</h2>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Model Used", model_name)

            with col2:
                st.metric("Inference Time", f"{prediction_time:.3f}s")

            with col3:
                st.metric("Text Length", f"{len(review_text)} chars")

            with col4:
                st.metric("Prediction ID", f"#{prediction_id}")

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='margin-top: 2rem; margin-bottom: 2rem;'></div>",
                unsafe_allow_html=True,
            )

            # Feedback section
            st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
            <div class="glass-card">
                <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">💬 Was this prediction helpful?</h2>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns([1, 1, 4])

            with col1:
                if st.button("👍 Yes, helpful", use_container_width=True):
                    db.update_feedback(prediction_id, 1)
                    st.success("Thank you for your feedback!")
                    logger.info(f"Positive feedback for prediction {prediction_id}")

            with col2:
                if st.button("👎 Not helpful", use_container_width=True):
                    db.update_feedback(prediction_id, -1)
                    st.info("Thank you for your feedback! We'll use it to improve.")
                    logger.info(f"Negative feedback for prediction {prediction_id}")

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            logger.error(f"Prediction error: {str(e)}", exc_info=True)

# Prediction history
st.markdown(
    "<div style='margin-top: 3rem; margin-bottom: 2rem;'></div>", unsafe_allow_html=True
)
st.markdown(
    """
<div class="glass-card" style="margin-bottom: 2rem;">
    <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">📜 Your Prediction History</h2>
""",
    unsafe_allow_html=True,
)

db = get_database()

# Option to filter by session or show all
# Default to showing all predictions (like home page) for consistency
show_all = st.checkbox("Show all predictions (not just this session)", value=True)

if show_all:
    # Show all recent predictions (like home page)
    history_df = db.get_recent_predictions(limit=10)
else:
    # Show only current session predictions
    history_df = db.get_recent_predictions(
        limit=10, session_id=st.session_state.session_id
    )

if not history_df.empty:
    # Format dataframe
    display_df = history_df.copy()
    display_df["text"] = display_df["text"].apply(
        lambda x: clean_text_for_display(x, max_length=80)
    )
    display_df["confidence"] = display_df["confidence"].apply(
        lambda x: format_confidence(x)
    )
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")

    # Add emoji to sentiment
    display_df["sentiment"] = display_df["sentiment"].apply(
        lambda x: f"{SENTIMENT_EMOJIS.get(x, '')} {x.capitalize()}"
    )

    # Rename columns
    display_df = display_df[
        ["timestamp", "text", "model_name", "sentiment", "confidence"]
    ].rename(
        columns={
            "timestamp": "Time",
            "text": "Review Preview",
            "model_name": "Model",
            "sentiment": "Sentiment",
            "confidence": "Confidence",
        }
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Export button
    if st.button("📥 Export History to CSV"):
        csv = db.export_all().to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )
else:
    st.markdown(
        """
    <div style="background: rgba(102, 126, 234, 0.1); padding: 2rem; border-radius: 10px;
        border-left: 4px solid #667eea; text-align: center;">
        <p style="color: #B0B0B0; font-size: 1.1rem; margin: 0;">
            No predictions yet in this session. Make your first prediction above! 🚀
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# Example reviews
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
st.markdown(
    """
<div class="glass-card">
    <h2 style="margin-top: 0; color: #FAFAFA; margin-bottom: 1.5rem;">💡 Try These Example Reviews</h2>
""",
    unsafe_allow_html=True,
)

examples = [
    {
        "text": (
            "This movie was absolutely phenomenal! The cinematography was breathtaking, "
            "and the performances were Oscar-worthy. A must-watch masterpiece that "
            "will stay with you long after the credits roll."
        ),
        "expected": "Positive",
    },
    {
        "text": (
            "What a waste of time and money. The plot was predictable, the acting was "
            "wooden, and the special effects looked like they were from the 90s. "
            "I want my two hours back!"
        ),
        "expected": "Negative",
    },
    {
        "text": (
            "An incredible journey from start to finish. The director's vision combined "
            "with stellar performances created something truly special. The soundtrack "
            "alone is worth the price of admission."
        ),
        "expected": "Positive",
    },
]

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

for i, (col, example) in enumerate(zip([col1, col2, col3], examples)):
    with col:
        expected_color = "#00D26A" if example["expected"] == "Positive" else "#FF4B4B"
        st.markdown(
            f"""
        <div style="background: rgba(38, 39, 48, 0.5); padding: 1.5rem; border-radius: 15px;
            border: 1px solid rgba(102, 126, 234, 0.2); margin-bottom: 1.5rem;">
            <h4 style="color: {expected_color}; margin-top: 0; margin-bottom: 1rem;">
                Example {i+1}</h4>
            <p style="color: #B0B0B0; font-size: 0.9rem; margin-bottom: 1rem; line-height: 1.6;">
                {example["text"]}</p>
            <p style="color: #667eea; font-size: 0.85rem; font-weight: 600; margin-bottom: 1.5rem;">
                Expected: {example['expected']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Use Example {i+1}", key=f"example_{i}", use_container_width=True
        ):
            st.session_state.review_text = example["text"]
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
