"""
Prediction page for sentiment analysis.
"""

import time

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

# Title
st.title("🎯 Predict Movie Review Sentiment")
st.markdown(
    """
Enter a movie review below and let our AI models analyze its sentiment!
"""
)

# Main prediction interface
st.markdown("---")

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📝 Enter Movie Review")

    # Text input
    review_text = st.text_area(
        "Review Text",
        height=200,
        placeholder=(
            "Example: This movie was absolutely amazing! The acting was superb and "
            "the plot kept me engaged throughout..."
        ),
        help="Enter a movie review (minimum 10 characters, maximum 5000 characters)",
        label_visibility="collapsed",
    )

    # Character counter
    char_count = len(review_text)
    if char_count > 0:
        color = "green" if MIN_TEXT_LENGTH <= char_count <= MAX_TEXT_LENGTH else "red"
        st.markdown(
            f"<p style='color: {color}; font-size: 14px;'>Characters: {char_count} / {MAX_TEXT_LENGTH}</p>",
            unsafe_allow_html=True,
        )

with col2:
    st.subheader("⚙️ Settings")

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Choose which model to use for prediction",
    )

    # Display model info
    model_info = AVAILABLE_MODELS[model_name]
    st.info(
        f"""
    **{model_name}**

    {model_info['description']}

    Accuracy: {model_info['accuracy']:.1%}
    """
    )

    # Predict button
    predict_button = st.button(
        "🎯 Predict Sentiment", use_container_width=True, type="primary"
    )

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
            st.markdown("---")
            st.subheader("🎉 Prediction Results")

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
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Confidence Score")
                fig_gauge = create_confidence_gauge(confidence, sentiment)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.subheader("📈 Probability Distribution")
                fig_bars = create_probability_bars(result["probabilities"])
                st.plotly_chart(fig_bars, use_container_width=True)

            # Metadata
            st.markdown("---")
            st.subheader("ℹ️ Prediction Details")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Model Used", model_name)

            with col2:
                st.metric("Inference Time", f"{prediction_time:.3f}s")

            with col3:
                st.metric("Text Length", f"{len(review_text)} chars")

            with col4:
                st.metric("Prediction ID", f"#{prediction_id}")

            # Feedback section
            st.markdown("---")
            st.subheader("💬 Was this prediction helpful?")

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

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            logger.error(f"Prediction error: {str(e)}", exc_info=True)

# Prediction history
st.markdown("---")
st.subheader("📜 Your Prediction History")

db = get_database()
history_df = db.get_recent_predictions(limit=10, session_id=st.session_state.session_id)

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
    st.info("No predictions yet in this session. Make your first prediction above!")

# Example reviews
st.markdown("---")
st.subheader("💡 Try These Example Reviews")

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

col1, col2, col3 = st.columns(3)

for i, (col, example) in enumerate(zip([col1, col2, col3], examples)):
    with col:
        with st.expander(f"Example {i+1} (Expected: {example['expected']})"):
            st.write(example["text"])
            if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                st.session_state.example_text = example["text"]
                st.rerun()

# Apply example text if selected
if "example_text" in st.session_state:
    st.info("Example loaded! Scroll up to see it in the text area.")
    # Note: We can't directly set text_area value, so we'll show instruction
    st.code(st.session_state.example_text)
    del st.session_state.example_text
