"""
Model loader for dashboard.
Handles loading and inference for both transformer and traditional models.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from src.preprocessing.text_cleaner import TextCleaner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardModelLoader:
    """
    Model loader optimized for dashboard usage.
    Supports both DistilBERT and traditional ML models.
    """

    def __init__(self):
        """Initialize model loader."""
        self.models = {}
        self.tokenizers = {}
        self.text_cleaner = TextCleaner()
        self.device = self._setup_device()

        logger.info(f"Dashboard model loader initialized on {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for inference")
        return device

    def load_distilbert(self, model_path: Path) -> bool:
        """
        Load DistilBERT model.

        Args:
            model_path: Path to model directory

        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading DistilBERT from {model_path}...")
            start_time = time.time()

            # Load tokenizer
            self.tokenizers["DistilBERT"] = DistilBertTokenizer.from_pretrained(
                str(model_path)
            )

            # Load model
            model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
            model.to(self.device)
            model.eval()

            self.models["DistilBERT"] = model

            load_time = time.time() - start_time
            logger.info(f"✅ DistilBERT loaded successfully in {load_time:.2f}s")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load DistilBERT: {str(e)}")
            return False

    def load_logistic_regression(self, model_path: Path) -> bool:
        """
        Load Logistic Regression model.

        Args:
            model_path: Path to model directory

        Returns:
            True if successful
        """
        try:
            import joblib

            logger.info(f"Loading Logistic Regression from {model_path}...")
            start_time = time.time()

            # Load model
            model_file = model_path / "model.pkl"
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return False

            model = joblib.load(model_file)
            self.models["Logistic Regression"] = model

            # Load vectorizer
            vectorizer_file = model_path / "vectorizer.pkl"
            if vectorizer_file.exists():
                vectorizer = joblib.load(vectorizer_file)
                self.tokenizers["Logistic Regression"] = vectorizer
            else:
                logger.warning("Vectorizer not found, will use global TF-IDF")

            load_time = time.time() - start_time
            logger.info(
                f"✅ Logistic Regression loaded successfully in {load_time:.2f}s"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load Logistic Regression: {str(e)}")
            return False

    def predict(self, text: str, model_name: str = "DistilBERT") -> Dict:
        """
        Make prediction.

        Args:
            text: Input text
            model_name: Model to use

        Returns:
            Prediction result dictionary
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        start_time = time.time()

        try:
            # Clean text
            cleaned_text = self.text_cleaner.clean(text, preserve_case=True)

            # Get prediction based on model type
            if model_name == "DistilBERT":
                result = self._predict_distilbert(cleaned_text)
            else:
                result = self._predict_traditional(cleaned_text, model_name)

            # Add metadata
            result["original_text"] = text
            result["cleaned_text"] = cleaned_text
            result["model_name"] = model_name
            result["inference_time"] = time.time() - start_time

            logger.info(
                f"Prediction: {result['sentiment']} "
                f"(confidence: {result['confidence']:.4f}, "
                f"time: {result['inference_time']:.3f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _predict_distilbert(self, text: str) -> Dict:
        """Predict using DistilBERT."""
        model = self.models["DistilBERT"]
        tokenizer = self.tokenizers["DistilBERT"]

        # Tokenize
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()

        # Map to sentiment
        sentiment = "positive" if prediction == 1 else "negative"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": {
                "negative": probabilities[0][0].item(),
                "positive": probabilities[0][1].item(),
            },
        }

    def _predict_traditional(self, text: str, model_name: str) -> Dict:
        """Predict using traditional ML model."""
        model = self.models[model_name]

        # Get vectorizer
        if model_name in self.tokenizers:
            vectorizer = self.tokenizers[model_name]
            X = vectorizer.transform([text])
        else:
            # Use global TF-IDF vectorizer
            from src.models.utils import load_tfidf_vectorizer

            vectorizer = load_tfidf_vectorizer()
            X = vectorizer.transform([text])

        # Predict
        prediction = model.predict(X)[0]

        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            confidence = probabilities[prediction]

            probs_dict = {"negative": probabilities[0], "positive": probabilities[1]}
        else:
            confidence = 1.0
            probs_dict = {
                "negative": 1.0 if prediction == 0 else 0.0,
                "positive": 1.0 if prediction == 1 else 0.0,
            }

        # Map to sentiment
        sentiment = "positive" if prediction == 1 else "negative"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probs_dict,
        }

    def is_loaded(self, model_name: str) -> bool:
        """Check if model is loaded."""
        return model_name in self.models

    def get_loaded_models(self) -> list:
        """Get list of loaded models."""
        return list(self.models.keys())

    def get_model_info(self, model_name: str) -> Dict:
        """Get model information."""
        if model_name not in self.models:
            return {}

        info = {"name": model_name, "loaded": True, "device": str(self.device)}

        if model_name == "DistilBERT":
            model = self.models[model_name]
            info["parameters"] = sum(p.numel() for p in model.parameters())
            info["type"] = "transformer"
        else:
            info["type"] = "traditional"

        return info


# Global model loader instance
_model_loader = None


def get_model_loader() -> DashboardModelLoader:
    """Get or create model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = DashboardModelLoader()
    return _model_loader


def initialize_models(
    distilbert_path: Path, logistic_path: Optional[Path] = None
) -> Tuple[bool, bool]:
    """
    Initialize all models.

    Args:
        distilbert_path: Path to DistilBERT model
        logistic_path: Path to Logistic Regression model

    Returns:
        (distilbert_loaded, logistic_loaded)
    """
    loader = get_model_loader()

    distilbert_loaded = loader.load_distilbert(distilbert_path)

    logistic_loaded = False
    if logistic_path and logistic_path.exists():
        logistic_loaded = loader.load_logistic_regression(logistic_path)

    return distilbert_loaded, logistic_loaded
