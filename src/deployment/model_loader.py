"""
Model loader and inference engine for DistilBERT sentiment analysis.
Handles model loading, preprocessing, and prediction with GPU support.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from src.preprocessing.text_cleaner import TextCleaner
from src.utils.config import Config
from src.utils.helpers import timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Handles loading and inference for the DistilBERT sentiment analysis model.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            model_path: Path to the trained model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = (
            Path(model_path) if model_path else Config.MODELS_DIR / "distilbert"
        )
        self.device = self._setup_device(device)

        # Model components
        self.model = None
        self.tokenizer = None
        self.text_cleaner = None

        # Model metadata
        self.model_metadata = {}
        self.is_loaded = False

        # Performance tracking
        self.inference_times = []

        logger.info(
            f"ModelLoader initialized - Path: {self.model_path}, Device: {self.device}"
        )

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device."""
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {gpu_name}")
            return torch.device("cuda")
        else:
            logger.info("GPU not available, using CPU")
            return torch.device("cpu")

    @timer
    def load_model(self) -> bool:
        """
        Load the trained DistilBERT model and tokenizer.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Loading DistilBERT model...")

            # Verify model files exist
            if not self._verify_model_files():
                return False

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.model_path))

            # Load model
            logger.info("Loading model weights...")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                str(self.model_path)
            )

            # Move to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Initialize text cleaner
            self.text_cleaner = TextCleaner()

            # Load metadata
            self._load_metadata()

            self.is_loaded = True

            logger.info(f"[SUCCESS] Model loaded successfully on {self.device}")
            logger.info(
                f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            return False

    def _verify_model_files(self) -> bool:
        """Verify all required model files exist."""
        # Files other than weights that must exist
        required_other_files = [
            "config.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ]

        # Check for either pytorch_model.bin or model.safetensors
        has_weights = False
        if (self.model_path / "pytorch_model.bin").exists():
            has_weights = True
        elif (self.model_path / "model.safetensors").exists():
            has_weights = True

        if not has_weights:
            logger.error(
                "Model weights file not found (pytorch_model.bin or model.safetensors)"
            )
            return False

        # Check other required files
        for file_name in required_other_files:
            file_path = self.model_path / file_name
            if not file_path.exists():
                logger.error(f"Required model file not found: {file_path}")
                return False

        logger.info("All required model files verified")
        return True

    def _load_metadata(self):
        """Load model metadata if available."""
        metadata_path = self.model_path / "metadata.json"
        if metadata_path.exists():
            try:
                import json

                with open(metadata_path, "r") as f:
                    self.model_metadata = json.load(f)
                logger.info("Model metadata loaded")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {str(e)}")

    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for a single text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Clean text
            cleaned_text = self.text_cleaner.clean(text, preserve_case=True)

            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][prediction].item()

            # Map prediction to label
            sentiment = "positive" if prediction == 1 else "negative"

            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            result = {
                "text": text,
                "cleaned_text": cleaned_text,
                "sentiment": sentiment,
                "confidence": confidence,
                "prediction_probabilities": {
                    "negative": probabilities[0][0].item(),
                    "positive": probabilities[0][1].item(),
                },
                "inference_time": inference_time,
                "model": "distilbert",
            }

            logger.debug(f"Prediction: {sentiment} (confidence: {confidence:.4f})")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_batch(
        self, texts: List[str], batch_size: int = 16
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for multiple texts with batching.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process in each batch

        Returns:
            List of prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if len(texts) > 1000:  # Maximum limit
            raise ValueError(
                f"Batch size too large: {len(texts)}. Maximum allowed: 1000"
            )

        logger.info(f"Processing batch of {len(texts)} texts...")
        start_time = time.time()

        results = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_results = self._process_batch(batch_texts)
                results.extend(batch_results)

            total_time = time.time() - start_time
            avg_time_per_text = total_time / len(texts)

            logger.info(
                f"Batch processing complete - Total: {total_time:.2f}s, "
                f"Avg per text: {avg_time_per_text:.3f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise

    def _process_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Process a single batch of texts."""
        # Clean texts
        cleaned_texts = [
            self.text_cleaner.clean(text, preserve_case=True) for text in texts
        ]

        # Tokenize batch
        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Batch inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        # Process results
        results = []
        for i, (text, cleaned_text) in enumerate(zip(texts, cleaned_texts)):
            prediction = predictions[i].item()
            confidence = probabilities[i][prediction].item()
            sentiment = "positive" if prediction == 1 else "negative"

            result = {
                "text": text,
                "cleaned_text": cleaned_text,
                "sentiment": sentiment,
                "confidence": confidence,
                "prediction_probabilities": {
                    "negative": probabilities[i][0].item(),
                    "positive": probabilities[i][1].item(),
                },
                "model": "distilbert",
            }

            results.append(result)

        return results

    def get_model_info(self) -> Dict:
        """Get model information and metadata."""
        if not self.is_loaded:
            return {"error": "Model not loaded"}

        info = {
            "model_name": "distilbert",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "tokenizer_vocab_size": len(self.tokenizer.vocab) if self.tokenizer else 0,
            "max_sequence_length": 512,
            "metadata": self.model_metadata,
        }

        # Add performance stats
        if self.inference_times:
            info["performance"] = {
                "total_predictions": len(self.inference_times),
                "avg_inference_time": sum(self.inference_times)
                / len(self.inference_times),
                "min_inference_time": min(self.inference_times),
                "max_inference_time": max(self.inference_times),
            }

        return info

    def health_check(self) -> Dict[str, Union[bool, str]]:
        """Perform health check on the model."""
        try:
            if not self.is_loaded:
                return {"healthy": False, "error": "Model not loaded"}

            # Test prediction with simple text
            test_result = self.predict_single("This is a test.")

            # Verify result structure
            required_keys = ["sentiment", "confidence", "model"]
            for key in required_keys:
                if key not in test_result:
                    return {
                        "healthy": False,
                        "error": f"Missing key in prediction: {key}",
                    }

            # Verify confidence is reasonable
            if not (0 <= test_result["confidence"] <= 1):
                return {"healthy": False, "error": "Invalid confidence score"}

            return {
                "healthy": True,
                "model_loaded": True,
                "device": str(self.device),
                "test_prediction": test_result["sentiment"],
            }

        except Exception as e:
            return {"healthy": False, "error": str(e)}


# Global model loader instance
_model_loader = None


def get_model_loader() -> ModelLoader:
    """Get or create global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        if not _model_loader.load_model():
            raise RuntimeError("Failed to load model")
    return _model_loader


def initialize_model(model_path: Optional[str] = None) -> bool:
    """Initialize the global model loader."""
    global _model_loader
    _model_loader = ModelLoader(model_path)
    return _model_loader.load_model()


if __name__ == "__main__":
    # Test model loading
    print("Testing DistilBERT Model Loader...")

    loader = ModelLoader()

    if loader.load_model():
        print("✅ Model loaded successfully!")

        # Test single prediction
        result = loader.predict_single("This movie was absolutely amazing!")
        print(
            f"Test prediction: {result['sentiment']} (confidence: {result['confidence']:.4f})"
        )

        # Test batch prediction
        test_texts = ["Great movie!", "Terrible film.", "It was okay, I guess."]
        batch_results = loader.predict_batch(test_texts)
        print(f"Batch predictions: {len(batch_results)} results")

        # Health check
        health = loader.health_check()
        print(f"Health check: {health}")

        # Model info
        info = loader.get_model_info()
        print(f"Model parameters: {info['parameters']:,}")

    else:
        print("❌ Failed to load model")
