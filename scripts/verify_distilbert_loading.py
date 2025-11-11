"""
Script to verify DistilBERT model can be loaded correctly using Transformers library.
This verifies compatibility and ensures the model is ready for deployment.
"""

import sys
from pathlib import Path

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Ensure project root on path before local project imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def verify_model_loading(model_dir: Path = None):
    """
    Verify that DistilBERT model can be loaded correctly.

    Args:
        model_dir: Path to model directory. Defaults to models/distilbert

    Returns:
        bool: True if model loads successfully, False otherwise
    """
    if model_dir is None:
        model_dir = project_root / "models" / "distilbert"

    logger.info("=" * 80)
    logger.info("DISTILBERT MODEL LOADING VERIFICATION")
    logger.info("=" * 80)

    # Check if model directory exists
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return False

    logger.info(f"Model directory: {model_dir}")

    # Check required files
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.txt",
    ]

    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            logger.info(f"✓ Found: {file}")

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    try:
        # Load tokenizer
        logger.info("\n[1/3] Loading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        logger.info("✓ Tokenizer loaded successfully")

        # Load model
        logger.info("\n[2/3] Loading model...")
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        logger.info("✓ Model loaded successfully")

        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        # Move model to device
        model.to(device)
        model.eval()
        logger.info(f"✓ Model moved to {device} and set to eval mode")

        # Test inference
        logger.info("\n[3/3] Testing inference...")
        test_texts = [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "Terrible movie, waste of time. Very disappointing.",
        ]

        for text in test_texts:
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Predict
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).item()

            sentiment_map = {0: "negative", 1: "positive"}
            sentiment = sentiment_map.get(pred, "unknown")
            confidence = probs[0][pred].item()

            logger.info(
                f"  Text: '{text[:50]}...'"
                f"  → Sentiment: {sentiment} (confidence: {confidence:.4f})"
            )

        logger.info("\n" + "=" * 80)
        logger.info("[SUCCESS] Model loading verification PASSED!")
        logger.info("=" * 80)
        logger.info("\nModel is ready for deployment!")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Tokenizer type: {type(tokenizer).__name__}")
        logger.info(f"Model device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return True

    except Exception as e:
        logger.error(f"\n[ERROR] Model loading failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = verify_model_loading()
    sys.exit(0 if success else 1)
