"""
Transformer model training using DistilBERT.
Fine-tunes pre-trained DistilBERT for sentiment classification.
"""

import time
from typing import Dict, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

from src.models.base_trainer import BaseTrainer
from src.models.model_registry import ModelRegistry
from src.models.utils import load_transformer_data
from src.utils.helpers import save_json, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis.
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TransformerTrainer(BaseTrainer):
    """
    Trainer for transformer models (DistilBERT).
    """

    def __init__(self, model_name: str, config: Dict, device: Optional[str] = None):
        """
        Initialize transformer trainer.

        Args:
            model_name: Name of the model
            config: Model configuration
            device: Device to use (cuda/cpu)
        """
        super().__init__(model_name, config)

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Model components
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        self.training_time = 0
        self.best_val_loss = float("inf")

        logger.info(f"Initialized {model_name} trainer")

    def _setup_model(self):
        """Setup model and tokenizer."""
        logger.info(f"Loading {self.config['model_name']}...")

        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config["model_name"])

        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config["model_name"], num_labels=self.config["num_labels"]
        )

        # Move to device
        self.model.to(self.device)

        logger.info(f"Model loaded and moved to {self.device}")

    def _create_dataloaders(
        self, train_texts: list, train_labels: list, val_texts: list, val_labels: list
    ):
        """Create data loaders."""
        logger.info("Creating data loaders...")

        # Create datasets
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer, self.config["max_length"]
        )

        val_dataset = SentimentDataset(
            val_texts, val_labels, self.tokenizer, self.config["max_length"]
        )

        # Create loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,  # Windows compatibility
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            f"Data loaders created - "
            f"Train batches: {len(self.train_loader)}, "
            f"Val batches: {len(self.val_loader)}"
        )

    def _setup_optimizer(self, num_training_steps: int):
        """Setup optimizer and scheduler."""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=num_training_steps,
        )

        logger.info("Optimizer and scheduler configured")

    def train(
        self, train_texts: list, train_labels: list, val_texts: list, val_labels: list
    ):
        """
        Train the model.

        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels

        Returns:
            Trained model
        """
        logger.info("=" * 80)
        logger.info("STARTING DISTILBERT TRAINING")
        logger.info("=" * 80)

        start_time = time.time()

        # Setup
        self._setup_model()
        self._create_dataloaders(train_texts, train_labels, val_texts, val_labels)

        num_training_steps = len(self.train_loader) * self.config["epochs"]
        self._setup_optimizer(num_training_steps)

        logger.info(f"Training for {self.config['epochs']} epochs")
        logger.info(f"Total training steps: {num_training_steps}")

        # Training loop
        for epoch in range(self.config["epochs"]):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")

            # Train
            train_loss, train_acc = self._train_epoch()

            # Validate
            val_loss, val_acc = self._validate_epoch()

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                logger.info(f"[NEW BEST] New best validation loss: {val_loss:.4f}")
                # Save checkpoint
                self._save_checkpoint(epoch)

        self.training_time = time.time() - start_time

        logger.info(
            f"\n[SUCCESS] Training completed in {self.training_time:.2f} seconds"
        )

        return self.model

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

            # Accumulate loss
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions.double() / total_predictions

        return avg_loss, accuracy.item()

    def _validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")

            for batch in progress_bar:
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

                # Accumulate loss
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions.double() / total_predictions

        return avg_loss, accuracy.item()

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.model_dir / f"checkpoint_epoch_{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def predict(self, texts: list) -> np.ndarray:
        """Make predictions."""
        self.model.eval()

        predictions = []

        # Create dataset
        dummy_labels = [0] * len(texts)  # Dummy labels
        dataset = SentimentDataset(
            texts, dummy_labels, self.tokenizer, self.config["max_length"]
        )

        loader = DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=False
        )

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, texts: list) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()

        probabilities = []

        # Create dataset
        dummy_labels = [0] * len(texts)
        dataset = SentimentDataset(
            texts, dummy_labels, self.tokenizer, self.config["max_length"]
        )

        loader = DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=False
        )

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def _save_model_impl(self):
        """Save model to disk."""
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

        # Save training history
        history_path = self.model_dir / "training_history.json"
        save_json(self.history, history_path)

        logger.info(f"Model saved to {self.model_dir}")

    def _log_model_to_mlflow(self):
        """Log model to MLflow."""
        mlflow.pytorch.log_model(self.model, "model")

        # Log training history
        for epoch, (train_loss, val_loss) in enumerate(
            zip(self.history["train_loss"], self.history["val_loss"])
        ):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric(
                "train_accuracy", self.history["train_accuracy"][epoch], step=epoch
            )
            mlflow.log_metric(
                "val_accuracy", self.history["val_accuracy"][epoch], step=epoch
            )

        mlflow.log_metric("training_time_seconds", self.training_time)
        mlflow.log_metric("best_val_loss", self.best_val_loss)

    def _get_framework_name(self) -> str:
        """Get framework name."""
        return "transformers"


@timer
def main():
    """Main training pipeline for transformer model."""
    logger.info("=" * 80)
    logger.info("TRANSFORMER (DISTILBERT) TRAINING PIPELINE")
    logger.info("=" * 80)

    # Load configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["model"]["transformer"]
    eval_config = params["evaluation"]

    # Load data
    logger.info("Loading preprocessed data...")
    train_df, val_df, test_df = load_transformer_data()

    # Prepare data
    train_texts = train_df["text_cleaned"].tolist()
    train_labels = train_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()

    val_texts = val_df["text_cleaned"].tolist()
    val_labels = val_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()

    test_texts = test_df["text_cleaned"].tolist()
    test_labels = test_df["sentiment"].map({"negative": 0, "positive": 1}).tolist()

    logger.info(
        f"Data loaded - Train: {len(train_texts)}, "
        f"Val: {len(val_texts)}, Test: {len(test_texts)}"
    )

    # Initialize trainer
    trainer = TransformerTrainer(model_name="distilbert", config=config)

    # Train
    trainer.train(train_texts, train_labels, val_texts, val_labels)

    # Evaluate on all splits
    logger.info("\nEvaluating on all splits...")
    trainer.evaluate(train_texts, np.array(train_labels), split_name="train")
    trainer.evaluate(val_texts, np.array(val_labels), split_name="val")
    trainer.evaluate(test_texts, np.array(test_labels), split_name="test")

    # Generate visualizations
    trainer.generate_confusion_matrix(
        test_texts, np.array(test_labels), split_name="test"
    )
    trainer.generate_roc_curve(test_texts, np.array(test_labels), split_name="test")
    trainer.generate_classification_report(
        test_texts, np.array(test_labels), split_name="test"
    )

    # Save model
    trainer.save_model()

    # Log to MLflow
    run_id = trainer.log_to_mlflow(run_name="distilbert_finetuning")

    # Register model
    registry = ModelRegistry()
    registry.register_model(
        model_name="distilbert",
        run_id=run_id,
        metrics=trainer.metrics,
        model_path=trainer.model_dir,
        metadata={"training_time": trainer.training_time},
    )

    # Check performance gates
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE GATES CHECK")
    logger.info("=" * 80)

    min_metrics = {
        "accuracy": eval_config["min_accuracy"],
        "f1": eval_config["min_f1"],
        "precision": eval_config["min_precision"],
    }

    trainer.check_performance_gates(min_metrics)

    logger.info("\n" + "=" * 80)
    logger.info("[SUCCESS] DISTILBERT TRAINING COMPLETE!")
    logger.info("=" * 80)

    return trainer


if __name__ == "__main__":
    main()
