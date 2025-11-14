"""
Database operations for prediction history.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.dashboard.config import DATABASE_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionDatabase:
    """
    Manages prediction history database.
    """

    def __init__(self, db_path: Path = DATABASE_PATH):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create predictions table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        text TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        probabilities TEXT,
                        feedback INTEGER,
                        used_for_training INTEGER DEFAULT 0,
                        session_id TEXT
                    )
                """
                )

                # Create indexes
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON predictions(timestamp)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_model_name
                    ON predictions(model_name)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sentiment
                    ON predictions(sentiment)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_feedback
                    ON predictions(feedback)
                """
                )

                conn.commit()

            # Run migration for existing databases BEFORE creating new indexes
            self._migrate_database()

            # Create indexes for new columns after migration
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if used_for_training column exists before creating index
                cursor.execute("PRAGMA table_info(predictions)")
                columns = [col[1] for col in cursor.fetchall()]

                if "used_for_training" in columns:
                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_used_for_training
                        ON predictions(used_for_training)
                    """
                    )
                    conn.commit()

            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def _migrate_database(self):
        """Add new columns to existing database if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if table exists first
                cursor.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='predictions'
                """
                )
                table_exists = cursor.fetchone() is not None

                if not table_exists:
                    # Table doesn't exist yet, will be created by CREATE TABLE
                    logger.info(
                        "Table predictions does not exist, will be created with new schema"
                    )
                    return

                # Check if used_for_training column exists
                cursor.execute("PRAGMA table_info(predictions)")
                columns = [col[1] for col in cursor.fetchall()]

                if "used_for_training" not in columns:
                    logger.info(
                        "Adding used_for_training column to predictions table..."
                    )
                    try:
                        cursor.execute(
                            """
                            ALTER TABLE predictions
                            ADD COLUMN used_for_training INTEGER DEFAULT 0
                        """
                        )
                        conn.commit()
                        logger.info("✅ Column used_for_training added successfully")
                    except sqlite3.OperationalError as e:
                        # Column might have been added by another process
                        if "duplicate column" in str(e).lower():
                            logger.info(
                                "Column used_for_training already exists (added by another process)"
                            )
                        else:
                            raise
                else:
                    logger.debug("Column used_for_training already exists")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            # Don't raise - let the system continue, but log the error
            import traceback

            logger.error(traceback.format_exc())

    def add_prediction(
        self,
        text: str,
        model_name: str,
        sentiment: str,
        confidence: float,
        probabilities: Dict[str, float],
        session_id: str,
    ) -> int:
        """
        Add prediction to database.

        Args:
            text: Input text
            model_name: Model used
            sentiment: Predicted sentiment
            confidence: Confidence score
            probabilities: Probability distribution
            session_id: Session identifier

        Returns:
            Prediction ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO predictions
                    (text, model_name, sentiment, confidence, probabilities, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        text,
                        model_name,
                        sentiment,
                        confidence,
                        json.dumps(probabilities),
                        session_id,
                    ),
                )

                conn.commit()
                prediction_id = cursor.lastrowid

            logger.info(f"Prediction {prediction_id} added to database")
            return prediction_id

        except Exception as e:
            logger.error(f"Failed to add prediction: {str(e)}")
            raise

    def update_feedback(self, prediction_id: int, feedback: int):
        """
        Update prediction feedback.

        Args:
            prediction_id: Prediction ID
            feedback: Feedback value (1: positive, -1: negative)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE predictions
                    SET feedback = ?
                    WHERE id = ?
                """,
                    (feedback, prediction_id),
                )

                conn.commit()

            logger.info(f"Feedback updated for prediction {prediction_id}")

        except Exception as e:
            logger.error(f"Failed to update feedback: {str(e)}")
            raise

    def get_recent_predictions(
        self, limit: int = 10, session_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get recent predictions.

        Args:
            limit: Number of predictions to retrieve
            session_id: Filter by session ID

        Returns:
            DataFrame with predictions
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if session_id:
                    query = """
                        SELECT id, timestamp, text, model_name, sentiment, confidence, feedback
                        FROM predictions
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    df = pd.read_sql_query(query, conn, params=(session_id, limit))
                else:
                    query = """
                        SELECT id, timestamp, text, model_name, sentiment, confidence, feedback
                        FROM predictions
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    df = pd.read_sql_query(query, conn, params=(limit,))

                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                return df

        except Exception as e:
            logger.error(f"Failed to get recent predictions: {str(e)}")
            return pd.DataFrame()

    def get_statistics(self) -> Dict:
        """
        Get prediction statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total predictions
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total = cursor.fetchone()[0]

                # Predictions by sentiment
                cursor.execute(
                    """
                    SELECT sentiment, COUNT(*) as count
                    FROM predictions
                    GROUP BY sentiment
                """
                )
                by_sentiment = dict(cursor.fetchall())

                # Predictions by model
                cursor.execute(
                    """
                    SELECT model_name, COUNT(*) as count
                    FROM predictions
                    GROUP BY model_name
                """
                )
                by_model = dict(cursor.fetchall())

                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM predictions")
                avg_confidence = cursor.fetchone()[0] or 0.0

                # Predictions today
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM predictions
                    WHERE DATE(timestamp) = DATE('now')
                """
                )
                today = cursor.fetchone()[0]

                return {
                    "total": total,
                    "by_sentiment": by_sentiment,
                    "by_model": by_model,
                    "avg_confidence": avg_confidence,
                    "today": today,
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {
                "total": 0,
                "by_sentiment": {},
                "by_model": {},
                "avg_confidence": 0.0,
                "today": 0,
            }

    def export_all(self) -> pd.DataFrame:
        """
        Export all predictions.

        Returns:
            DataFrame with all predictions
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, text, model_name, sentiment, confidence, feedback
                    FROM predictions
                    ORDER BY timestamp DESC
                """
                df = pd.read_sql_query(query, conn)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df

        except Exception as e:
            logger.error(f"Failed to export predictions: {str(e)}")
            return pd.DataFrame()

    def get_feedback_for_training(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get predictions with feedback that haven't been used for training.

        Args:
            limit: Maximum number of feedback samples to retrieve

        Returns:
            DataFrame with columns: id, text, feedback, timestamp
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if limit:
                    query = """
                        SELECT id, text, feedback, timestamp
                        FROM predictions
                        WHERE feedback IS NOT NULL AND used_for_training = 0
                        ORDER BY timestamp ASC
                        LIMIT ?
                    """
                    df = pd.read_sql_query(query, conn, params=(limit,))
                else:
                    query = """
                        SELECT id, text, feedback, timestamp
                        FROM predictions
                        WHERE feedback IS NOT NULL AND used_for_training = 0
                        ORDER BY timestamp ASC
                    """
                    df = pd.read_sql_query(query, conn)

                # Convert timestamp to datetime
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                logger.info(f"Retrieved {len(df)} feedback samples for training")
                return df

        except Exception as e:
            logger.error(f"Failed to get feedback for training: {str(e)}")
            return pd.DataFrame()

    def mark_feedback_as_used(self, prediction_ids: List[int]):
        """
        Mark feedback as used for training.

        Args:
            prediction_ids: List of prediction IDs to mark
        """
        try:
            if not prediction_ids:
                logger.warning("No prediction IDs provided to mark as used")
                return

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create placeholders for IN clause
                placeholders = ",".join("?" * len(prediction_ids))

                query = f"""
                    UPDATE predictions
                    SET used_for_training = 1
                    WHERE id IN ({placeholders})
                """

                cursor.execute(query, prediction_ids)
                updated_count = cursor.rowcount

                conn.commit()

            logger.info(
                f"✅ Marked {updated_count} feedback samples as used for training"
            )

        except Exception as e:
            logger.error(f"Failed to mark feedback as used: {str(e)}")
            raise

    def clear_old_predictions(self, days: int = 30):
        """
        Clear predictions older than specified days.

        Args:
            days: Number of days to keep
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM predictions
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """,
                    (days,),
                )

                deleted = cursor.rowcount
                conn.commit()

            logger.info(f"Deleted {deleted} old predictions")

        except Exception as e:
            logger.error(f"Failed to clear old predictions: {str(e)}")
            raise


# Global database instance
_db_instance = None


def get_database() -> PredictionDatabase:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = PredictionDatabase()
    return _db_instance
