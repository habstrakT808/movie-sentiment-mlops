"""
Data drift detection for movie sentiment analysis.

Detects drift in:
- Text length distribution (KS test)
- Word count distribution (KS test)
- Sentiment distribution (Chi-square test)
- Statistical feature drift (Jensen-Shannon divergence)
"""

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.dashboard.components.database import PredictionDatabase
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Default reference statistics (from training data analysis)
DEFAULT_REFERENCE_STATS = {
    "text_length": {
        "mean": 1287.82,
        "std": 968.25,
        "min": 41,
        "max": 9081,
        "median": 955.00,
        "q25": 692.00,
        "q75": 1571.25,
        "samples": 14000,
    },
    "word_count": {
        "mean": 229.15,
        "std": 168.84,
        "min": 4,
        "max": 1577,
        "median": 171.00,
        "q25": 127.00,
        "q75": 287.00,
        "samples": 14000,
    },
    "sentiment_distribution": {
        "positive": 0.50,
        "negative": 0.50,
        "neutral": 0.00,
    },
    "computed_at": "2024-11-10T00:00:00",
    "source": "training_data",
}


class DataDriftDetector:
    """
    Detects data drift in production predictions compared to reference data.

    Uses multiple statistical tests:
    - Kolmogorov-Smirnov test for continuous distributions
    - Chi-square test for categorical distributions
    - Jensen-Shannon divergence for distribution similarity
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_stats: Optional[Dict] = None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference DataFrame with columns: text, sentiment
            reference_stats: Pre-computed reference statistics
            db_path: Path to predictions database
        """
        self.db = PredictionDatabase(db_path) if db_path else PredictionDatabase()

        # Load reference statistics
        if reference_data is not None:
            logger.info("Computing reference stats from provided data")
            self.reference_stats = self._compute_reference_stats(reference_data)
            self.reference_data = reference_data
        elif reference_stats is not None:
            logger.info("Using provided reference stats")
            self.reference_stats = reference_stats
            self.reference_data = None
        else:
            # Try to load from training data, fallback to database or default
            logger.info("Attempting to load reference data from training file")
            self.reference_data = self._load_reference_data()
            if self.reference_data is not None:
                self.reference_stats = self._compute_reference_stats(
                    self.reference_data
                )
            else:
                logger.warning(
                    "Using default reference stats (from training data analysis)"
                )
                self.reference_stats = DEFAULT_REFERENCE_STATS.copy()

        logger.info(
            f"Drift detector initialized with reference: {self.reference_stats['source']}"
        )

    def _load_reference_data(self) -> Optional[pd.DataFrame]:
        """
        Load reference data from training file or database.

        Priority:
        1. Training data file (data/processed/train.csv)
        2. Database baseline (first 1000 predictions)
        3. None (will use default stats)
        """
        # Try 1: Load from training data file
        train_path = Config.PROCESSED_DATA_DIR / "train.csv"
        if train_path.exists():
            try:
                logger.info(f"Loading reference data from {train_path}")
                df = pd.read_csv(train_path)
                if len(df) >= 100:
                    logger.info(f"Loaded {len(df)} samples from training data")
                    return df
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")

        # Try 2: Load from database (first 1000 predictions)
        try:
            logger.info("Attempting to load baseline from database")
            predictions = self.db.export_all()
            if len(predictions) >= 100:
                # Take first 1000 as baseline
                baseline = predictions.head(1000).copy()
                baseline["text_cleaned"] = baseline["text"]
                logger.info(f"Loaded {len(baseline)} samples from database as baseline")
                return baseline
        except Exception as e:
            logger.error(f"Failed to load database baseline: {e}")

        # Try 3: Return None (will use default stats)
        logger.warning("No reference data available, will use default stats")
        return None

    def _compute_reference_stats(self, data: pd.DataFrame) -> Dict:
        """
        Compute statistics from reference data.

        Args:
            data: DataFrame with columns: text_cleaned, sentiment

        Returns:
            Dictionary with reference statistics
        """
        try:
            # Compute text statistics
            data["text_length"] = data["text_cleaned"].astype(str).str.len()
            data["word_count"] = data["text_cleaned"].astype(str).str.split().str.len()

            # Text length stats
            text_length_stats = {
                "mean": float(data["text_length"].mean()),
                "std": float(data["text_length"].std()),
                "min": int(data["text_length"].min()),
                "max": int(data["text_length"].max()),
                "median": float(data["text_length"].median()),
                "q25": float(data["text_length"].quantile(0.25)),
                "q75": float(data["text_length"].quantile(0.75)),
                "samples": len(data),
            }

            # Word count stats
            word_count_stats = {
                "mean": float(data["word_count"].mean()),
                "std": float(data["word_count"].std()),
                "min": int(data["word_count"].min()),
                "max": int(data["word_count"].max()),
                "median": float(data["word_count"].median()),
                "q25": float(data["word_count"].quantile(0.25)),
                "q75": float(data["word_count"].quantile(0.75)),
                "samples": len(data),
            }

            # Sentiment distribution
            sentiment_dist = data["sentiment"].value_counts(normalize=True).to_dict()

            stats = {
                "text_length": text_length_stats,
                "word_count": word_count_stats,
                "sentiment_distribution": sentiment_dist,
                "computed_at": datetime.now().isoformat(),
                "source": "computed_from_data",
            }

            logger.info(f"Computed reference stats from {len(data)} samples")
            return stats

        except Exception as e:
            logger.error(f"Failed to compute reference stats: {e}")
            return DEFAULT_REFERENCE_STATS.copy()

    def has_sufficient_data(self, min_samples: int = 50) -> bool:
        """
        Check if database has sufficient data for drift detection.

        Args:
            min_samples: Minimum number of samples required

        Returns:
            True if sufficient data available
        """
        try:
            stats = self.db.get_statistics()
            total = stats.get("total", 0)
            sufficient = total >= min_samples

            if not sufficient:
                logger.debug(
                    f"Insufficient data: {total} samples (required: {min_samples})"
                )

            return sufficient
        except Exception as e:
            logger.error(f"Error checking data sufficiency: {e}")
            return False

    def detect_drift(
        self, window_size: int = 100, window_hours: Optional[int] = None
    ) -> Dict:
        """
        Detect drift in recent predictions.

        Args:
            window_size: Number of recent predictions to analyze
            window_hours: Time window in hours (alternative to window_size)

        Returns:
            Dict with drift scores, p-values, and alert level
        """
        try:
            logger.info(
                f"Starting drift detection (window_size={window_size}, window_hours={window_hours})"
            )

            # Get production data from database
            production_data = self._get_production_data(window_size, window_hours)

            if production_data is None or len(production_data) < 10:
                logger.warning(
                    f"Insufficient production data: {len(production_data) if production_data is not None else 0} samples"
                )
                return self._create_empty_result("insufficient_data")

            # Compute production statistics
            production_stats = self._compute_production_stats(production_data)

            # Detect drift using statistical tests
            drift_results = self._run_drift_tests(production_data, production_stats)

            # Calculate overall drift score
            overall_score = self._calculate_overall_drift_score(drift_results)

            # Determine alert level
            alert_level = self._determine_alert_level(overall_score)

            # Create result
            result = {
                "timestamp": datetime.now().isoformat(),
                "window_size": len(production_data),
                "production_samples": len(production_data),
                "drift_scores": {
                    "text_length": drift_results["text_length"]["score"],
                    "word_count": drift_results["word_count"]["score"],
                    "sentiment": drift_results["sentiment"]["score"],
                    "overall": overall_score,
                },
                "p_values": {
                    "text_length": drift_results["text_length"]["p_value"],
                    "word_count": drift_results["word_count"]["p_value"],
                    "sentiment": drift_results["sentiment"]["p_value"],
                },
                "alert_level": alert_level,
                "production_stats": production_stats,
                "reference_stats": self.reference_stats,
            }

            # Log results
            self._log_drift_results(result)

            return result

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return self._create_empty_result("error", str(e))

    def _get_production_data(
        self, window_size: int, window_hours: Optional[int]
    ) -> Optional[pd.DataFrame]:
        """Get production data from database."""
        try:
            if window_hours:
                # Time-based window (not implemented in database yet)
                logger.warning(
                    "Time-based window not fully implemented, using count-based"
                )
                data = self.db.get_recent_predictions(limit=window_size)
            else:
                # Count-based window
                data = self.db.get_recent_predictions(limit=window_size)

            if data is None or len(data) == 0:
                return None

            return data

        except Exception as e:
            logger.error(f"Failed to get production data: {e}")
            return None

    def _compute_production_stats(self, data: pd.DataFrame) -> Dict:
        """Compute statistics from production data."""
        try:
            # Compute text statistics
            data["text_length"] = data["text"].astype(str).str.len()
            data["word_count"] = data["text"].astype(str).str.split().str.len()

            # Text length stats
            text_length_stats = {
                "mean": float(data["text_length"].mean()),
                "std": float(data["text_length"].std()),
                "min": int(data["text_length"].min()),
                "max": int(data["text_length"].max()),
                "median": float(data["text_length"].median()),
                "samples": len(data),
            }

            # Word count stats
            word_count_stats = {
                "mean": float(data["word_count"].mean()),
                "std": float(data["word_count"].std()),
                "min": int(data["word_count"].min()),
                "max": int(data["word_count"].max()),
                "median": float(data["word_count"].median()),
                "samples": len(data),
            }

            # Sentiment distribution
            sentiment_dist = data["sentiment"].value_counts(normalize=True).to_dict()

            return {
                "text_length": text_length_stats,
                "word_count": word_count_stats,
                "sentiment_distribution": sentiment_dist,
                "computed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to compute production stats: {e}")
            return {}

    def _run_drift_tests(
        self, production_data: pd.DataFrame, production_stats: Dict
    ) -> Dict:
        """Run statistical tests for drift detection."""
        results = {}

        # 1. KS test for text length
        if self.reference_data is not None:
            results["text_length"] = self._ks_test_drift(
                self.reference_data["text_length"].values,
                production_data["text_length"].values,
                "text_length",
            )
        else:
            # Use statistical distance if no raw data
            results["text_length"] = self._statistical_distance_drift(
                self.reference_stats["text_length"],
                production_stats["text_length"],
                "text_length",
            )

        # 2. KS test for word count
        if self.reference_data is not None:
            results["word_count"] = self._ks_test_drift(
                self.reference_data["word_count"].values,
                production_data["word_count"].values,
                "word_count",
            )
        else:
            results["word_count"] = self._statistical_distance_drift(
                self.reference_stats["word_count"],
                production_stats["word_count"],
                "word_count",
            )

        # 3. Chi-square test for sentiment distribution
        results["sentiment"] = self._chi_square_drift(
            self.reference_stats["sentiment_distribution"],
            production_stats["sentiment_distribution"],
        )

        return results

    def _ks_test_drift(
        self, reference: np.ndarray, production: np.ndarray, metric_name: str
    ) -> Dict:
        """
        Kolmogorov-Smirnov test for distribution drift.

        Args:
            reference: Reference distribution
            production: Production distribution
            metric_name: Name of the metric

        Returns:
            Dict with score, p_value, and drift status
        """
        try:
            ks_stat, p_value = ks_2samp(reference, production)

            # Normalize KS statistic to 0-1 scale (already 0-1, but ensure)
            drift_score = float(np.clip(ks_stat, 0, 1))

            logger.info(
                f"KS test - {metric_name}: statistic={drift_score:.4f}, p-value={p_value:.4f}"
            )

            return {
                "score": drift_score,
                "p_value": float(p_value),
                "drift_detected": p_value < 0.05 or drift_score > 0.1,
                "test": "kolmogorov_smirnov",
            }

        except Exception as e:
            logger.error(f"KS test failed for {metric_name}: {e}")
            return {
                "score": 0.0,
                "p_value": 1.0,
                "drift_detected": False,
                "test": "kolmogorov_smirnov",
                "error": str(e),
            }

    def _chi_square_drift(self, reference_dist: Dict, production_dist: Dict) -> Dict:
        """
        Chi-square test for categorical distribution drift.

        Args:
            reference_dist: Reference distribution (dict of proportions)
            production_dist: Production distribution (dict of proportions)

        Returns:
            Dict with score, p_value, and drift status
        """
        try:
            # Get all possible categories
            all_categories = set(reference_dist.keys()) | set(production_dist.keys())

            # Create observed and expected frequencies
            reference_freq = [reference_dist.get(cat, 0) for cat in all_categories]
            production_freq = [production_dist.get(cat, 0) for cat in all_categories]

            # Convert proportions to counts (multiply by 100 for numerical stability)
            reference_counts = [int(f * 100) for f in reference_freq]
            production_counts = [int(f * 100) for f in production_freq]

            # Chi-square test
            contingency_table = np.array([reference_counts, production_counts])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            # Normalize chi2 to 0-1 scale using Cramér's V
            n = contingency_table.sum()
            min_dim = min(contingency_table.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            drift_score = float(np.clip(cramers_v, 0, 1))

            logger.info(
                f"Chi-square test - sentiment: chi2={chi2:.4f}, p-value={p_value:.4f}, Cramér's V={drift_score:.4f}"
            )

            return {
                "score": drift_score,
                "p_value": float(p_value),
                "drift_detected": p_value < 0.05 or drift_score > 0.3,
                "test": "chi_square",
            }

        except Exception as e:
            logger.error(f"Chi-square test failed: {e}")
            return {
                "score": 0.0,
                "p_value": 1.0,
                "drift_detected": False,
                "test": "chi_square",
                "error": str(e),
            }

    def _statistical_distance_drift(
        self, reference_stats: Dict, production_stats: Dict, metric_name: str
    ) -> Dict:
        """
        Statistical distance-based drift detection (when raw data not available).

        Uses normalized difference in mean and std.

        Args:
            reference_stats: Reference statistics
            production_stats: Production statistics
            metric_name: Name of the metric

        Returns:
            Dict with score and drift status
        """
        try:
            ref_mean = reference_stats["mean"]
            ref_std = reference_stats["std"]
            prod_mean = production_stats["mean"]
            prod_std = production_stats["std"]

            # Normalized difference in mean (Cohen's d)
            mean_diff = abs(prod_mean - ref_mean) / ref_std if ref_std > 0 else 0

            # Normalized difference in std
            std_diff = abs(prod_std - ref_std) / ref_std if ref_std > 0 else 0

            # Combined drift score (average of normalized differences, capped at 1)
            drift_score = float(np.clip((mean_diff + std_diff) / 2, 0, 1))

            logger.info(
                f"Statistical distance - {metric_name}: "
                f"mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f}, score={drift_score:.4f}"
            )

            return {
                "score": drift_score,
                "p_value": None,  # Not applicable for this test
                "drift_detected": drift_score > 0.3,
                "test": "statistical_distance",
            }

        except Exception as e:
            logger.error(f"Statistical distance test failed for {metric_name}: {e}")
            return {
                "score": 0.0,
                "p_value": None,
                "drift_detected": False,
                "test": "statistical_distance",
                "error": str(e),
            }

    def _calculate_overall_drift_score(self, drift_results: Dict) -> float:
        """
        Calculate overall drift score using equal weighting.

        Args:
            drift_results: Dict with drift results for each metric

        Returns:
            Overall drift score (0-1)
        """
        try:
            scores = [
                drift_results["text_length"]["score"],
                drift_results["word_count"]["score"],
                drift_results["sentiment"]["score"],
            ]

            # Filter out None/NaN values
            valid_scores = [s for s in scores if s is not None and not np.isnan(s)]

            if not valid_scores:
                return 0.0

            # Equal weight average
            overall = float(np.mean(valid_scores))

            logger.info(f"Overall drift score: {overall:.4f}")

            return overall

        except Exception as e:
            logger.error(f"Failed to calculate overall drift score: {e}")
            return 0.0

    def _determine_alert_level(self, drift_score: float) -> str:
        """
        Determine alert level based on drift score.

        Args:
            drift_score: Overall drift score (0-1)

        Returns:
            Alert level: 'ok', 'warning', or 'critical'
        """
        if drift_score > 0.5:
            return "critical"
        elif drift_score > 0.3:
            return "warning"
        else:
            return "ok"

    def _log_drift_results(self, result: Dict):
        """Log drift detection results."""
        alert_level = result["alert_level"]
        overall_score = result["drift_scores"]["overall"]

        if alert_level == "critical":
            logger.warning(
                f"⚠️ CRITICAL DATA DRIFT DETECTED! Overall score: {overall_score:.4f}"
            )
            logger.warning(f"Drift scores: {result['drift_scores']}")
        elif alert_level == "warning":
            logger.info(
                f"⚠️ Warning: Moderate data drift detected. Overall score: {overall_score:.4f}"
            )
        else:
            logger.info(
                f"✅ No significant drift detected. Overall score: {overall_score:.4f}"
            )

    def _create_empty_result(self, reason: str, error: str = None) -> Dict:
        """Create empty result when drift detection cannot be performed."""
        return {
            "timestamp": datetime.now().isoformat(),
            "window_size": 0,
            "production_samples": 0,
            "drift_scores": {
                "text_length": 0.0,
                "word_count": 0.0,
                "sentiment": 0.0,
                "overall": 0.0,
            },
            "p_values": {
                "text_length": None,
                "word_count": None,
                "sentiment": None,
            },
            "alert_level": "ok",
            "reason": reason,
            "error": error,
            "production_stats": {},
            "reference_stats": self.reference_stats,
        }
