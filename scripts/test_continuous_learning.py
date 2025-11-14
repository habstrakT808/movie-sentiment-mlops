#!/usr/bin/env python3
"""
Test Continuous Learning Pipeline.

This script tests the continuous learning retraining with low threshold.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.continuous_learning import ContinuousLearner  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main():
    """Test continuous learning retraining."""
    print("=" * 80)
    print("🧪 TESTING CONTINUOUS LEARNING PIPELINE")
    print("=" * 80)
    print()

    # Initialize with low threshold for testing
    threshold = 10  # Low threshold for testing
    min_improvement = 0.01  # 1% improvement required

    print("📊 Configuration:")
    print(f"   Retrain threshold: {threshold} feedback samples")
    print(f"   Min improvement: {min_improvement:.1%}")
    print()

    # Initialize learner
    learner = ContinuousLearner(
        retrain_threshold=threshold,
        min_improvement=min_improvement,
        model_name="distilbert",
    )

    # Check feedback
    print("🔍 Checking feedback...")
    feedback_df = learner.collect_feedback_from_db()
    feedback_count = len(feedback_df)

    print(f"   Total feedback available: {feedback_count}")
    print(f"   Threshold: {threshold}")
    print()

    if feedback_count < threshold:
        print(f"❌ Insufficient feedback: {feedback_count} < {threshold}")
        print("   Please generate more feedback using:")
        print(
            f"   python scripts/generate_sample_feedback.py --random --count {threshold}"
        )
        return

    print("✅ Sufficient feedback available!")
    print()
    print("🚀 Starting retraining...")
    print("   (This may take 60-90 minutes on CPU, or 20-30 minutes on GPU)")
    print()

    # Trigger retraining
    try:
        result = learner.trigger_retraining()

        print()
        print("=" * 80)
        print("📊 RETRAINING RESULTS")
        print("=" * 80)
        print()

        if result.get("status") == "success":
            print("✅ Retraining completed successfully!")
            print()
            print("📈 Model Metrics:")
            print(
                f"   New Model F1:  {result['new_model_metrics'].get('test_f1', 0):.4f}"
            )
            print(
                f"   Production F1: {result['production_metrics'].get('test_f1', 0):.4f}"
            )
            print()
            print("📊 Improvement:")
            for metric, imp in result["improvement"].items():
                print(f"   {metric}: {imp:+.2%}")
            print()
            print(
                f"🚀 Deployment Decision: {'DEPLOYED' if result['should_deploy'] else 'NOT DEPLOYED'}"
            )
            print()
            print(
                f"⏱️  Training Time: {result['training_time']:.2f} seconds ({result['training_time']/60:.1f} minutes)"
            )
            print()
            print(f"📦 Model Version: {result['model_version']}")
            print(f"📁 Model Path: {result['model_path']}")
            print()
            if result["should_deploy"]:
                print("⚠️  Note: Please restart API to load new model:")
                print("   cd docker && docker-compose restart sentiment-api")
        else:
            print(f"❌ Retraining failed: {result.get('message', 'Unknown error')}")
            if "traceback" in result:
                print()
                print("Traceback:")
                print(result["traceback"])

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
