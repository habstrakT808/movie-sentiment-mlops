"""
Script to manually trigger data collection for continuous learning.
Can be run independently without waiting for scheduled time.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio  # noqa: E402

from src.data_collection.periodic_collector import PeriodicDataCollector  # noqa: E402
from src.preprocessing.incremental_preprocessor import (  # noqa: E402
    IncrementalPreprocessor,
)
from src.training.continuous_learning import ContinuousLearner  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


async def trigger_data_collection_manual():
    """Manually trigger data collection and retraining if threshold met."""
    print("\n" + "=" * 80)
    print("MANUAL DATA COLLECTION TRIGGER")
    print("=" * 80)

    try:
        # 1. Initialize collectors
        print("\n📥 Step 1: Initializing collectors...")
        collector = PeriodicDataCollector()
        preprocessor = IncrementalPreprocessor()
        learner = ContinuousLearner()

        # 2. Collect new data
        print("\n📥 Step 2: Collecting new data from Reddit and Kaggle...")
        collection_result = collector.collect_incremental_data(
            min_samples_per_source=100,
            time_filter="week",
            max_reddit_samples=1000,
            max_kaggle_samples=500,
        )

        if collection_result["status"] != "success":
            print(f"❌ Data collection failed: {collection_result.get('message')}")
            return

        new_samples = collection_result["total_new_samples"]
        print(f"✅ Collected {new_samples} new samples")

        # 3. Check threshold
        THRESHOLD = 3000  # Same as default
        if new_samples < THRESHOLD:
            print(
                f"\n⏳ Insufficient new data for retraining "
                f"({new_samples} samples, need {THRESHOLD})"
            )
            print("✅ Data collection completed. Retraining skipped.")
            return

        # 4. Preprocess and merge
        print("\n🔄 Step 3: Preprocessing and merging new data...")
        preprocess_result = preprocessor.preprocess_incremental_data(
            incremental_data_path=Path(collection_result["output_path"]),
            merge_with_existing=True,
        )

        if preprocess_result["status"] != "success":
            print(f"❌ Preprocessing failed: {preprocess_result.get('message')}")
            return

        print(
            f"✅ Preprocessing complete. "
            f"Total samples after merge: {preprocess_result['total_samples_after_merge']}"
        )

        # 5. Trigger retraining
        print("\n🚀 Step 4: Starting model retraining...")
        results = learner.trigger_data_collection_retraining(
            new_data_path=Path(preprocess_result["output_paths"]["train"])
        )

        # 6. Check results
        if results.get("status") == "success":
            if results.get("should_deploy"):
                print("\n" + "=" * 80)
                print("✅ SUCCESS: NEW MODEL DEPLOYED!")
                print("=" * 80)
                print(f"F1 Score: {results['new_model_metrics']['test_f1']:.4f}")
                print(f"Improvement: {results['improvement']['f1']:.2%}")
                print(f"Total training samples: {results['total_training_samples']}")
                print("\n⚠️ Please restart API to load new model:")
                print("   docker-compose restart sentiment-api")
            else:
                print("\n" + "=" * 80)
                print("⚠️ Retraining completed but model not deployed")
                print("=" * 80)
                print(f"New F1: {results['new_model_metrics']['test_f1']:.4f}")
                print(f"Production F1: {results['production_metrics']['test_f1']:.4f}")
                print(f"Improvement: {results['improvement']['f1']:.2%}")
                print("\n(Improvement below threshold, model not deployed)")
        else:
            print(f"\n❌ Retraining failed: {results.get('message')}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Starting manual data collection...")
    asyncio.run(trigger_data_collection_manual())
    print("\n✅ Process completed!")
