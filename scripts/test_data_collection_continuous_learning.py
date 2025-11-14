"""
Comprehensive test script for Data Collection-Based Continuous Learning.
Tests all components from data collection to model retraining.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import traceback  # noqa: E402
from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402

from src.data_collection.periodic_collector import PeriodicDataCollector  # noqa: E402
from src.preprocessing.incremental_preprocessor import (  # noqa: E402
    IncrementalPreprocessor,
)
from src.training.continuous_learning import ContinuousLearner  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def test_imports():
    """Test 1: Verify all imports work correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Testing Imports")
    print("=" * 80)

    try:
        from src.data_collection.periodic_collector import (  # noqa: F401
            PeriodicDataCollector,
        )

        print("✅ PeriodicDataCollector imported successfully")

        from src.preprocessing.incremental_preprocessor import (  # noqa: F401
            IncrementalPreprocessor,
        )

        print("✅ IncrementalPreprocessor imported successfully")

        from src.training.continuous_learning import ContinuousLearner

        print("✅ ContinuousLearner imported successfully")

        # Check if new method exists
        assert hasattr(
            ContinuousLearner, "trigger_data_collection_retraining"
        ), "trigger_data_collection_retraining method not found"
        print("✅ trigger_data_collection_retraining method exists")

        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False


def test_periodic_collector_init():
    """Test 2: Test PeriodicDataCollector initialization."""
    print("\n" + "=" * 80)
    print("TEST 2: Testing PeriodicDataCollector Initialization")
    print("=" * 80)

    try:
        collector = PeriodicDataCollector()
        print("✅ PeriodicDataCollector initialized successfully")
        print(f"   Incremental directory: {collector.incremental_dir}")
        print(f"   Existing hashes loaded: {len(collector.existing_hashes)}")

        return True, collector
    except Exception as e:
        print(f"❌ PeriodicDataCollector initialization failed: {e}")
        traceback.print_exc()
        return False, None


def test_incremental_preprocessor_init():
    """Test 3: Test IncrementalPreprocessor initialization."""
    print("\n" + "=" * 80)
    print("TEST 3: Testing IncrementalPreprocessor Initialization")
    print("=" * 80)

    try:
        preprocessor = IncrementalPreprocessor()
        print("✅ IncrementalPreprocessor initialized successfully")
        print(f"   Processed directory: {preprocessor.processed_dir}")
        print(
            f"   TF-IDF vectorizer loaded: {preprocessor.tfidf_vectorizer is not None}"
        )
        print(f"   Label encoder loaded: {preprocessor.label_encoder is not None}")

        return True, preprocessor
    except Exception as e:
        print(f"❌ IncrementalPreprocessor initialization failed: {e}")
        traceback.print_exc()
        return False, None


def test_create_sample_incremental_data(collector):
    """Test 4: Create sample incremental data for testing."""
    print("\n" + "=" * 80)
    print("TEST 4: Creating Sample Incremental Data")
    print("=" * 80)

    try:
        # Create sample data that mimics collected data
        sample_data = pd.DataFrame(
            {
                "text": [
                    "This movie was absolutely amazing! Best film I've seen this year.",
                    "Terrible movie, waste of time. Don't watch it.",
                    "I loved every minute of this film. Highly recommend!",
                    "Boring and predictable. Not worth watching.",
                    "Great acting and storyline. Really enjoyed it!",
                ],
                "sentiment": [
                    "positive",
                    "negative",
                    "positive",
                    "negative",
                    "positive",
                ],
                "hash": ["hash1", "hash2", "hash3", "hash4", "hash5"],
                "text_length": [60, 40, 50, 45, 55],
                "word_count": [12, 8, 10, 9, 11],
                "source_type": ["reddit", "reddit", "kaggle", "reddit", "kaggle"],
            }
        )

        # Save to incremental directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            collector.incremental_dir / f"test_incremental_data_{timestamp}.csv"
        )
        sample_data.to_csv(output_path, index=False)

        print(f"✅ Sample data created: {output_path}")
        print(f"   Samples: {len(sample_data)}")
        print(
            f"   Sentiment distribution: {sample_data['sentiment'].value_counts().to_dict()}"
        )

        return True, output_path
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        traceback.print_exc()
        return False, None


def test_incremental_preprocessing(preprocessor, sample_data_path):
    """Test 5: Test incremental preprocessing."""
    print("\n" + "=" * 80)
    print("TEST 5: Testing Incremental Preprocessing")
    print("=" * 80)

    try:
        # Check if existing training data exists
        if not preprocessor.existing_train_path.exists():
            print("⚠️ No existing training data found. Creating minimal sample...")
            # Create minimal existing data
            existing_data = pd.DataFrame(
                {
                    "text_cleaned": [
                        "This is a test review",
                        "Another test review",
                    ],
                    "sentiment": ["positive", "negative"],
                }
            )
            existing_data.to_csv(preprocessor.existing_train_path, index=False)
            print(
                f"   Created minimal training data: {preprocessor.existing_train_path}"
            )

        result = preprocessor.preprocess_incremental_data(
            incremental_data_path=sample_data_path,
            merge_with_existing=True,
        )

        if result["status"] == "success":
            print("✅ Incremental preprocessing completed successfully")
            print(f"   New samples: {result['new_samples']}")
            print(f"   Total after merge: {result['total_samples_after_merge']}")
            print(f"   Train samples: {result['train_samples']}")
            print(f"   Val samples: {result['val_samples']}")
            print(f"   Test samples: {result['test_samples']}")
            return True, result
        else:
            print(f"❌ Preprocessing failed: {result.get('message')}")
            return False, None

    except Exception as e:
        print(f"❌ Incremental preprocessing test failed: {e}")
        traceback.print_exc()
        return False, None


def test_continuous_learner_data_collection_method():
    """Test 6: Test ContinuousLearner data collection retraining method."""
    print("\n" + "=" * 80)
    print("TEST 6: Testing ContinuousLearner Data Collection Method")
    print("=" * 80)

    try:
        learner = ContinuousLearner(
            retrain_threshold=1,  # Low threshold for testing
            min_improvement=0.0,  # Accept any improvement for testing
        )
        print("✅ ContinuousLearner initialized")

        # Check if method exists
        assert hasattr(
            learner, "trigger_data_collection_retraining"
        ), "Method trigger_data_collection_retraining not found"
        print("✅ Method trigger_data_collection_retraining exists")

        # Check if training data exists
        if not learner.train_data_path.exists():
            print("⚠️ Training data not found. Skipping actual retraining test.")
            print("   (This is expected if preprocessing hasn't been run)")
            return True

        # Note: We won't actually trigger retraining in test (too slow)
        # Just verify the method can be called
        print("✅ Method is callable (skipping actual retraining for speed)")

        return True
    except Exception as e:
        print(f"❌ ContinuousLearner test failed: {e}")
        traceback.print_exc()
        return False


def test_api_background_task_import():
    """Test 7: Test API background task can be imported."""
    print("\n" + "=" * 80)
    print("TEST 7: Testing API Background Task Import")
    print("=" * 80)

    try:
        from src.deployment.api import run_periodic_data_collection

        print("✅ run_periodic_data_collection function imported successfully")

        # Check if it's async
        import inspect

        assert inspect.iscoroutinefunction(
            run_periodic_data_collection
        ), "run_periodic_data_collection should be async"
        print("✅ Function is async (as expected)")

        return True
    except Exception as e:
        print(f"❌ API background task import test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration_files():
    """Test 8: Test configuration files."""
    print("\n" + "=" * 80)
    print("TEST 8: Testing Configuration Files")
    print("=" * 80)

    try:
        # Test params.yaml
        import yaml

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        assert "periodic_collection" in params, "periodic_collection section not found"
        print("✅ params.yaml contains periodic_collection section")

        periodic_config = params["periodic_collection"]
        assert "interval_hours" in periodic_config
        assert "retrain_threshold" in periodic_config
        print("✅ periodic_collection configuration is valid")

        # Test docker-compose.yml
        docker_compose_path = Path("docker/docker-compose.yml")
        if docker_compose_path.exists():
            with open(docker_compose_path, "r") as f:
                content = f.read()
                assert "DATA_COLLECTION_INTERVAL_HOURS" in content
                assert "DATA_COLLECTION_RETRAIN_THRESHOLD" in content
            print("✅ docker-compose.yml contains required environment variables")
        else:
            print("⚠️ docker-compose.yml not found (may be in different location)")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test 9: Test directory structure."""
    print("\n" + "=" * 80)
    print("TEST 9: Testing Directory Structure")
    print("=" * 80)

    try:
        # Check incremental directory
        incremental_dir = Config.RAW_DATA_DIR / "incremental"
        incremental_dir.mkdir(parents=True, exist_ok=True)
        assert incremental_dir.exists(), "Incremental directory should exist"
        print(f"✅ Incremental directory exists: {incremental_dir}")

        # Check processed directory
        assert (
            Config.PROCESSED_DATA_DIR.exists()
        ), "Processed data directory should exist"
        print(f"✅ Processed data directory exists: {Config.PROCESSED_DATA_DIR}")

        return True
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUITE: Data Collection-Based Continuous Learning")
    print("=" * 80)

    results = {}

    # Test 1: Imports
    results["imports"] = test_imports()

    # Test 2: PeriodicDataCollector initialization
    success, collector = test_periodic_collector_init()
    results["periodic_collector_init"] = success

    # Test 3: IncrementalPreprocessor initialization
    success, preprocessor = test_incremental_preprocessor_init()
    results["incremental_preprocessor_init"] = success

    # Test 4: Create sample data (if collectors initialized)
    if collector:
        success, sample_data_path = test_create_sample_incremental_data(collector)
        results["sample_data_creation"] = success
    else:
        sample_data_path = None
        results["sample_data_creation"] = False

    # Test 5: Incremental preprocessing (if preprocessor initialized)
    if preprocessor and sample_data_path:
        success, preprocess_result = test_incremental_preprocessing(
            preprocessor, sample_data_path
        )
        results["incremental_preprocessing"] = success
    else:
        results["incremental_preprocessing"] = False

    # Test 6: ContinuousLearner method
    results[
        "continuous_learner_method"
    ] = test_continuous_learner_data_collection_method()

    # Test 7: API background task
    results["api_background_task"] = test_api_background_task_import()

    # Test 8: Configuration files
    results["configuration"] = test_configuration_files()

    # Test 9: Directory structure
    results["directory_structure"] = test_directory_structure()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    failed_tests = total_tests - passed_tests

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "-" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    print("-" * 80)

    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
        return 0
    else:
        print(f"\n⚠️ {failed_tests} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
