"""
Test code structure without importing dependencies.
Verifies that all new methods and functions exist and have correct structure.
"""

import ast
from pathlib import Path


def test_file_contains_method(file_path, class_name, method_name):
    """Test if a file contains a specific method in a class."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return True, f"Method {method_name} found in {class_name}"
        return False, f"Method {method_name} not found in {class_name}"
    except Exception as e:
        return False, f"Error parsing {file_path}: {e}"


def test_file_contains_function(file_path, function_name):
    """Test if a file contains a specific function."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return True, f"Function {function_name} found"
        return False, f"Function {function_name} not found"
    except Exception as e:
        return False, f"Error parsing {file_path}: {e}"


def test_file_contains_string(file_path, search_string):
    """Test if a file contains a specific string."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if search_string in content:
                return True, f"String '{search_string}' found"
            return False, f"String '{search_string}' not found"
    except Exception as e:
        return False, f"Error reading {file_path}: {e}"


def main():
    """Run structure tests."""
    print("\n" + "=" * 80)
    print("CODE STRUCTURE VERIFICATION")
    print("=" * 80)

    results = {}

    # Test 1: PeriodicDataCollector class and methods
    print("\n1. Testing PeriodicDataCollector...")
    file_path = Path("src/data_collection/periodic_collector.py")
    if file_path.exists():
        success, msg = test_file_contains_method(
            file_path, "PeriodicDataCollector", "collect_incremental_data"
        )
        results["periodic_collector_method"] = success
        print(f"   {'✅' if success else '❌'} {msg}")

        success, msg = test_file_contains_method(
            file_path, "PeriodicDataCollector", "_check_duplicates"
        )
        results["periodic_collector_check_duplicates"] = success
        print(f"   {'✅' if success else '❌'} {msg}")
    else:
        print("   ❌ File not found")
        results["periodic_collector_method"] = False

    # Test 2: IncrementalPreprocessor class and methods
    print("\n2. Testing IncrementalPreprocessor...")
    file_path = Path("src/preprocessing/incremental_preprocessor.py")
    if file_path.exists():
        success, msg = test_file_contains_method(
            file_path, "IncrementalPreprocessor", "preprocess_incremental_data"
        )
        results["incremental_preprocessor_method"] = success
        print(f"   {'✅' if success else '❌'} {msg}")

        success, msg = test_file_contains_method(
            file_path, "IncrementalPreprocessor", "_merge_with_existing"
        )
        results["incremental_preprocessor_merge"] = success
        print(f"   {'✅' if success else '❌'} {msg}")
    else:
        print("   ❌ File not found")
        results["incremental_preprocessor_method"] = False

    # Test 3: ContinuousLearner new method
    print("\n3. Testing ContinuousLearner...")
    file_path = Path("src/training/continuous_learning.py")
    if file_path.exists():
        success, msg = test_file_contains_method(
            file_path, "ContinuousLearner", "trigger_data_collection_retraining"
        )
        results["continuous_learner_method"] = success
        print(f"   {'✅' if success else '❌'} {msg}")

        # Check if old method still exists (should not be removed)
        success, msg = test_file_contains_method(
            file_path, "ContinuousLearner", "trigger_retraining"
        )
        results["continuous_learner_old_method"] = success
        print(f"   {'✅' if success else '❌'} Old method still exists: {msg}")
    else:
        print("   ❌ File not found")
        results["continuous_learner_method"] = False

    # Test 4: API background task function
    print("\n4. Testing API background task...")
    file_path = Path("src/deployment/api.py")
    if file_path.exists():
        success, msg = test_file_contains_function(
            file_path, "run_periodic_data_collection"
        )
        results["api_background_task"] = success
        print(f"   {'✅' if success else '❌'} {msg}")

        # Check if it's async
        success, msg = test_file_contains_string(
            file_path, "async def run_periodic_data_collection"
        )
        results["api_background_task_async"] = success
        print(f"   {'✅' if success else '❌'} Function is async: {msg}")
    else:
        print("   ❌ File not found")
        results["api_background_task"] = False

    # Test 5: Scheduler job in startup
    print("\n5. Testing scheduler registration...")
    file_path = Path("src/deployment/api.py")
    if file_path.exists():
        success, msg = test_file_contains_string(file_path, "periodic_data_collection")
        results["scheduler_registration"] = success
        print(f"   {'✅' if success else '❌'} Scheduler job registered: {msg}")
    else:
        results["scheduler_registration"] = False

    # Test 6: Docker compose environment variables
    print("\n6. Testing docker-compose.yml...")
    file_path = Path("docker/docker-compose.yml")
    if file_path.exists():
        success, msg = test_file_contains_string(
            file_path, "DATA_COLLECTION_INTERVAL_HOURS"
        )
        results["docker_env_interval"] = success
        print(f"   {'✅' if success else '❌'} {msg}")

        success, msg = test_file_contains_string(
            file_path, "DATA_COLLECTION_RETRAIN_THRESHOLD"
        )
        results["docker_env_threshold"] = success
        print(f"   {'✅' if success else '❌'} {msg}")
    else:
        results["docker_env_interval"] = False
        results["docker_env_threshold"] = False

    # Test 7: params.yaml configuration
    print("\n7. Testing params.yaml...")
    file_path = Path("params.yaml")
    if file_path.exists():
        success, msg = test_file_contains_string(file_path, "periodic_collection")
        results["params_periodic_collection"] = success
        print(f"   {'✅' if success else '❌'} {msg}")

        success, msg = test_file_contains_string(file_path, "interval_hours")
        results["params_interval_hours"] = success
        print(f"   {'✅' if success else '❌'} {msg}")
    else:
        results["params_periodic_collection"] = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")

    print("\n" + "-" * 80)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    print("-" * 80)

    if failed == 0:
        print("\n🎉 ALL STRUCTURE TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())
