"""
Tests for MLflow setup and configuration.
"""


import mlflow
import pytest

from src.utils.config import Config
from src.utils.mlflow_setup import MLflowManager, setup_mlflow


class TestMLflowSetup:
    """Test MLflow setup and configuration."""

    def test_mlflow_manager_initialization(self):
        """Test MLflowManager initialization."""
        manager = MLflowManager()
        assert manager.tracking_uri == Config.MLFLOW_TRACKING_URI
        assert manager.experiment_name == Config.MLFLOW_EXPERIMENT_NAME
        assert manager.experiment is not None

    def test_experiment_creation(self):
        """Test experiment creation."""
        test_experiment_name = "test_experiment"
        manager = MLflowManager(experiment_name=test_experiment_name)
        assert isinstance(manager, MLflowManager)

        experiment = mlflow.get_experiment_by_name(test_experiment_name)
        assert experiment is not None
        assert experiment.name == test_experiment_name

    def test_start_run(self):
        """Test starting an MLflow run."""
        manager = MLflowManager()

        with manager.start_run(run_name="test_run") as run:
            assert run is not None
            assert mlflow.active_run() is not None

    def test_log_params(self):
        """Test logging parameters."""
        manager = MLflowManager()

        with manager.start_run(run_name="test_params"):
            test_params = {"learning_rate": 0.01, "batch_size": 32, "epochs": 10}
            manager.log_params(test_params)

            run = mlflow.active_run()
            run_data = mlflow.get_run(run.info.run_id)
            run_data = mlflow.get_run(run.info.run_id)

            for key, value in test_params.items():
                assert key in run_data.data.params
                assert run_data.data.params[key] == str(value)

    def test_log_metrics(self):
        """Test logging metrics."""
        manager = MLflowManager()

        with manager.start_run(run_name="test_metrics"):
            test_metrics = {"accuracy": 0.95, "f1_score": 0.93, "precision": 0.94}
            manager.log_metrics(test_metrics)

            run = mlflow.active_run()
            run_data = mlflow.get_run(run.info.run_id)

            for key, value in test_metrics.items():
                assert key in run_data.data.metrics
                assert abs(run_data.data.metrics[key] - value) < 0.001

    def test_setup_mlflow_convenience_function(self):
        """Test setup_mlflow convenience function."""
        manager = setup_mlflow()
        assert isinstance(manager, MLflowManager)
        assert manager.experiment is not None


class TestMLflowIntegration:
    """Test MLflow integration with models."""

    def test_log_sklearn_model(self):
        """Test logging scikit-learn model."""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression

        manager = MLflowManager()

        # Create dummy model
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = LogisticRegression()
        model.fit(X, y)

        with manager.start_run(run_name="test_sklearn_model"):
            mlflow.sklearn.log_model(model, "model")

            run = mlflow.active_run()

            # Check if model artifact exists
            artifacts = [
                a.path
                for a in mlflow.tracking.MlflowClient().list_artifacts(run.info.run_id)
            ]
            assert "model" in artifacts

    def test_model_registry_operations(self):
        """Test model registry operations."""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression

        manager = MLflowManager()
        model_name = "test_model_registry"

        # Train and log model
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = LogisticRegression()
        model.fit(X, y)

        with manager.start_run(run_name="test_registry"):
            mlflow.sklearn.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id

        # Register model
        model_version = manager.register_model(run_id, model_name)
        assert model_version is not None
        assert model_version.name == model_name

        # Transition to staging
        updated_version = manager.transition_model_stage(
            model_name, model_version.version, "Staging"
        )
        assert updated_version.current_stage == "Staging"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
