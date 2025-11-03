# MLflow Usage Guide

## Overview

MLflow is used for experiment tracking, model versioning, and model registry in this project.

## Starting MLflow UI

```bash
# Using the startup script
./scripts/start_mlflow.sh

# Or manually
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Access UI at: http://localhost:5000

## Basic Usage

### 1. Initialize MLflow Manager

```python
from src.utils.mlflow_setup import setup_mlflow

manager = setup_mlflow()
```

### 2. Start a Run and Log Experiments

```python
with manager.start_run(run_name="my_experiment") as run:
    # Log parameters
    manager.log_params({
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 10
    })

    # Train model
    model = train_model()

    # Log metrics
    manager.log_metrics({
        'accuracy': 0.95,
        'f1_score': 0.93
    })

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 3. Register Model

```python
# After training
run_id = "your_run_id"
model_name = "sentiment_model"

model_version = manager.register_model(run_id, model_name)
```

### 4. Transition Model Stage

```python
# Move to staging
manager.transition_model_stage(
    model_name="sentiment_model",
    version=1,
    stage="Staging"
)

# Move to production
manager.transition_model_stage(
    model_name="sentiment_model",
    version=1,
    stage="Production"
)
```

## Experiment Organization

### Experiments Structure

- `movie_sentiment_analysis` - Main experiment
- `movie_sentiment_traditional_ml` - Traditional ML models
- `movie_sentiment_transformer` - Transformer models

### Run Naming Convention

Format: `{model_type}_{timestamp}`

Examples:

- `logistic_regression_20240103_143022`
- `distilbert_20240103_143022`

## Model Registry

### Registered Models

1. **sentiment\_logistic\_regression**
2. **sentiment\_random\_forest**
3. **sentiment\_svm**
4. **sentiment\_distilbert**

### Model Stages

- **None**: Newly registered model
- **Staging**: Model being tested
- **Production**: Model in production use
- **Archived**: Old/deprecated model

## Best Practices

1. **Always use meaningful run names**
2. **Log all hyperparameters**
3. **Log comprehensive metrics**
4. **Tag runs appropriately**
5. **Register best models**
6. **Document model transitions**

## Querying Experiments

### Get Best Run

```python
best_run = manager.get_best_run(metric_name="f1_score")
```

### Compare Runs

```python
comparison = manager.compare_runs(
    run_ids=["run_id_1", "run_id_2"],
    metrics=["accuracy", "f1_score"]
)
```

### Search Runs

```python
import mlflow

runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.f1_score DESC"]
)
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 5000
lsof -ti:5000

# Kill the process
lsof -ti:5000 | xargs kill -9
```

### Database Locked

```bash
# Stop MLflow server
pkill -f "mlflow"

# Restart
./scripts/start_mlflow.sh
```

### Clear All Experiments

```bash
# Delete MLflow database (WARNING: This deletes all data)
rm mlflow.db
rm -rf mlruns/

# Restart MLflow
./scripts/start_mlflow.sh
```

```javascript
---

## **Step 1.3.7: Run MLflow Tests**

```bash
# Run MLflow tests
pytest tests/test_mlflow_setup.py -v

# Check MLflow UI
# Open browser to http://localhost:5000
```

***

## **✅ Task 1.3 Completion Checklist**

```bash
# Verify MLflow is running
curl http://localhost:5000/health

# Verify MLflow manager
python -c "from src.utils.mlflow_setup import setup_mlflow; m = setup_mlflow(); print('✓ MLflow Manager OK')"

# Check experiments
python -c "
import mlflow
experiments = mlflow.search_experiments()
print(f'✓ Found {len(experiments)} experiment(s)')
for exp in experiments:
    print(f'  - {exp.name}')
"

# Run tests
pytest tests/test_mlflow_setup.py -v
```

**Checklist:**

```javascript
[ ] MLflow manager created
[ ] MLflow configuration file created
[ ] Startup script created and working
[ ] MLflow UI accessible at http://localhost:5000
[ ] Test run successfully logged
[ ] Tests passing
[ ] Documentation created
[ ] All changes committed to git
```

***

## **Commit Changes**

```bash
# Add all MLflow files
git add src/utils/mlflow_setup.py
git add configs/mlflow_config.yaml
git add scripts/start_mlflow.sh
git add tests/test_mlflow_setup.py
git add docs/MLFLOW_GUIDE.md

# Commit
git commit -m "Set up MLflow tracking and model registry"

# Push
git push origin main
```

***

# **🎯 CHECKPOINT 1.3 COMPLETE**

**What we've accomplished:**

✅ MLflow Manager class created

✅ MLflow configuration set up

✅ Startup script created

✅ MLflow UI running and accessible

✅ Test suite created and passing

✅ Documentation completed
