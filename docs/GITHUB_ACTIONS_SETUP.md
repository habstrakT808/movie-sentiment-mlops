# GitHub Actions CI/CD Setup Guide

## Overview

This project uses GitHub Actions for automated CI/CD pipelines. The workflows include:

1. **CI Pipeline** - Code quality, linting, and testing
2. **Data Collection Pipeline** - Automated periodic data collection
3. **Model Training Pipeline** - Automated model training and evaluation
4. **Deployment Pipeline** - Docker image building and testing

## Required GitHub Secrets

To use the workflows, you need to configure the following secrets in your GitHub repository:

### Repository Settings → Secrets and variables → Actions

#### For Data Collection Pipeline:
- `REDDIT_CLIENT_ID` - Reddit API client ID
- `REDDIT_CLIENT_SECRET` - Reddit API client secret
- `REDDIT_USER_AGENT` - Reddit API user agent string
- `KAGGLE_USERNAME` - Kaggle username
- `KAGGLE_KEY` - Kaggle API key

#### For DVC (if using remote storage):
- `DVC_REMOTE_URL` - DVC remote storage URL (optional)
- `DVC_REMOTE_AUTH` - DVC remote authentication (optional)

## Workflow Details

### 1. CI Pipeline (`.github/workflows/ci-pipeline.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**Jobs:**
- **Code Quality**: Runs flake8, black, and isort checks
- **Unit Tests**: Runs unit tests with coverage reporting
- **Integration Tests**: Runs integration and API tests

**Duration:** ~10-15 minutes

### 2. Data Collection Pipeline (`.github/workflows/data-pipeline.yml`)

**Triggers:**
- Scheduled: Every Sunday at midnight UTC
- Manual: Via workflow_dispatch with source selection

**Jobs:**
- Collects new data from Reddit and/or Kaggle
- Validates collected data
- Updates DVC tracking
- Commits changes (if any)

**Duration:** ~30-60 minutes

### 3. Model Training Pipeline (`.github/workflows/model-training.yml`)

**Triggers:**
- Scheduled: Every Monday at 2 AM UTC
- Manual: Via workflow_dispatch with model type selection

**Jobs:**
- Trains Traditional ML models (Logistic Regression, Random Forest, SVM)
- Trains Transformer model (DistilBERT)
- Compares and evaluates all models
- Checks performance gates
- Uploads model artifacts

**Duration:** ~60-120 minutes (depends on model type)

**Inputs:**
- `model_type`: Choose `all`, `traditional`, or `transformer`
- `skip_evaluation`: Skip model evaluation (default: false)

### 4. Deployment Pipeline (`.github/workflows/deployment.yml`)

**Triggers:**
- After successful Model Training Pipeline
- Manual: Via workflow_dispatch

**Jobs:**
- Builds Docker images for API and Dashboard
- Tests containers (health checks, API endpoints)
- Uploads Docker images as artifacts

**Duration:** ~20-30 minutes

**Inputs:**
- `build_docker`: Build Docker images (default: true)
- `test_containers`: Test containers (default: true)

## Usage

### Running Workflows Manually

1. Go to **Actions** tab in GitHub
2. Select the workflow you want to run
3. Click **Run workflow**
4. Select branch and configure inputs
5. Click **Run workflow**

### Monitoring Workflows

- View workflow runs in the **Actions** tab
- Check logs for each step
- Download artifacts (models, Docker images, coverage reports)
- View workflow summaries

### Troubleshooting

**Common Issues:**

1. **Workflow fails on secrets:**
   - Ensure all required secrets are configured
   - Check secret names match exactly

2. **DVC pull fails:**
   - This is expected if no DVC remote is configured
   - Workflow will continue with local data

3. **Model training timeout:**
   - Increase timeout in workflow file
   - Or train models separately (traditional vs transformer)

4. **Docker build fails:**
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt

## Best Practices

1. **Test locally first:** Run workflows locally before pushing
2. **Monitor resource usage:** GitHub Actions has free tier limits
3. **Use workflow_dispatch:** Test workflows manually before enabling schedules
4. **Review logs:** Check workflow logs for errors and warnings
5. **Artifact retention:** Artifacts are kept for 7 days (configurable)

## Workflow Status Badges

Add these badges to your README.md:

```markdown
![CI Pipeline](https://github.com/your-username/movie-sentiment-mlops/workflows/CI%20Pipeline/badge.svg)
![Data Collection](https://github.com/your-username/movie-sentiment-mlops/workflows/Data%20Collection%20Pipeline/badge.svg)
![Model Training](https://github.com/your-username/movie-sentiment-mlops/workflows/Model%20Training%20Pipeline/badge.svg)
![Deployment](https://github.com/your-username/movie-sentiment-mlops/workflows/Deployment%20Pipeline/badge.svg)
```

## Next Steps

1. Configure GitHub secrets
2. Test workflows manually
3. Enable scheduled workflows (if desired)
4. Monitor workflow runs
5. Review and optimize workflow performance
