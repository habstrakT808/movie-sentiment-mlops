# DVC Usage Guide

## Overview

This project uses DVC (Data Version Control) to manage data and model versioning.

## Basic Commands

### Check Status
```bash
dvc status
```

### Add Data to DVC

```bash
# Add a file or directory
dvc add data/raw/my_data.csv

# This creates my_data.csv.dvc file
# Add .dvc file to git
git add data/raw/my_data.csv.dvc data/raw/.gitignore
git commit -m "Add dataset"
```

### Push Data to Remote

```bash
dvc push
```

### Pull Data from Remote

```bash
dvc pull
```

### Reproduce Pipeline

```bash
# Run all stages that have changed
dvc repro

# Run specific stage
dvc repro data_collection
```

### View Pipeline

```bash
# Show pipeline as DAG
dvc dag

# Show pipeline with ASCII art
dvc dag --ascii
```

## Pipeline Stages

Our DVC pipeline consists of:

1. **data\_collection** - Collect data from Reddit, Kaggle, IMDb
2. **data\_validation** - Validate and clean collected data
3. **preprocessing** - Preprocess text data
4. **feature\_engineering** - Create features for ML models
5. **train\_traditional\_ml** - Train traditional ML models
6. **train\_transformer** - Train transformer model
7. **evaluate\_models** - Evaluate all models

## Working with Parameters

Parameters are defined in `params.yaml`. To modify:

```bash
# Edit params.yaml
nano params.yaml

# Reproduce pipeline with new parameters
dvc repro
```

## Data Versioning Workflow

### Adding New Data

```bash
# Collect new data
python src/data_collection/collect_all.py

# Add to DVC
dvc add data/raw/reddit_reviews.csv

# Commit .dvc file
git add data/raw/reddit_reviews.csv.dvc
git commit -m "Update Reddit data"

# Push data to remote
dvc push

# Push git changes
git push
```

### Switching Between Data Versions

```bash
# Checkout specific git commit
git checkout <commit-hash>

# Pull corresponding data
dvc pull
```

## Remote Storage

### Current Configuration

- Type: Local filesystem
- Location: `/d/dvc-storage/movie-sentiment-mlops`

### Changing Remote (Future)

```bash
# Add Google Drive remote
dvc remote add gdrive gdrive://folder_id

# Add S3 remote
dvc remote add s3remote s3://mybucket/path

# Set as default
dvc remote default s3remote
```

## Troubleshooting

### DVC Cache Issues

```bash
# Clear cache
dvc cache dir

# Verify cache
dvc cache dir -v
```

### Remote Connection Issues

```bash
# Check remote configuration
dvc remote list

# Test remote connection
dvc push --remote local -v
```

### Pipeline Issues

```bash
# Force reproduce all stages
dvc repro --force

# Unlock DVC files
dvc unlock data/raw/my_data.csv
```

## Best Practices

1. **Always commit .dvc files to git**
2. **Push data to DVC remote before pushing to git**
3. **Use meaningful commit messages**
4. **Don't commit large files to git**
5. **Regularly check&#32;`dvc status`**
6. **Document parameter changes in params.yaml**

```javascript
---

## **✅ Task 1.2 Completion Checklist**

```bash
# Verify DVC initialization
dvc version

# Verify DVC remote
dvc remote list

# Verify DVC config
cat .dvc/config

# Verify params.yaml
cat params.yaml | head -20

# Verify dvc.yaml
cat dvc.yaml | head -30

# Test DVC add/push/pull
echo "test" > data/raw/test_dvc.txt
dvc add data/raw/test_dvc.txt
dvc push
rm data/raw/test_dvc.txt
dvc pull
cat data/raw/test_dvc.txt
```

**Checklist:**

```javascript
[ ] DVC initialized
[ ] DVC remote configured (local storage)
[ ] params.yaml created with all parameters
[ ] dvc.yaml created with pipeline stages
[ ] .dvcignore configured
[ ] DVC helper scripts created
[ ] DVC documentation created
[ ] Test file successfully added, pushed, and pulled
[ ] All changes committed to git
```

***

## **Commit Changes**

```bash
# Add all new files
git add .dvc/ dvc.yaml params.yaml .dvcignore
git add scripts/dvc_helpers.sh docs/DVC_GUIDE.md
git add data/raw/.gitkeep data/processed/.gitkeep data/external/.gitkeep
git add models/.gitkeep metrics/.gitkeep

# Commit
git commit -m "Configure DVC pipeline and parameters"

# Push to GitHub
git push origin main
```

***

# **🎯 CHECKPOINT 1.2 COMPLETE**

**What we've accomplished:**

✅ DVC initialized in the project

✅ Local remote storage configured

✅ Complete pipeline structure defined in dvc.yaml

✅ All parameters configured in params.yaml

✅ DVC helper scripts created

✅ Documentation completed

✅ DVC tested and verified working
