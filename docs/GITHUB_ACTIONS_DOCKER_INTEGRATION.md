# **INTEGRASI GITHUB ACTIONS DAN DOCKER**
## **Dokumentasi Lengkap Deployment Pipeline**

---

## **DOKUMENT CONTROL**

| Version | Date | Author | Changes |
| --- | --- | --- | --- |
| 1.0 | 14 November 2025 | Hafiyan Al Muqaffi Umary | Dokumentasi lengkap integrasi GitHub Actions dan Docker |

---

## **DAFTAR ISI**

1. [Overview](#overview)
2. [Arsitektur Integrasi](#arsitektur-integrasi)
3. [GitHub Actions Workflows](#github-actions-workflows)
4. [Docker Configuration](#docker-configuration)
5. [Deployment Pipeline](#deployment-pipeline)
6. [Cara Kerja Integrasi](#cara-kerja-integrasi)
7. [Setup & Configuration](#setup--configuration)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## **1. OVERVIEW**

### **1.1 Tujuan Integrasi**

Integrasi GitHub Actions dan Docker dalam project ini bertujuan untuk:

- ✅ **Automated CI/CD Pipeline**: Otomatisasi build, test, dan deployment
- ✅ **Containerization**: Package aplikasi dalam Docker containers untuk konsistensi
- ✅ **Quality Assurance**: Automated testing dan code quality checks
- ✅ **Model Training Automation**: Automated model training dan evaluation
- ✅ **Data Collection Automation**: Scheduled data collection dari Reddit dan Kaggle
- ✅ **Deployment Automation**: Automated Docker image building dan testing

### **1.2 Komponen Utama**

1. **GitHub Actions Workflows** (4 workflows):
   - CI Pipeline (Code Quality & Testing)
   - Data Collection Pipeline
   - Model Training Pipeline
   - Deployment Pipeline

2. **Docker Containers** (4 services):
   - Sentiment API (FastAPI)
   - Dashboard (Streamlit)
   - Prometheus (Metrics)
   - Grafana (Visualization)

3. **Docker Images**:
   - `movie-sentiment-api:latest`
   - `movie-sentiment-dashboard:latest`

---

## **2. ARSITEKTUR INTEGRASI**

### **2.1 Diagram Alur Integrasi**

```
┌─────────────────────────────────────────────────────────────┐
│                    GITHUB REPOSITORY                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         GitHub Actions Workflows                      │  │
│  │                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │
│  │  │ CI Pipeline   │  │ Data Pipeline│  │ Training  │  │  │
│  │  │               │  │              │  │ Pipeline  │  │  │
│  │  └──────┬────────┘  └──────┬───────┘  └─────┬─────┘  │  │
│  │         │                  │                 │        │  │
│  │         └──────────────────┴─────────────────┘        │  │
│  │                        │                              │  │
│  │                        ▼                              │  │
│  │              ┌──────────────────┐                    │  │
│  │              │ Deployment       │                    │  │
│  │              │ Pipeline         │                    │  │
│  │              └────────┬─────────┘                    │  │
│  └───────────────────────┼──────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Docker Buildx         │
              │   - Build API Image     │
              │   - Build Dashboard    │
              │   - Test Containers    │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Docker Images        │
              │   - API Image          │
              │   - Dashboard Image   │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Docker Compose       │
              │   - sentiment-api       │
              │   - dashboard          │
              │   - prometheus         │
              │   - grafana            │
              └────────────────────────┘
```

### **2.2 Workflow Dependencies**

```
CI Pipeline (Independent)
    │
    ├──► Code Quality Checks
    ├──► Unit Tests
    └──► Integration Tests (Skipped)

Data Collection Pipeline (Independent)
    │
    ├──► Collect from Reddit
    ├──► Collect from Kaggle
    └──► Update DVC

Model Training Pipeline (Independent)
    │
    ├──► Train Traditional ML
    ├──► Train Transformer
    ├──► Evaluate Models
    └──► Performance Gates

Deployment Pipeline (Dependent on Training)
    │
    ├──► Triggered by Training Success
    ├──► Build Docker Images
    ├──► Test Containers
    └──► Upload Artifacts
```

---

## **3. GITHUB ACTIONS WORKFLOWS**

### **3.1 CI Pipeline** (`.github/workflows/ci-pipeline.yml`)

#### **3.1.1 Tujuan**
- Code quality checks (flake8, black, isort)
- Unit testing dengan coverage
- Integration testing (temporarily skipped)

#### **3.1.2 Triggers**
```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
```

#### **3.1.3 Jobs**

**A. Code Quality & Linting**
- **Timeout**: 25 minutes
- **Steps**:
  1. Checkout code
  2. Setup Python 3.9
  3. Cache pip packages
  4. Install linting tools (flake8, black, isort)
  5. Run flake8 (critical errors)
  6. Run flake8 (warnings)
  7. Auto-format with black
  8. Format check with black
  9. Import sorting check with isort

**B. Unit Tests**
- **Timeout**: 25 minutes
- **Dependencies**: `requirements-test.txt` (minimal)
- **Steps**:
  1. Checkout code
  2. Setup Python 3.9
  3. Cache pip packages
  4. Install test dependencies
  5. Run unit tests (fast, no coverage)
  6. Run unit tests with coverage
  7. Upload coverage to Codecov

**C. Integration Tests**
- **Status**: Skipped (`if: false`)
- **Reason**: Long installation time (40+ minutes)

**D. Test Summary**
- Aggregates results from all test jobs

#### **3.1.4 Key Features**

```yaml
# Optimized dependency installation
- Uses minimal requirements-test.txt for unit tests
- Caches pip packages for faster builds
- Skips heavy dependencies (torch, transformers) in unit tests

# Tolerant formatting checks
- Auto-formats files before checking
- Provides warnings instead of failures for formatting issues
- Handles line ending differences (CRLF/LF)
```

### **3.2 Data Collection Pipeline** (`.github/workflows/data-pipeline.yml`)

#### **3.2.1 Tujuan**
- Automated data collection from Reddit dan Kaggle
- Data validation
- DVC update dan commit

#### **3.2.2 Triggers**
```yaml
on:
  schedule:
    - cron: "0 0 * * 0"  # Weekly on Sunday at midnight UTC
  workflow_dispatch:  # Manual trigger
```

#### **3.2.3 Jobs**

**Collect Data**
- **Timeout**: 60 minutes
- **Dependencies**: `requirements-data-collection.txt` (minimal)
- **Steps**:
  1. Checkout code
  2. Setup Python 3.9
  3. Cache pip packages
  4. Install minimal dependencies
  5. Pull latest data (DVC)
  6. Collect from Reddit (if enabled)
  7. Collect from Kaggle (if enabled)
  8. Validate collected data
  9. Update DVC
  10. Commit and push changes

#### **3.2.4 Required Secrets**

```yaml
REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET
REDDIT_USER_AGENT
KAGGLE_USERNAME
KAGGLE_KEY
```

### **3.3 Model Training Pipeline** (`.github/workflows/model-training.yml`)

#### **3.3.1 Tujuan**
- Automated model training (Traditional ML & Transformer)
- Model evaluation dan comparison
- Performance gates
- Model registration

#### **3.3.2 Triggers**
```yaml
on:
  schedule:
    - cron: "0 2 * * 1"  # Weekly on Monday at 2 AM UTC
  workflow_dispatch:  # Manual trigger with options
```

#### **3.3.3 Jobs**

**Train Model**
- **Timeout**: 120 minutes
- **Steps**:
  1. Checkout code
  2. Setup Python 3.9
  3. Install dependencies (`requirements.txt`)
  4. Setup DVC
  5. Pull latest data
  6. Train Traditional ML models (if enabled)
  7. Train Transformer model (if enabled)
  8. Compare and evaluate models
  9. Check performance gates (min accuracy: 85%, min F1: 83%)
  10. Verify model registration
  11. Upload model artifacts
  12. Notify on failure (creates GitHub issue)

#### **3.3.4 Performance Gates**

```python
min_accuracy = 0.85  # 85%
min_f1 = 0.83        # 83%

# Fails if model doesn't meet thresholds
if accuracy < min_accuracy:
    raise ValueError(f'Accuracy {accuracy:.2%} < threshold {min_accuracy:.2%}')
if f1 < min_f1:
    raise ValueError(f'F1 {f1:.2%} < threshold {min_f1:.2%}')
```

### **3.4 Deployment Pipeline** (`.github/workflows/deployment.yml`)

#### **3.4.1 Tujuan**
- Build Docker images (API & Dashboard)
- Test containers (health checks, API endpoints)
- Upload Docker images as artifacts

#### **3.4.2 Triggers**
```yaml
on:
  workflow_run:
    workflows: ["Model Training Pipeline"]
    types: [completed]
  workflow_dispatch:  # Manual trigger
```

#### **3.4.3 Jobs**

**Build & Test Docker Images**
- **Timeout**: 30 minutes
- **Condition**: Only runs if training pipeline succeeded
- **Steps**:
  1. Checkout code
  2. Setup Docker Buildx
  3. Build API Docker image
  4. Build Dashboard Docker image
  5. Test containers:
     - Start services with docker-compose
     - Wait for services (30s)
     - Health check API (`/health`)
     - Health check Dashboard (`/_stcore/health`)
     - Test prediction endpoint (`/predict`)
     - Cleanup
  6. Save Docker images (tar.gz)
  7. Upload Docker images as artifacts

#### **3.4.4 Docker Build Configuration**

```yaml
# API Image
context: .
file: ./docker/Dockerfile
tags: movie-sentiment-api:latest
cache-from: type=gha
cache-to: type=gha,mode=max

# Dashboard Image
context: .
file: ./docker/Dockerfile.dashboard
tags: movie-sentiment-dashboard:latest
cache-from: type=gha
cache-to: type=gha,mode=max
```

#### **3.4.5 Container Testing**

```bash
# Start services
docker-compose up -d sentiment-api sentiment-dashboard

# Wait for services
sleep 30

# Health checks
curl -f http://localhost:8000/health
curl -f http://localhost:8501/_stcore/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is great!"}'

# Cleanup
docker-compose down
```

---

## **4. DOCKER CONFIGURATION**

### **4.1 Docker Compose** (`docker/docker-compose.yml`)

#### **4.1.1 Services**

**A. Sentiment API** (`sentiment-api`)
- **Image**: Built from `docker/Dockerfile`
- **Port**: `8000:8000`
- **Environment Variables**:
  ```yaml
  LOG_LEVEL: INFO
  API_HOST: 0.0.0.0
  API_PORT: 8000
  RETRAIN_THRESHOLD: 10
  MIN_IMPROVEMENT: 0.01
  RETRAIN_INTERVAL_HOURS: 1
  DATA_COLLECTION_INTERVAL_HOURS: 24
  DATA_COLLECTION_RETRAIN_THRESHOLD: 3000
  ```
- **Volumes**:
  - `../logs:/app/logs`
  - `../src:/app/src:ro`
  - `../data:/app/data`
- **Health Check**: `curl -f http://localhost:8000/health`
- **GPU Support**: Optional (NVIDIA GPU)

**B. Dashboard** (`dashboard`)
- **Image**: Built from `docker/Dockerfile.dashboard`
- **Port**: `8501:7860` (Streamlit default: 7860)
- **Environment Variables**:
  ```yaml
  STREAMLIT_SERVER_PORT: 7860
  STREAMLIT_SERVER_ADDRESS: 0.0.0.0
  STREAMLIT_SERVER_RUN_ON_SAVE: true
  ```
- **Volumes**:
  - `../data/database:/app/data/database`
  - `../logs:/app/logs`
  - `../src:/app/src:ro`
  - `../models:/app/models:ro`
  - `../data/processed:/app/data/processed:ro`
- **Health Check**: `curl -f http://localhost:7860/_stcore/health`
- **Depends On**: `sentiment-api`

**C. Prometheus** (`prometheus`)
- **Image**: `prom/prometheus:latest`
- **Port**: `9090:9090`
- **Volumes**:
  - `./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro`
  - `prometheus_data:/prometheus`
- **Retention**: 200 hours
- **Depends On**: `sentiment-api`

**D. Grafana** (`grafana`)
- **Image**: `grafana/grafana:latest`
- **Port**: `3000:3000`
- **Environment Variables**:
  ```yaml
  GF_SECURITY_ADMIN_PASSWORD: admin123
  GF_USERS_ALLOW_SIGN_UP: false
  ```
- **Volumes**:
  - `grafana_data:/var/lib/grafana`
  - `./grafana/provisioning:/etc/grafana/provisioning:ro`
  - `./grafana/dashboards:/var/lib/grafana/dashboards:ro`
- **Depends On**: `prometheus`

#### **4.1.2 Networks**
- **sentiment-network**: Bridge network untuk semua services

#### **4.1.3 Volumes**
- **prometheus_data**: Persistent storage untuk Prometheus
- **grafana_data**: Persistent storage untuk Grafana

### **4.2 Dockerfile API** (`docker/Dockerfile`)

#### **4.2.1 Multi-Stage Build**

**Stage 1: Builder**
- Base: `python:3.11-slim`
- Installs all dependencies in stages:
  1. fsspec (pinned version)
  2. Core packages (pandas, numpy, scipy, scikit-learn)
  3. PyTorch
  4. Transformers & datasets
  5. MLOps tools (MLflow)
  6. DVC
  7. FastAPI & Uvicorn
  8. Remaining packages

**Stage 2: Runtime**
- Base: `python:3.11-slim`
- Copies packages from builder
- Installs minimal runtime dependencies
- Copies application code
- Copies models
- Creates directories
- Exposes port 8000
- Health check configured

#### **4.2.2 Key Features**

```dockerfile
# Optimized dependency installation
- Multi-stage build untuk mengurangi image size
- Pinned versions untuk reproducibility
- Dependency resolution untuk menghindari conflicts

# Security
- Non-root user (appuser) - commented out for now
- Minimal base image (slim)
- No unnecessary packages

# Health Check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### **4.3 Dockerfile Dashboard** (`docker/Dockerfile.dashboard`)

#### **4.3.1 Single-Stage Build**
- Base: `python:3.11-slim`
- Installs dependencies from `requirements-dashboard.txt`
- Copies application files
- Copies models
- Copies processed data
- Creates database directory
- Exposes port 7860
- Health check configured

#### **4.3.2 Key Features**

```dockerfile
# Streamlit Configuration
CMD ["streamlit", "run", "src/dashboard/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
```

---

## **5. DEPLOYMENT PIPELINE**

### **5.1 Alur Deployment Lengkap**

```
1. Code Push / PR
   │
   ├──► CI Pipeline (Code Quality & Tests)
   │    └──► Must pass before merge
   │
2. Model Training Trigger (Schedule / Manual)
   │
   ├──► Data Collection (if needed)
   │    └──► Collect from Reddit/Kaggle
   │
   ├──► Model Training
   │    ├──► Train Traditional ML
   │    ├──► Train Transformer
   │    └──► Evaluate & Compare
   │
   ├──► Performance Gates
   │    └──► Must meet thresholds
   │
   └──► Model Registration
        │
3. Deployment Pipeline (Triggered by Training Success)
   │
   ├──► Build Docker Images
   │    ├──► Build API Image
   │    └──► Build Dashboard Image
   │
   ├──► Test Containers
   │    ├──► Start services
   │    ├──► Health checks
   │    └──► API endpoint tests
   │
   └──► Upload Artifacts
        └──► Docker images (tar.gz)
```

### **5.2 Deployment Scenarios**

#### **Scenario 1: Automated (Recommended)**
1. Model training completes successfully
2. Deployment pipeline automatically triggers
3. Docker images built and tested
4. Artifacts uploaded for download

#### **Scenario 2: Manual**
1. Go to GitHub Actions → Deployment Pipeline
2. Click "Run workflow"
3. Choose options:
   - Build Docker: true/false
   - Test containers: true/false
4. Workflow runs and produces artifacts

### **5.3 Artifact Management**

**Location**: GitHub Actions → Artifacts
**Retention**: 7 days
**Contents**:
- `api-image.tar.gz`
- `dashboard-image.tar.gz`

**Download & Load**:
```bash
# Download artifact from GitHub
# Extract
gunzip api-image.tar.gz
gunzip dashboard-image.tar.gz

# Load into Docker
docker load < api-image.tar
docker load < dashboard-image.tar

# Tag images
docker tag movie-sentiment-api:latest movie-sentiment-api:v1.0
docker tag movie-sentiment-dashboard:latest movie-sentiment-dashboard:v1.0
```

---

## **6. CARA KERJA INTEGRASI**

### **6.1 Trigger Chain**

```
Push to main/develop
    │
    ├──► CI Pipeline (runs immediately)
    │    └──► Code quality checks
    │    └──► Unit tests
    │
Schedule (Weekly)
    │
    ├──► Data Collection Pipeline (Sunday 00:00 UTC)
    │    └──► Collect new data
    │    └──► Update DVC
    │
    └──► Model Training Pipeline (Monday 02:00 UTC)
         │
         ├──► Train models
         ├──► Evaluate models
         ├──► Performance gates
         │
         └──► SUCCESS triggers Deployment Pipeline
              │
              ├──► Build Docker images
              ├──► Test containers
              └──► Upload artifacts
```

### **6.2 Dependency Resolution**

**GitHub Actions**:
- Uses `workflow_run` trigger untuk dependency
- Checks `conclusion == 'success'` before running
- Manual trigger bypasses dependency check

**Docker Compose**:
- Uses `depends_on` untuk service dependencies
- Health checks ensure services are ready
- Network isolation dengan bridge network

### **6.3 Caching Strategy**

**GitHub Actions**:
```yaml
# Pip packages
cache: ~/.cache/pip
key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

# Docker Buildx
cache-from: type=gha
cache-to: type=gha,mode=max
```

**Docker**:
- Multi-stage build untuk layer caching
- Requirements copied first untuk better caching
- Base images cached by Docker daemon

---

## **7. SETUP & CONFIGURATION**

### **7.1 GitHub Secrets Setup**

**Required Secrets**:

1. **Reddit API** (untuk Data Collection):
   ```
   REDDIT_CLIENT_ID
   REDDIT_CLIENT_SECRET
   REDDIT_USER_AGENT
   ```

2. **Kaggle API** (untuk Data Collection):
   ```
   KAGGLE_USERNAME
   KAGGLE_KEY
   ```

3. **DVC Remote** (optional, untuk data versioning):
   ```
   AWS_ACCESS_KEY_ID
   AWS_SECRET_ACCESS_KEY
   ```

**Cara Setup**:
1. Go to GitHub Repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each secret dengan name dan value

### **7.2 Local Docker Setup**

**Prerequisites**:
- Docker Desktop / Docker Engine
- Docker Compose v2

**Setup Steps**:
```bash
# 1. Clone repository
git clone https://github.com/your-username/movie-sentiment-mlops.git
cd movie-sentiment-mlops

# 2. Build Docker images
cd docker
docker-compose build

# 3. Start services
docker-compose up -d

# 4. Check services
docker-compose ps

# 5. View logs
docker-compose logs -f sentiment-api
docker-compose logs -f dashboard
```

### **7.3 Environment Variables**

**Create `.env` file** (optional):
```env
# API Configuration
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Continuous Learning
RETRAIN_THRESHOLD=10
MIN_IMPROVEMENT=0.01
RETRAIN_INTERVAL_HOURS=1
DATA_COLLECTION_INTERVAL_HOURS=24
DATA_COLLECTION_RETRAIN_THRESHOLD=3000

# Grafana
GF_SECURITY_ADMIN_PASSWORD=admin123
```

**Load in docker-compose**:
```yaml
env_file:
  - .env
```

---

## **8. TESTING & VALIDATION**

### **8.1 CI Pipeline Tests**

**Code Quality**:
```bash
# Run locally
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
black --check src/ scripts/
isort --check-only --profile black src/ scripts/
```

**Unit Tests**:
```bash
# Run locally
pytest tests/unit/ -v
pytest tests/unit/ --cov=src --cov-report=html
```

### **8.2 Docker Container Tests**

**Manual Testing**:
```bash
# Start services
docker-compose up -d

# Wait for services
sleep 30

# Health checks
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'

# Check logs
docker-compose logs sentiment-api
docker-compose logs dashboard
```

**Automated Testing** (in GitHub Actions):
```yaml
- name: Test containers
  run: |
    cd docker
    docker-compose up -d sentiment-api sentiment-dashboard
    sleep 30
    curl -f http://localhost:8000/health || exit 1
    curl -f http://localhost:8501/_stcore/health || exit 1
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "This movie is great!"}' || exit 1
    docker-compose down
```

### **8.3 Integration Tests**

**API Integration**:
```bash
# Test all endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this movie!"}'
```

**Dashboard Integration**:
```bash
# Open browser
http://localhost:8501

# Check pages load
- Predict page
- Performance page
- Insights page
```

---

## **9. TROUBLESHOOTING**

### **9.1 GitHub Actions Issues**

#### **Issue: Workflow Timeout**
**Symptoms**: Workflow fails dengan timeout
**Solutions**:
- Increase `timeout-minutes` in workflow
- Optimize dependency installation (use minimal requirements)
- Use caching untuk pip packages

#### **Issue: Docker Build Fails**
**Symptoms**: `docker/build-push-action` fails
**Solutions**:
- Check Dockerfile syntax
- Verify context path
- Check for missing files
- Review build logs

#### **Issue: Container Test Fails**
**Symptoms**: Health checks fail
**Solutions**:
- Increase wait time (`sleep 30` → `sleep 60`)
- Check service logs
- Verify ports are not in use
- Check environment variables

### **9.2 Docker Issues**

#### **Issue: Container Won't Start**
**Symptoms**: Container exits immediately
**Solutions**:
```bash
# Check logs
docker-compose logs sentiment-api

# Check container status
docker-compose ps

# Restart services
docker-compose restart sentiment-api
```

#### **Issue: Port Already in Use**
**Symptoms**: `Bind for 0.0.0.0:8000 failed: port is already allocated`
**Solutions**:
```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or change port in docker-compose.yml
```

#### **Issue: Volume Mount Errors**
**Symptoms**: Permission denied errors
**Solutions**:
```bash
# Fix permissions
sudo chown -R $USER:$USER data/ logs/

# Or run as root (not recommended for production)
```

### **9.3 Common Errors**

#### **Error: ModuleNotFoundError**
**Cause**: Missing dependencies or wrong PYTHONPATH
**Solution**:
```bash
# Check requirements.txt
pip install -r requirements.txt

# Verify PYTHONPATH in Dockerfile
ENV PYTHONPATH="/app"
```

#### **Error: Model Not Found**
**Cause**: Models not copied to Docker image
**Solution**:
```dockerfile
# Ensure models are copied
COPY models/distilbert/ ./models/distilbert/
```

#### **Error: DVC Remote Not Configured**
**Cause**: DVC remote not set up
**Solution**:
```bash
# Skip DVC pull in workflows
dvc pull || echo "No DVC remote configured"
```

---

## **10. BEST PRACTICES**

### **10.1 GitHub Actions**

✅ **DO**:
- Use caching untuk dependencies
- Set appropriate timeouts
- Use minimal requirements untuk faster builds
- Provide clear error messages
- Use `continue-on-error` untuk non-critical checks

❌ **DON'T**:
- Don't commit secrets
- Don't skip security checks
- Don't ignore test failures
- Don't use deprecated actions

### **10.2 Docker**

✅ **DO**:
- Use multi-stage builds
- Pin dependency versions
- Use minimal base images
- Implement health checks
- Use `.dockerignore`
- Run as non-root user (production)

❌ **DON'T**:
- Don't include unnecessary files
- Don't use `latest` tags in production
- Don't store secrets in images
- Don't run as root (if possible)

### **10.3 Docker Compose**

✅ **DO**:
- Use environment variables
- Implement health checks
- Use named volumes untuk persistence
- Set restart policies
- Use networks untuk isolation

❌ **DON'T**:
- Don't expose unnecessary ports
- Don't use default passwords
- Don't mount sensitive files

### **10.4 CI/CD Pipeline**

✅ **DO**:
- Run tests before deployment
- Use performance gates
- Test containers before pushing
- Keep artifacts for debugging
- Document deployment process

❌ **DON'T**:
- Don't deploy on test failures
- Don't skip security checks
- Don't ignore performance regressions

---

## **11. MONITORING & OBSERVABILITY**

### **11.1 GitHub Actions Monitoring**

**Workflow Status**:
- View di GitHub Actions tab
- Check workflow runs
- Review logs for errors

**Metrics**:
- Success rate
- Average run time
- Failure reasons

### **11.2 Docker Monitoring**

**Container Health**:
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Resource usage
docker stats
```

**Prometheus Metrics**:
- Available at `http://localhost:9090`
- Scrapes metrics from API
- Stores metrics for 200 hours

**Grafana Dashboards**:
- Available at `http://localhost:3000`
- Login: `admin` / `admin123`
- 17-panel comprehensive dashboard

---

## **12. DEPLOYMENT CHECKLIST**

### **12.1 Pre-Deployment**

- [ ] All tests passing
- [ ] Code quality checks passing
- [ ] Models trained and evaluated
- [ ] Performance gates met
- [ ] Docker images built successfully
- [ ] Container tests passing
- [ ] Secrets configured
- [ ] Environment variables set

### **12.2 Deployment**

- [ ] Download Docker images from artifacts
- [ ] Load images into Docker
- [ ] Update docker-compose.yml if needed
- [ ] Start services
- [ ] Verify health checks
- [ ] Test API endpoints
- [ ] Test Dashboard
- [ ] Check monitoring (Prometheus/Grafana)

### **12.3 Post-Deployment**

- [ ] Monitor logs
- [ ] Check metrics
- [ ] Verify predictions working
- [ ] Test continuous learning
- [ ] Document any issues
- [ ] Update documentation

---

## **13. CONCLUSION**

Integrasi GitHub Actions dan Docker dalam project ini menyediakan:

✅ **Automated CI/CD Pipeline** yang lengkap
✅ **Containerization** untuk konsistensi deployment
✅ **Quality Assurance** dengan automated testing
✅ **Model Training Automation** dengan performance gates
✅ **Data Collection Automation** dengan scheduled runs
✅ **Deployment Automation** dengan Docker image building dan testing

**Key Benefits**:
- **Reproducibility**: Docker ensures consistent environments
- **Automation**: GitHub Actions automates entire pipeline
- **Quality**: Automated tests ensure code quality
- **Efficiency**: Caching reduces build times
- **Reliability**: Health checks and monitoring

**Next Steps**:
1. Deploy to production (VPS/Cloud)
2. Setup SSL/HTTPS
3. Configure domain
4. Setup monitoring alerts
5. Implement backup strategy

---

## **14. REFERENCES**

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Document Version**: 1.0
**Last Updated**: 14 November 2025
**Author**: Hafiyan Al Muqaffi Umary
