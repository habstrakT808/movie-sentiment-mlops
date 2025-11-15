# **INTEGRASI GITHUB ACTIONS DAN DOCKER**
## **Dokumentasi Deployment Pipeline**

---

## **OVERVIEW**

### **Tujuan**
- ✅ Automated CI/CD Pipeline (build, test, deploy)
- ✅ Containerization dengan Docker untuk konsistensi
- ✅ Quality Assurance dengan automated testing
- ✅ Model Training Automation dengan performance gates
- ✅ Data Collection Automation (scheduled)
- ✅ Deployment Automation (Docker build & test)

### **Komponen Utama**

**GitHub Actions Workflows** (4 workflows):
- CI Pipeline (Code Quality & Testing)
- Data Collection Pipeline
- Model Training Pipeline
- Deployment Pipeline

**Docker Containers** (4 services):
- Sentiment API (FastAPI) - Port 8000
- Dashboard (Streamlit) - Port 8501
- Prometheus (Metrics) - Port 9090
- Grafana (Visualization) - Port 3000

---

## **ARSITEKTUR INTEGRASI**

```
Code Push/PR → CI Pipeline (Quality & Tests)
                    ↓
Schedule/Manual → Data Collection → Model Training
                                      ↓
                              Deployment Pipeline
                                      ↓
                              Docker Build & Test
                                      ↓
                              Upload Artifacts
```

### **Workflow Dependencies**

```
CI Pipeline (Independent)
Data Collection Pipeline (Independent)
Model Training Pipeline (Independent)
    ↓ (triggers on success)
Deployment Pipeline
    ├── Build Docker Images
    ├── Test Containers
    └── Upload Artifacts
```

---

## **GITHUB ACTIONS WORKFLOWS**

> 📸 **SCREENSHOT 1**: GitHub Actions Workflows List
>
> **Lokasi**: Repository → Actions tab
>
> **Yang ditampilkan**: List semua 4 workflows (CI Pipeline, Data Collection Pipeline, Model Training Pipeline, Deployment Pipeline)
>
> **Cara ambil**:
> 1. Buka repository di GitHub
> 2. Klik tab "Actions"
> 3. Screenshot halaman workflows list
>
> **File**: `screenshots/01-github-actions-workflows-list.png`

### **1. CI Pipeline** (`.github/workflows/ci-pipeline.yml`)

**Triggers**: Push/PR ke `main` atau `develop`

**Jobs**:
- **Code Quality**: flake8, black, isort (timeout: 25m)
- **Unit Tests**: pytest dengan coverage (timeout: 25m)
- **Integration Tests**: Skipped (too slow)

**Key Features**:
- Minimal dependencies (`requirements-test.txt`)
- Pip caching untuk speed
- Auto-format sebelum check
- Tolerant formatting checks

> 📸 **SCREENSHOT 2**: CI Pipeline Workflow Run (Success)
>
> **Lokasi**: Repository → Actions → CI Pipeline → Latest run
>
> **Yang ditampilkan**:
> - Workflow run status (green checkmark)
> - Jobs: Code Quality, Unit Tests, Test Summary (all passed)
> - Run time dan duration
>
> **Cara ambil**:
> 1. Buka Actions tab
> 2. Klik "CI Pipeline" workflow
> 3. Pilih latest successful run
> 4. Screenshot workflow summary dengan semua jobs passed
>
> **File**: `screenshots/02-ci-pipeline-success.png`

### **2. Data Collection Pipeline** (`.github/workflows/data-pipeline.yml`)

**Triggers**:
- Schedule: Weekly (Sunday 00:00 UTC)
- Manual: `workflow_dispatch`

**Jobs**:
- Collect dari Reddit & Kaggle
- Validate data
- Update DVC
- Commit & push changes

**Required Secrets**:
```
REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET
REDDIT_USER_AGENT
KAGGLE_USERNAME
KAGGLE_KEY
```

> 📸 **SCREENSHOT 3**: Data Collection Pipeline Workflow Run
>
> **Lokasi**: Repository → Actions → Data Collection Pipeline → Latest run
>
> **Yang ditampilkan**:
> - Workflow run status
> - Steps: Collect from Reddit, Collect from Kaggle, Validate data
> - Summary dengan jumlah samples collected
>
> **Cara ambil**:
> 1. Buka Actions tab
> 2. Klik "Data Collection Pipeline"
> 3. Pilih latest run (manual atau scheduled)
> 4. Screenshot workflow dengan steps yang completed
>
> **File**: `screenshots/03-data-collection-pipeline.png`

### **3. Model Training Pipeline** (`.github/workflows/model-training.yml`)

**Triggers**:
- Schedule: Weekly (Monday 02:00 UTC)
- Manual: `workflow_dispatch` dengan options

**Jobs**:
- Train Traditional ML models
- Train Transformer model
- Evaluate & compare models
- Performance gates (min accuracy: 85%, min F1: 83%)
- Register models
- Upload artifacts

**Performance Gates**:
```python
min_accuracy = 0.85  # 85%
min_f1 = 0.83        # 83%
```

> 📸 **SCREENSHOT 4**: Model Training Pipeline Workflow Run
>
> **Lokasi**: Repository → Actions → Model Training Pipeline → Latest run
>
> **Yang ditampilkan**:
> - Workflow run status
> - Steps: Train models, Evaluate, Performance gates passed
> - Model metrics (accuracy, F1 score)
> - Performance gates check (✅ passed)
>
> **Cara ambil**:
> 1. Buka Actions tab
> 2. Klik "Model Training Pipeline"
> 3. Pilih latest successful run
> 4. Screenshot workflow dengan performance gates passed
> 5. Expand step "Check model performance gates" untuk lihat metrics
>
> **File**: `screenshots/04-model-training-pipeline.png`

### **4. Deployment Pipeline** (`.github/workflows/deployment.yml`)

**Triggers**:
- Automatic: Setelah Model Training Pipeline sukses
- Manual: `workflow_dispatch`

**Jobs**:
- Build API Docker image
- Build Dashboard Docker image
- Test containers (health checks, API endpoints)
- Upload Docker images sebagai artifacts

> 📸 **SCREENSHOT 5**: Deployment Pipeline Workflow Run
>
> **Lokasi**: Repository → Actions → Deployment Pipeline → Latest run
>
> **Yang ditampilkan**:
> - Workflow run status
> - Steps: Build API image, Build Dashboard image, Test containers
> - Container test results (health checks passed)
> - Artifacts uploaded
>
> **Cara ambil**:
> 1. Buka Actions tab
> 2. Klik "Deployment Pipeline"
> 3. Pilih latest successful run
> 4. Screenshot workflow dengan Docker build steps
> 5. Expand step "Test containers" untuk lihat test results
>
> **File**: `screenshots/05-deployment-pipeline.png`

> 📸 **SCREENSHOT 6**: Docker Build Logs
>
> **Lokasi**: Repository → Actions → Deployment Pipeline → Latest run → Build API/Dashboard Docker image step
>
> **Yang ditampilkan**:
> - Docker build process
> - Multi-stage build steps
> - Image size information
> - Build completion
>
> **Cara ambil**:
> 1. Buka Deployment Pipeline workflow run
> 2. Expand step "Build API Docker image" atau "Build Dashboard Docker image"
> 3. Scroll untuk lihat build logs
> 4. Screenshot build process dan completion
>
> **File**: `screenshots/06-docker-build-logs.png`

**Container Testing**:
```bash
# Start services
docker-compose up -d sentiment-api sentiment-dashboard
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

## **DOCKER CONFIGURATION**

### **Docker Compose** (`docker/docker-compose.yml`)

**Services**:

1. **sentiment-api**
   - Port: `8000:8000`
   - Health: `http://localhost:8000/health`
   - Volumes: `logs`, `src`, `data`
   - GPU: Optional (NVIDIA)

2. **dashboard**
   - Port: `8501:7860`
   - Health: `http://localhost:7860/_stcore/health`
   - Depends on: `sentiment-api`

3. **prometheus**
   - Port: `9090:9090`
   - Retention: 200 hours

4. **grafana**
   - Port: `3000:3000`
   - Default: `admin` / `admin123`

**Networks**: `sentiment-network` (bridge)

**Volumes**: `prometheus_data`, `grafana_data`

### **Dockerfile API** (`docker/Dockerfile`)

**Multi-stage build**:
- **Stage 1 (Builder)**: Install semua dependencies
- **Stage 2 (Runtime)**: Copy packages, minimal runtime

**Key Features**:
- Optimized dependency installation
- Pinned versions untuk reproducibility
- Health check configured
- Non-root user (optional)

### **Dockerfile Dashboard** (`docker/Dockerfile.dashboard`)

**Single-stage build**:
- Install dari `requirements-dashboard.txt`
- Copy models & data
- Streamlit configuration

---

## **DEPLOYMENT PIPELINE**

### **Alur Lengkap**

```
1. Code Push → CI Pipeline (must pass)
2. Model Training (Schedule/Manual)
   ├── Data Collection (if needed)
   ├── Train Models
   ├── Evaluate
   └── Performance Gates
3. Deployment Pipeline (auto-triggered)
   ├── Build Docker Images
   ├── Test Containers
   └── Upload Artifacts
```

### **Deployment Scenarios**

**Automated** (Recommended):
- Training sukses → Deployment otomatis trigger
- Images built & tested
- Artifacts tersedia untuk download

**Manual**:
- GitHub Actions → Deployment Pipeline → Run workflow
- Options: Build Docker, Test containers

### **Artifact Management**

**Location**: GitHub Actions → Artifacts
**Retention**: 7 days
**Contents**: `api-image.tar.gz`, `dashboard-image.tar.gz`

> 📸 **SCREENSHOT 7**: GitHub Actions Artifacts
>
> **Lokasi**: Repository → Actions → Deployment Pipeline → Latest run → Artifacts section
>
> **Yang ditampilkan**:
> - Artifacts list: `docker-images`
> - File: `api-image.tar.gz`, `dashboard-image.tar.gz`
> - File sizes
> - Download button
>
> **Cara ambil**:
> 1. Buka Deployment Pipeline workflow run
> 2. Scroll ke bagian "Artifacts"
> 3. Screenshot artifacts list dengan file sizes
>
> **File**: `screenshots/07-github-actions-artifacts.png`

**Download & Load**:
```bash
# Download dari GitHub
gunzip api-image.tar.gz
gunzip dashboard-image.tar.gz

# Load ke Docker
docker load < api-image.tar
docker load < dashboard-image.tar
```

---

## **SETUP & CONFIGURATION**

### **GitHub Secrets**

**Required**:
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
- `KAGGLE_USERNAME`, `KAGGLE_KEY`

**Setup**: Repository → Settings → Secrets and variables → Actions → New repository secret

> 📸 **SCREENSHOT 8**: GitHub Secrets Configuration
>
> **Lokasi**: Repository → Settings → Secrets and variables → Actions
>
> **Yang ditampilkan**:
> - Secrets list (dengan names, tanpa values)
> - Required secrets: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, KAGGLE_USERNAME, KAGGLE_KEY
> - "New repository secret" button
>
> **Cara ambil**:
> 1. Buka repository Settings
> 2. Klik "Secrets and variables" → "Actions"
> 3. Screenshot secrets list (pastikan values tidak terlihat)
>
> **File**: `screenshots/08-github-secrets.png`

### **Local Docker Setup**

```bash
# 1. Clone repository
git clone https://github.com/habstrakT808/movie-sentiment-mlops.git
cd movie-sentiment-mlops

# 2. Build & Start
cd docker
docker-compose build
docker-compose up -d

# 3. Check services
docker-compose ps
docker-compose logs -f sentiment-api
```

> 📸 **SCREENSHOT 9**: Docker Compose Services Running
>
> **Lokasi**: Terminal setelah menjalankan `docker-compose ps`
>
> **Yang ditampilkan**:
> - List semua services: sentiment-api, dashboard, prometheus, grafana
> - Status: Up (running)
> - Ports mapping
> - Health status
>
> **Cara ambil**:
> 1. Jalankan `docker-compose ps` di terminal
> 2. Screenshot output dengan semua services running
>
> **File**: `screenshots/09-docker-compose-services.png`

> 📸 **SCREENSHOT 10**: Docker Images List
>
> **Lokasi**: Terminal setelah menjalankan `docker images`
>
> **Yang ditampilkan**:
> - Docker images: movie-sentiment-api, movie-sentiment-dashboard
> - Image tags: latest
> - Image sizes
> - Created dates
>
> **Cara ambil**:
> 1. Jalankan `docker images | grep movie-sentiment`
> 2. Screenshot output dengan images list
>
> **File**: `screenshots/10-docker-images.png`

### **Environment Variables**

**Optional `.env` file**:
```env
LOG_LEVEL=INFO
API_PORT=8000
RETRAIN_THRESHOLD=10
MIN_IMPROVEMENT=0.01
GF_SECURITY_ADMIN_PASSWORD=admin123
```

---

## **TESTING & VALIDATION**

### **CI Pipeline Tests**

```bash
# Code Quality
flake8 src/ --count --select=E9,F63,F7,F82
black --check src/ scripts/
isort --check-only --profile black src/ scripts/

# Unit Tests
pytest tests/unit/ -v
pytest tests/unit/ --cov=src --cov-report=html
```

### **Docker Container Tests**

```bash
# Start services
docker-compose up -d

# Health checks
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'

# Check logs
docker-compose logs sentiment-api
```

> 📸 **SCREENSHOT 11**: API Health Check Response
>
> **Lokasi**: Terminal setelah menjalankan `curl http://localhost:8000/health`
>
> **Yang ditampilkan**:
> - HTTP response: `{"status":"healthy"}`
> - Status code: 200 OK
>
> **Cara ambil**:
> 1. Jalankan `curl http://localhost:8000/health`
> 2. Screenshot terminal dengan response
>
> **File**: `screenshots/11-api-health-check.png`

> 📸 **SCREENSHOT 12**: API Prediction Test
>
> **Lokasi**: Terminal setelah menjalankan prediction curl command
>
> **Yang ditampilkan**:
> - Request: POST dengan JSON body
> - Response: Prediction result dengan sentiment dan confidence
> - Status code: 200 OK
>
> **Cara ambil**:
> 1. Jalankan prediction curl command
> 2. Screenshot terminal dengan request dan response
>
> **File**: `screenshots/12-api-prediction-test.png`

> 📸 **SCREENSHOT 13**: Dashboard Browser View
>
> **Lokasi**: Browser di `http://localhost:8501`
>
> **Yang ditampilkan**:
> - Streamlit dashboard homepage
> - Navigation menu
> - Main content area
>
> **Cara ambil**:
> 1. Buka browser ke `http://localhost:8501`
> 2. Screenshot dashboard homepage
>
> **File**: `screenshots/13-dashboard-browser.png`

> 📸 **SCREENSHOT 14**: Docker Container Logs
>
> **Lokasi**: Terminal setelah menjalankan `docker-compose logs sentiment-api`
>
> **Yang ditampilkan**:
> - Container logs
> - API startup messages
> - Health check logs
> - No errors
>
> **Cara ambil**:
> 1. Jalankan `docker-compose logs sentiment-api | tail -20`
> 2. Screenshot terminal dengan logs
>
> **File**: `screenshots/14-docker-logs.png`

---

## **TROUBLESHOOTING**

### **GitHub Actions Issues**

**Workflow Timeout**:
- Increase `timeout-minutes`
- Optimize dependencies (minimal requirements)
- Use caching

**Docker Build Fails**:
- Check Dockerfile syntax
- Verify context path
- Review build logs

**Container Test Fails**:
- Increase wait time (`sleep 60`)
- Check service logs
- Verify ports available

### **Docker Issues**

**Container Won't Start**:
```bash
docker-compose logs sentiment-api
docker-compose ps
docker-compose restart sentiment-api
```

**Port Already in Use**:
```bash
# Find process
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Change port in docker-compose.yml
```

**Volume Mount Errors**:
```bash
sudo chown -R $USER:$USER data/ logs/
```

### **Common Errors**

**ModuleNotFoundError**: Install dependencies atau check PYTHONPATH
**Model Not Found**: Ensure models copied in Dockerfile
**DVC Remote Not Configured**: Skip DVC pull (`dvc pull || echo "..."`)

---

## **BEST PRACTICES**

### **GitHub Actions**
✅ Use caching, set timeouts, minimal requirements
❌ Don't commit secrets, skip security checks

### **Docker**
✅ Multi-stage builds, pin versions, health checks
❌ Don't use `latest` tags, store secrets, run as root

### **Docker Compose**
✅ Environment variables, health checks, named volumes
❌ Don't expose unnecessary ports, use default passwords

### **CI/CD Pipeline**
✅ Run tests before deploy, use performance gates
❌ Don't deploy on failures, skip security checks

---

## **MONITORING**

### **GitHub Actions**
- View workflow status di GitHub Actions tab
- Check logs untuk errors
- Monitor success rate & run time

### **Docker**
```bash
# Container status
docker-compose ps

# Logs
docker-compose logs -f

# Resource usage
docker stats
```

### **Prometheus & Grafana**
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin123)
- 17-panel comprehensive dashboard

> 📸 **SCREENSHOT 15**: Prometheus Metrics Page
>
> **Lokasi**: Browser di `http://localhost:9090`
>
> **Yang ditampilkan**:
> - Prometheus web UI
> - Metrics list (sentiment_api_*)
> - Graph atau table view
>
> **Cara ambil**:
> 1. Buka browser ke `http://localhost:9090`
> 2. Klik "Graph" atau "Status" → "Targets"
> 3. Screenshot Prometheus UI dengan metrics
>
> **File**: `screenshots/15-prometheus-metrics.png`

> 📸 **SCREENSHOT 16**: Grafana Dashboard
>
> **Lokasi**: Browser di `http://localhost:3000` (login: admin/admin123)
>
> **Yang ditampilkan**:
> - Grafana dashboard: "Model Monitoring Dashboard"
> - 17 panels dengan metrics:
>   - Prediction metrics (accuracy, F1, confusion matrix)
>   - API performance (requests, latency, errors)
>   - Data drift detection
>   - System metrics
>
> **Cara ambil**:
> 1. Login ke Grafana (`http://localhost:3000`)
> 2. Buka dashboard "Model Monitoring Dashboard"
> 3. Screenshot full dashboard dengan semua panels
>
> **File**: `screenshots/16-grafana-dashboard.png`

> 📸 **SCREENSHOT 17**: Docker Stats (Resource Usage)
>
> **Lokasi**: Terminal setelah menjalankan `docker stats`
>
> **Yang ditampilkan**:
> - Container names: movie-sentiment-api, movie-sentiment-dashboard, dll
> - CPU usage (%)
> - Memory usage (MB/GB)
> - Network I/O
>
> **Cara ambil**:
> 1. Jalankan `docker stats --no-stream`
> 2. Screenshot terminal dengan resource usage
>
> **File**: `screenshots/17-docker-stats.png`

---

## **DEPLOYMENT CHECKLIST**

### **Pre-Deployment**
- [ ] All tests passing
- [ ] Code quality checks passing
- [ ] Models trained & evaluated
- [ ] Performance gates met
- [ ] Docker images built
- [ ] Container tests passing
- [ ] Secrets configured

### **Deployment**
- [ ] Download Docker images
- [ ] Load images into Docker
- [ ] Start services
- [ ] Verify health checks
- [ ] Test API endpoints
- [ ] Test Dashboard
- [ ] Check monitoring

### **Post-Deployment**
- [ ] Monitor logs
- [ ] Check metrics
- [ ] Verify predictions
- [ ] Test continuous learning
- [ ] Document issues

---

## **KESIMPULAN**

Integrasi GitHub Actions dan Docker menyediakan:

✅ **Automated CI/CD Pipeline** lengkap
✅ **Containerization** untuk konsistensi
✅ **Quality Assurance** dengan automated testing
✅ **Model Training Automation** dengan performance gates
✅ **Deployment Automation** dengan Docker build & test

**Key Benefits**:
- Reproducibility (Docker)
- Automation (GitHub Actions)
- Quality (Automated tests)
- Efficiency (Caching)
- Reliability (Health checks)

**Next Steps**:
1. Deploy to production (VPS/Cloud)
2. Setup SSL/HTTPS
3. Configure domain
4. Setup monitoring alerts
5. Implement backup strategy

---

---

## **DAFTAR SCREENSHOT**

Berikut adalah daftar lengkap screenshot yang diperlukan sebagai bukti implementasi:

| No | Screenshot | File | Lokasi | Keterangan |
|----|-----------|------|--------|------------|
| 1 | GitHub Actions Workflows List | `01-github-actions-workflows-list.png` | Repository → Actions | List semua 4 workflows |
| 2 | CI Pipeline Success | `02-ci-pipeline-success.png` | Actions → CI Pipeline | Workflow run dengan semua jobs passed |
| 3 | Data Collection Pipeline | `03-data-collection-pipeline.png` | Actions → Data Collection | Workflow run dengan data collection steps |
| 4 | Model Training Pipeline | `04-model-training-pipeline.png` | Actions → Model Training | Performance gates passed dengan metrics |
| 5 | Deployment Pipeline | `05-deployment-pipeline.png` | Actions → Deployment | Docker build dan test steps |
| 6 | Docker Build Logs | `06-docker-build-logs.png` | Actions → Deployment → Build step | Build process dan completion |
| 7 | GitHub Actions Artifacts | `07-github-actions-artifacts.png` | Actions → Deployment → Artifacts | Docker images artifacts |
| 8 | GitHub Secrets | `08-github-secrets.png` | Settings → Secrets | Secrets configuration |
| 9 | Docker Compose Services | `09-docker-compose-services.png` | Terminal | `docker-compose ps` output |
| 10 | Docker Images | `10-docker-images.png` | Terminal | `docker images` output |
| 11 | API Health Check | `11-api-health-check.png` | Terminal | `curl /health` response |
| 12 | API Prediction Test | `12-api-prediction-test.png` | Terminal | Prediction curl test |
| 13 | Dashboard Browser | `13-dashboard-browser.png` | Browser | Streamlit dashboard |
| 14 | Docker Logs | `14-docker-logs.png` | Terminal | Container logs |
| 15 | Prometheus Metrics | `15-prometheus-metrics.png` | Browser | Prometheus UI |
| 16 | Grafana Dashboard | `16-grafana-dashboard.png` | Browser | Grafana monitoring dashboard |
| 17 | Docker Stats | `17-docker-stats.png` | Terminal | Resource usage |

### **Struktur Folder Screenshot**

```
docs/
├── GITHUB_ACTIONS_DOCKER_INTEGRATION.md
└── screenshots/
    ├── 01-github-actions-workflows-list.png
    ├── 02-ci-pipeline-success.png
    ├── 03-data-collection-pipeline.png
    ├── 04-model-training-pipeline.png
    ├── 05-deployment-pipeline.png
    ├── 06-docker-build-logs.png
    ├── 07-github-actions-artifacts.png
    ├── 08-github-secrets.png
    ├── 09-docker-compose-services.png
    ├── 10-docker-images.png
    ├── 11-api-health-check.png
    ├── 12-api-prediction-test.png
    ├── 13-dashboard-browser.png
    ├── 14-docker-logs.png
    ├── 15-prometheus-metrics.png
    ├── 16-grafana-dashboard.png
    └── 17-docker-stats.png
```

### **Cara Menambahkan Screenshot ke Dokumentasi**

Setelah mengambil screenshot, simpan di folder `docs/screenshots/` dengan nama sesuai daftar di atas. Screenshot akan otomatis terlihat di dokumentasi karena sudah ada placeholder dengan format:

```markdown
> 📸 **SCREENSHOT X**: Description
>
> **File**: `screenshots/XX-description.png`
```

Untuk menampilkan screenshot di markdown, tambahkan:

```markdown
![Description](screenshots/XX-description.png)
```

**Document Version**: 1.2 (Dengan Screenshot Placeholders)
**Last Updated**: 14 November 2025
**Author**: Hafiyan Al Muqaffi Umary
