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

### **4. Deployment Pipeline** (`.github/workflows/deployment.yml`)

**Triggers**:
- Automatic: Setelah Model Training Pipeline sukses
- Manual: `workflow_dispatch`

**Jobs**:
- Build API Docker image
- Build Dashboard Docker image
- Test containers (health checks, API endpoints)
- Upload Docker images sebagai artifacts

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

### **Local Docker Setup**

```bash
# 1. Clone repository
git clone https://github.com/your-username/movie-sentiment-mlops.git
cd movie-sentiment-mlops

# 2. Build & Start
cd docker
docker-compose build
docker-compose up -d

# 3. Check services
docker-compose ps
docker-compose logs -f sentiment-api
```

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

**Document Version**: 1.1 (Ringkas)
**Last Updated**: 14 November 2025
**Author**: Hafiyan Al Muqaffi Umary
