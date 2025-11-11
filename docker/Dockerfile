# Multi-stage build for optimized image size
# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip first to improve dependency resolution
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies in stages to avoid resolution conflicts
# The main issue is fsspec version conflict between torch and datasets

# Stage 1: Install fsspec first at the version required by datasets
RUN pip install --no-cache-dir --user fsspec==2024.3.1

# Stage 2: Install core packages
RUN pip install --no-cache-dir --user \
    pandas==2.1.3 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

# Stage 3: Install torch (it may upgrade fsspec, but we'll pin it back)
RUN pip install --no-cache-dir --user torch==2.5.1 && \
    pip install --no-cache-dir --user --force-reinstall fsspec==2024.3.1

# Stage 4: Install datasets and transformers (these need the pinned fsspec)
RUN pip install --no-cache-dir --user \
    pyarrow==14.0.2 \
    datasets==2.19.1 \
    transformers==4.35.2 \
    accelerate==0.25.0

# Stage 5: Install MLOps tools
RUN pip install --no-cache-dir --user mlflow==2.8.1

# Stage 6: Install DVC (complex dependencies, try with fallback)
RUN pip install --no-cache-dir --user dvc==3.63.0 || \
    (pip install --no-cache-dir --user dvc==3.63.0 --no-deps && \
     pip install --no-cache-dir --user attrs colorama configobj distro dpath dulwich dvc-data dvc-http dvc-objects dvc-render dvc-studio-client dvc-task flatten_dict flufl.lock funcy grandalf gto hydra-core iterative-telemetry kombu omegaconf pathspec platformdirs pydot pygtrie pyparsing rich ruamel.yaml scmrepo shortuuid shtab tabulate tomlkit voluptuous zc.lockfile)

# Stage 7: Install DVC S3 extras
RUN pip install --no-cache-dir --user s3fs boto3 aiobotocore || true

# Stage 8: Install remaining packages
RUN pip install --no-cache-dir --user \
    fastapi==0.104.1 \
    'uvicorn[standard]==0.24.0' \
    streamlit==1.28.2 \
    pydantic==2.12.3 \
    python-multipart==0.0.6 \
    prometheus-client==0.19.0 \
    psutil==5.9.6 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.18.0 \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    loguru==0.7.2 \
    tqdm==4.66.4 \
    click==8.1.7 \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    pytest-asyncio==0.21.1 \
    pytest-mock==3.12.0 \
    httpx==0.25.2 \
    black==23.12.0 \
    flake8==6.1.0 \
    isort==5.13.2 \
    mypy==1.7.1 \
    pre-commit==3.6.0 \
    mkdocs==1.5.3 \
    mkdocs-material==9.5.3 \
    jupyter==1.0.0 \
    ipykernel==6.27.1 \
    ipywidgets==8.1.1 \
    praw==7.7.1 \
    kaggle==1.5.16 \
    beautifulsoup4==4.12.2 \
    'requests>=2.32.3' \
    lxml==4.9.3 \
    nltk==3.8.1 \
    spacy==3.7.2 \
    textblob==0.17.1 \
    wordcloud==1.9.3 \
    imbalanced-learn==0.11.0 \
    xgboost==2.0.3

# Stage 2: Runtime stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/usr/local/bin:/root/.local/bin:$PATH"
ENV PYTHONPATH="/app:/root/.local/lib/python3.11/site-packages:/usr/local/lib/python3.11/site-packages"

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install packages globally
COPY --from=builder /build/requirements.txt /tmp/requirements.txt

# Install packages globally so they're accessible to all users
# We'll install key packages that are needed for the API to run
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    'uvicorn[standard]==0.24.0' \
    pydantic==2.12.3 \
    python-multipart==0.0.6 \
    prometheus-client==0.19.0 \
    psutil==5.9.6 \
    pyyaml==6.0.1 \
    loguru==0.7.2 \
    python-dotenv==1.0.0

# Copy Python packages from builder stage for other dependencies
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY params.yaml ./
COPY setup.py ./
COPY readme.md ./

# Copy models (this will be the largest part)
COPY models/distilbert/ ./models/distilbert/

# Create necessary directories
RUN mkdir -p logs data/processed

# Install the package (as root, before switching user)
RUN pip install -e .

# Change ownership to app user (optional, for file permissions)
RUN chown -R appuser:appuser /app

# Note: Running as root for now to ensure all packages are accessible
# For production, consider copying packages to appuser's home directory
# USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
