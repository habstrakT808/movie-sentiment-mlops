#!/bin/bash

# MLflow Server Startup Script

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
MLFLOW_PORT=5000
BACKEND_STORE_URI="sqlite:///mlflow.db"
DEFAULT_ARTIFACT_ROOT="./mlruns"

echo -e "${YELLOW}Starting MLflow Tracking Server...${NC}"

# Check if MLflow is installed
if ! command -v mlflow &> /dev/null; then
    echo -e "${RED}Error: MLflow is not installed${NC}"
    echo "Install with: pip install mlflow"
    exit 1
fi

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Check if port is already in use
if lsof -Pi :$MLFLOW_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port $MLFLOW_PORT is already in use${NC}"
    echo "Kill existing process with: lsof -ti:$MLFLOW_PORT | xargs kill -9"
    exit 1
fi

# Start MLflow server
echo -e "${GREEN}Starting MLflow UI on http://localhost:$MLFLOW_PORT${NC}"
echo -e "${YELLOW}Backend Store: $BACKEND_STORE_URI${NC}"
echo -e "${YELLOW}Artifact Store: $DEFAULT_ARTIFACT_ROOT${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop the server${NC}"
echo ""

mlflow server \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$DEFAULT_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port $MLFLOW_PORT
