#!/bin/bash

# DVC Helper Scripts for common operations

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check DVC status
check_dvc_status() {
    echo -e "${YELLOW}Checking DVC status...${NC}"
    dvc status
}

# Function to add data to DVC
add_to_dvc() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please provide a file or directory to add${NC}"
        echo "Usage: add_to_dvc <path>"
        return 1
    fi

    echo -e "${YELLOW}Adding $1 to DVC...${NC}"
    dvc add "$1"

    # Add .dvc file to git
    git add "$1.dvc" .gitignore
    echo -e "${GREEN}Added $1 to DVC${NC}"
}

# Function to push data to DVC remote
push_to_dvc() {
    echo -e "${YELLOW}Pushing data to DVC remote...${NC}"
    dvc push
    echo -e "${GREEN}Data pushed successfully${NC}"
}

# Function to pull data from DVC remote
pull_from_dvc() {
    echo -e "${YELLOW}Pulling data from DVC remote...${NC}"
    dvc pull
    echo -e "${GREEN}Data pulled successfully${NC}"
}

# Function to reproduce DVC pipeline
reproduce_pipeline() {
    echo -e "${YELLOW}Reproducing DVC pipeline...${NC}"
    dvc repro
    echo -e "${GREEN}Pipeline reproduced successfully${NC}"
}

# Function to show DVC pipeline DAG
show_pipeline() {
    echo -e "${YELLOW}DVC Pipeline DAG:${NC}"
    dvc dag
}

# Function to check data integrity
check_integrity() {
    echo -e "${YELLOW}Checking data integrity...${NC}"
    dvc status --cloud
}

# Make functions available when script is sourced
if [ "$1" == "status" ]; then
    check_dvc_status
elif [ "$1" == "add" ]; then
    add_to_dvc "$2"
elif [ "$1" == "push" ]; then
    push_to_dvc
elif [ "$1" == "pull" ]; then
    pull_from_dvc
elif [ "$1" == "repro" ]; then
    reproduce_pipeline
elif [ "$1" == "dag" ]; then
    show_pipeline
elif [ "$1" == "check" ]; then
    check_integrity
else
    echo "DVC Helper Script"
    echo "Usage: ./scripts/dvc_helpers.sh [command]"
    echo ""
    echo "Commands:"
    echo "  status  - Check DVC status"
    echo "  add     - Add file to DVC (requires path)"
    echo "  push    - Push data to DVC remote"
    echo "  pull    - Pull data from DVC remote"
    echo "  repro   - Reproduce DVC pipeline"
    echo "  dag     - Show pipeline DAG"
    echo "  check   - Check data integrity"
fi
