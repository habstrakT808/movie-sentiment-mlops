#!/bin/bash

# Docker Safe Cleanup Script
# Only removes unused resources, keeps what's needed for the application

echo "🐳 Docker Safe Cleanup - Movie Sentiment Analysis"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Get current project containers
print_info "Checking running application containers..."
PROJECT_CONTAINERS=(
    "movie-sentiment-api"
    "movie-sentiment-dashboard"
    "movie-sentiment-prometheus"
    "movie-sentiment-grafana"
)

echo ""
echo "📊 Current Docker Disk Usage:"
echo "----------------------------"
docker system df
echo ""

# Analyze what can be safely removed
print_info "Analyzing unused resources..."
echo ""

# 1. Dangling images
echo -e "${CYAN}1. Checking dangling images (untagged)...${NC}"
DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
if [ -n "$DANGLING_IMAGES" ]; then
    DANGLING_COUNT=$(echo "$DANGLING_IMAGES" | wc -l)
    echo -e "${YELLOW}   Found $DANGLING_COUNT dangling images${NC}"
else
    echo -e "${GREEN}   No dangling images found${NC}"
fi

# 2. Stopped containers
echo -e "${CYAN}2. Checking stopped containers...${NC}"
ALL_STOPPED=$(docker ps -a --filter "status=exited" --format "{{.Names}}")
OTHER_STOPPED=""
for container in $ALL_STOPPED; do
    is_project=false
    for project_container in "${PROJECT_CONTAINERS[@]}"; do
        if [ "$container" = "$project_container" ]; then
            is_project=true
            break
        fi
    done
    if [ "$is_project" = false ]; then
        OTHER_STOPPED="$OTHER_STOPPED $container"
    fi
done

if [ -n "$OTHER_STOPPED" ]; then
    OTHER_COUNT=$(echo "$OTHER_STOPPED" | wc -w)
    echo -e "${YELLOW}   Found $OTHER_COUNT stopped containers (not from this project)${NC}"
else
    echo -e "${GREEN}   No stopped containers to remove${NC}"
fi

# 3. Build cache
echo -e "${CYAN}3. Checking build cache...${NC}"
BUILD_CACHE=$(docker system df | grep "Build Cache")
echo -e "${YELLOW}   $BUILD_CACHE${NC}"

echo ""
echo "🧹 Safe Cleanup Options:"
echo "----------------------"
echo "1. Remove dangling images only (SAFEST)"
echo "2. Remove dangling images + stopped containers (not from this project)"
echo "3. Remove dangling images + unused images (not used by any container)"
echo "4. Remove build cache only"
echo "5. Remove dangling images + build cache (RECOMMENDED)"
echo "6. Remove dangling images + unused images + build cache"
echo "7. Show what will be removed (dry run)"
echo "8. Exit"
echo ""

read -p "Select option (1-8): " choice

case $choice in
    1)
        print_info "Removing dangling images only..."
        if [ -n "$DANGLING_IMAGES" ]; then
            docker image prune -f
            print_success "Dangling images removed"
        else
            print_info "No dangling images to remove"
        fi
        ;;
    2)
        print_info "Removing dangling images + stopped containers..."
        if [ -n "$DANGLING_IMAGES" ]; then
            docker image prune -f
        fi
        if [ -n "$OTHER_STOPPED" ]; then
            docker container prune -f
            print_success "Cleanup complete"
        else
            print_info "No stopped containers to remove"
        fi
        ;;
    3)
        print_warning "Removing dangling images + unused images..."
        print_warning "This will remove images not used by any container"
        read -p "Continue? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            docker image prune -a -f
            print_success "Unused images removed"
        else
            print_info "Cancelled"
        fi
        ;;
    4)
        print_info "Removing build cache..."
        docker builder prune -f
        print_success "Build cache removed"
        ;;
    5)
        print_info "Removing dangling images + build cache (RECOMMENDED)..."
        if [ -n "$DANGLING_IMAGES" ]; then
            docker image prune -f
        fi
        docker builder prune -f
        print_success "Cleanup complete"
        ;;
    6)
        print_warning "Removing dangling images + unused images + build cache..."
        print_warning "This will remove images not used by any container"
        read -p "Continue? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            docker image prune -a -f
            docker builder prune -f
            print_success "Cleanup complete"
        else
            print_info "Cancelled"
        fi
        ;;
    7)
        echo ""
        echo "📋 Dry Run - What will be removed:"
        echo "----------------------------------"
        if [ -n "$DANGLING_IMAGES" ]; then
            echo -e "${CYAN}Dangling images:${NC}"
            docker images -f "dangling=true" --format "  - {{.ID}} ({{.Size}})"
        fi
        if [ -n "$OTHER_STOPPED" ]; then
            echo -e "${CYAN}Stopped containers:${NC}"
            for container in $OTHER_STOPPED; do
                echo "  - $container"
            done
        fi
        echo ""
        echo -e "${CYAN}Build cache will be removed${NC}"
        echo ""
        print_info "This was a dry run. Nothing was removed."
        ;;
    8)
        print_info "Exiting..."
        exit 0
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "📊 Updated Docker Disk Usage:"
echo "----------------------------"
docker system df
echo ""

# Show what's still running
echo "✅ Protected Resources (Still Running):"
RUNNING=$(docker ps --format "{{.Names}}")
if [ -n "$RUNNING" ]; then
    echo "$RUNNING" | while read -r container; do
        echo -e "${GREEN}  - $container${NC}"
    done
else
    echo -e "${GREEN}  No containers running${NC}"
fi

echo ""
print_success "Safe cleanup complete!"
echo ""
echo "💡 Tip: Run this script regularly to keep Docker disk usage low"
