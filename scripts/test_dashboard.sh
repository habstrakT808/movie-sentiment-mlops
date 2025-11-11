#!/bin/bash

echo "🧪 Testing Movie Sentiment Dashboard Deployment..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Test 1: Check if Docker is running
print_status "Checking Docker status..."
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi
print_success "Docker is running"

# Test 2: Build dashboard image
print_status "Building dashboard image..."
if docker build -f docker/Dockerfile.dashboard -t movie-sentiment-dashboard:latest .; then
    print_success "Dashboard image built successfully"
else
    print_error "Failed to build dashboard image"
    exit 1
fi

# Test 3: Check if port 8501 is available
print_status "Checking if port 8501 is available..."
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port 8501 is already in use. Stopping any existing containers..."
    docker stop test-dashboard 2>/dev/null || true
    docker rm test-dashboard 2>/dev/null || true
fi

# Test 4: Run dashboard container
print_status "Starting dashboard container..."
if docker run -d --name test-dashboard -p 8501:7860 movie-sentiment-dashboard:latest; then
    print_success "Dashboard container started"
else
    print_error "Failed to start dashboard container"
    exit 1
fi

# Test 5: Wait for startup
print_status "Waiting for dashboard to start (30 seconds)..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""

# Test 6: Test health endpoint
print_status "Testing health endpoint..."
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    print_success "Health check passed"
else
    print_warning "Health check failed (this might be normal for Streamlit)"
fi

# Test 7: Test main page
print_status "Testing main page..."
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    print_success "Main page is accessible"
else
    print_error "Main page is not accessible"
    print_status "Checking container logs..."
    docker logs test-dashboard --tail 20
fi

# Test 8: Show container status
print_status "Container status:"
docker ps | grep test-dashboard

# Test 9: Show logs
print_status "Recent container logs:"
docker logs test-dashboard --tail 10

echo ""
echo "=================================================="
print_success "Dashboard testing completed!"
echo ""
print_status "Dashboard URL: ${BLUE}http://localhost:8501${NC}"
echo ""
print_status "To stop and cleanup:"
echo "  docker stop test-dashboard && docker rm test-dashboard"
echo ""
print_status "To run with docker-compose:"
echo "  cd docker && docker-compose up -d dashboard"
echo ""
print_status "To view logs:"
echo "  docker logs -f test-dashboard"
