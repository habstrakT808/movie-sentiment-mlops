#!/bin/bash

# Docker Disk Cleanup Script
# Helps analyze and clean up Docker disk usage

echo "🐳 Docker Disk Usage Analysis & Cleanup"
echo "======================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo "📊 Current Docker Disk Usage:"
echo "----------------------------"
docker system df
echo ""

# Show detailed breakdown
echo "📦 Detailed Breakdown:"
echo "---------------------"
echo ""
echo "Images:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | head -20
echo ""

echo "Containers (all):"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Size}}" | head -20
echo ""

echo "Volumes:"
docker volume ls
echo ""

# Calculate total size
TOTAL_SIZE=$(docker system df --format "{{.Size}}" | head -1)
print_info "Total Docker disk usage: $TOTAL_SIZE"
echo ""

# Ask user what to clean
echo "🧹 Cleanup Options:"
echo "-------------------"
echo "1. Remove unused images (dangling images)"
echo "2. Remove stopped containers"
echo "3. Remove unused volumes"
echo "4. Remove build cache"
echo "5. Remove all unused data (images, containers, networks, volumes, build cache)"
echo "6. Prune everything (⚠️  DANGEROUS - removes all unused resources)"
echo "7. Exit"
echo ""

read -p "Select option (1-7): " choice

case $choice in
    1)
        print_info "Removing dangling images..."
        docker image prune -f
        print_success "Dangling images removed"
        ;;
    2)
        print_info "Removing stopped containers..."
        docker container prune -f
        print_success "Stopped containers removed"
        ;;
    3)
        print_warning "Removing unused volumes (this may delete data)..."
        read -p "Are you sure? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            docker volume prune -f
            print_success "Unused volumes removed"
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
        print_warning "Removing all unused data..."
        read -p "Are you sure? This will remove unused images, containers, networks, and volumes (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            docker system prune -a --volumes -f
            print_success "All unused data removed"
        else
            print_info "Cancelled"
        fi
        ;;
    6)
        print_error "⚠️  DANGER: This will remove ALL unused resources including images!"
        read -p "Are you absolutely sure? Type 'yes' to confirm: " confirm
        if [[ $confirm == "yes" ]]; then
            docker system prune -a --volumes -f
            print_success "Everything pruned"
        else
            print_info "Cancelled"
        fi
        ;;
    7)
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

print_success "Cleanup complete!"
