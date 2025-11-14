# Docker Disk Cleanup Script for Windows PowerShell
# Helps analyze and clean up Docker disk usage

Write-Host "🐳 Docker Disk Usage Analysis & Cleanup" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "[ERROR] Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

Write-Host "📊 Current Docker Disk Usage:" -ForegroundColor Blue
Write-Host "----------------------------" -ForegroundColor Blue
docker system df
Write-Host ""

# Show detailed breakdown
Write-Host "📦 Detailed Breakdown:" -ForegroundColor Blue
Write-Host "---------------------" -ForegroundColor Blue
Write-Host ""

Write-Host "Images:" -ForegroundColor Yellow
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | Select-Object -First 20
Write-Host ""

Write-Host "Containers (all):" -ForegroundColor Yellow
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Size}}" | Select-Object -First 20
Write-Host ""

Write-Host "Volumes:" -ForegroundColor Yellow
docker volume ls
Write-Host ""

# Ask user what to clean
Write-Host "🧹 Cleanup Options:" -ForegroundColor Cyan
Write-Host "-------------------" -ForegroundColor Cyan
Write-Host "1. Remove unused images (dangling images)"
Write-Host "2. Remove stopped containers"
Write-Host "3. Remove unused volumes"
Write-Host "4. Remove build cache"
Write-Host "5. Remove all unused data (images, containers, networks, volumes, build cache)"
Write-Host "6. Prune everything (⚠️  DANGEROUS - removes all unused resources)"
Write-Host "7. Exit"
Write-Host ""

$choice = Read-Host "Select option (1-7)"

switch ($choice) {
    "1" {
        Write-Host "[INFO] Removing dangling images..." -ForegroundColor Blue
        docker image prune -f
        Write-Host "[SUCCESS] Dangling images removed" -ForegroundColor Green
    }
    "2" {
        Write-Host "[INFO] Removing stopped containers..." -ForegroundColor Blue
        docker container prune -f
        Write-Host "[SUCCESS] Stopped containers removed" -ForegroundColor Green
    }
    "3" {
        Write-Host "[WARNING] Removing unused volumes (this may delete data)..." -ForegroundColor Yellow
        $confirm = Read-Host "Are you sure? (y/N)"
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            docker volume prune -f
            Write-Host "[SUCCESS] Unused volumes removed" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Cancelled" -ForegroundColor Blue
        }
    }
    "4" {
        Write-Host "[INFO] Removing build cache..." -ForegroundColor Blue
        docker builder prune -f
        Write-Host "[SUCCESS] Build cache removed" -ForegroundColor Green
    }
    "5" {
        Write-Host "[WARNING] Removing all unused data..." -ForegroundColor Yellow
        $confirm = Read-Host "Are you sure? This will remove unused images, containers, networks, and volumes (y/N)"
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            docker system prune -a --volumes -f
            Write-Host "[SUCCESS] All unused data removed" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Cancelled" -ForegroundColor Blue
        }
    }
    "6" {
        Write-Host "[ERROR] ⚠️  DANGER: This will remove ALL unused resources including images!" -ForegroundColor Red
        $confirm = Read-Host "Are you absolutely sure? Type 'yes' to confirm"
        if ($confirm -eq "yes") {
            docker system prune -a --volumes -f
            Write-Host "[SUCCESS] Everything pruned" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Cancelled" -ForegroundColor Blue
        }
    }
    "7" {
        Write-Host "[INFO] Exiting..." -ForegroundColor Blue
        exit 0
    }
    default {
        Write-Host "[ERROR] Invalid option" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "📊 Updated Docker Disk Usage:" -ForegroundColor Blue
Write-Host "----------------------------" -ForegroundColor Blue
docker system df
Write-Host ""

Write-Host "[SUCCESS] Cleanup complete!" -ForegroundColor Green
