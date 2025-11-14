# Docker Safe Cleanup Script for Windows PowerShell
# Only removes unused resources, keeps what's needed for the application

Write-Host "🐳 Docker Safe Cleanup - Movie Sentiment Analysis" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "[ERROR] Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Function to check if resource is in use
function Is-ResourceInUse {
    param($resourceName, $resourceType)

    switch ($resourceType) {
        "image" {
            $containers = docker ps -a --filter "ancestor=$resourceName" --format "{{.Names}}"
            return $containers -ne ""
        }
        "volume" {
            $containers = docker ps -a --filter "volume=$resourceName" --format "{{.Names}}"
            return $containers -ne ""
        }
    }
    return $false
}

# Get current project containers
Write-Host "📋 Checking running application containers..." -ForegroundColor Blue
$projectContainers = @(
    "movie-sentiment-api",
    "movie-sentiment-dashboard",
    "movie-sentiment-prometheus",
    "movie-sentiment-grafana"
)

$runningContainers = docker ps --format "{{.Names}}"
$stoppedContainers = docker ps -a --filter "status=exited" --format "{{.Names}}"

Write-Host ""
Write-Host "📊 Current Docker Disk Usage:" -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow
docker system df
Write-Host ""

# Analyze what can be safely removed
Write-Host "🔍 Analyzing unused resources..." -ForegroundColor Blue
Write-Host ""

# 1. Dangling images (safe to remove)
Write-Host "1. Checking dangling images (untagged)..." -ForegroundColor Cyan
$danglingImages = docker images -f "dangling=true" -q
if ($danglingImages) {
    $danglingCount = ($danglingImages | Measure-Object).Count
    $danglingSize = docker images -f "dangling=true" --format "{{.Size}}" | ForEach-Object {
        [int]($_ -replace '[^0-9]', '')
    } | Measure-Object -Sum | Select-Object -ExpandProperty Sum
    Write-Host "   Found $danglingCount dangling images" -ForegroundColor Yellow
} else {
    Write-Host "   No dangling images found" -ForegroundColor Green
}

# 2. Stopped containers (not from this project)
Write-Host "2. Checking stopped containers..." -ForegroundColor Cyan
$allStopped = docker ps -a --filter "status=exited" --format "{{.Names}}"
$projectStopped = $allStopped | Where-Object { $projectContainers -contains $_ }
$otherStopped = $allStopped | Where-Object { $projectContainers -notcontains $_ }
if ($otherStopped) {
    Write-Host "   Found $($otherStopped.Count) stopped containers (not from this project)" -ForegroundColor Yellow
} else {
    Write-Host "   No stopped containers to remove" -ForegroundColor Green
}

# 3. Unused images (not used by any container)
Write-Host "3. Checking unused images..." -ForegroundColor Cyan
$allImages = docker images --format "{{.Repository}}:{{.Tag}}"
$unusedImages = @()
foreach ($image in $allImages) {
    if (-not $image.StartsWith('&lt;none&gt;')) {
        $containers = docker ps -a --filter "ancestor=$image" --format "{{.Names}}"
        if (-not $containers) {
            $unusedImages += $image
        }
    }
}
if ($unusedImages.Count -gt 0) {
    Write-Host "   Found $($unusedImages.Count) unused images" -ForegroundColor Yellow
    Write-Host "   Examples: $($unusedImages[0..2] -join ', ')" -ForegroundColor Gray
} else {
    Write-Host "   No unused images found" -ForegroundColor Green
}

# 4. Build cache
Write-Host "4. Checking build cache..." -ForegroundColor Cyan
$buildCache = docker system df | Select-String "Build Cache"
Write-Host "   $buildCache" -ForegroundColor Yellow

# 5. Unused volumes (not used by project)
Write-Host "5. Checking unused volumes..." -ForegroundColor Cyan
$allVolumes = docker volume ls --format "{{.Name}}"
$projectVolumes = @("prometheus_data", "grafana_data", "docker_prometheus_data", "docker_grafana_data")
$unusedVolumes = @()
foreach ($volume in $allVolumes) {
    $containers = docker ps -a --filter "volume=$volume" --format "{{.Names}}"
    if (-not $containers -and $projectVolumes -notcontains $volume) {
        $unusedVolumes += $volume
    }
}
if ($unusedVolumes.Count -gt 0) {
    Write-Host "   Found $($unusedVolumes.Count) unused volumes" -ForegroundColor Yellow
} else {
    Write-Host "   No unused volumes found" -ForegroundColor Green
}

Write-Host ""
Write-Host "🧹 Safe Cleanup Options:" -ForegroundColor Cyan
Write-Host "----------------------" -ForegroundColor Cyan
Write-Host "1. Remove dangling images only (SAFEST)"
Write-Host "2. Remove dangling images + stopped containers (not from this project)"
Write-Host "3. Remove dangling images + unused images (not used by any container)"
Write-Host "4. Remove build cache only"
Write-Host "5. Remove dangling images + build cache (RECOMMENDED)"
Write-Host "6. Remove dangling images + unused images + build cache"
Write-Host "7. Show what will be removed (dry run)"
Write-Host "8. Exit"
Write-Host ""

$choice = Read-Host "Select option (1-8)"

switch ($choice) {
    "1" {
        Write-Host "[INFO] Removing dangling images only..." -ForegroundColor Blue
        if ($danglingImages) {
            docker image prune -f
            Write-Host "[SUCCESS] Dangling images removed" -ForegroundColor Green
        } else {
            Write-Host "[INFO] No dangling images to remove" -ForegroundColor Blue
        }
    }
    "2" {
        Write-Host "[INFO] Removing dangling images + stopped containers..." -ForegroundColor Blue
        if ($danglingImages) {
            docker image prune -f
        }
        if ($otherStopped) {
            docker container prune -f
            Write-Host "[SUCCESS] Cleanup complete" -ForegroundColor Green
        } else {
            Write-Host "[INFO] No stopped containers to remove" -ForegroundColor Blue
        }
    }
    "3" {
        Write-Host "[INFO] Removing dangling images + unused images..." -ForegroundColor Blue
        Write-Host "[WARNING] This will remove images not used by any container" -ForegroundColor Yellow
        $confirm = Read-Host "Continue? (y/N)"
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            docker image prune -a -f
            Write-Host "[SUCCESS] Unused images removed" -ForegroundColor Green
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
        Write-Host "[INFO] Removing dangling images + build cache (RECOMMENDED)..." -ForegroundColor Blue
        if ($danglingImages) {
            docker image prune -f
        }
        docker builder prune -f
        Write-Host "[SUCCESS] Cleanup complete" -ForegroundColor Green
    }
    "6" {
        Write-Host "[INFO] Removing dangling images + unused images + build cache..." -ForegroundColor Blue
        Write-Host "[WARNING] This will remove images not used by any container" -ForegroundColor Yellow
        $confirm = Read-Host "Continue? (y/N)"
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            docker image prune -a -f
            docker builder prune -f
            Write-Host "[SUCCESS] Cleanup complete" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Cancelled" -ForegroundColor Blue
        }
    }
    "7" {
        Write-Host ""
        Write-Host "📋 Dry Run - What will be removed:" -ForegroundColor Yellow
        Write-Host "----------------------------------" -ForegroundColor Yellow
        if ($danglingImages) {
            Write-Host "Dangling images:" -ForegroundColor Cyan
            docker images -f "dangling=true" --format "  - {{.ID}} ({{.Size}})"
        }
        if ($otherStopped) {
            Write-Host "Stopped containers:" -ForegroundColor Cyan
            $otherStopped | ForEach-Object { Write-Host "  - $_" }
        }
        if ($unusedImages.Count -gt 0) {
            Write-Host "Unused images:" -ForegroundColor Cyan
            $unusedImages[0..9] | ForEach-Object { Write-Host "  - $_" }
            if ($unusedImages.Count -gt 10) {
                Write-Host "  ... and $($unusedImages.Count - 10) more" -ForegroundColor Gray
            }
        }
        if ($unusedVolumes.Count -gt 0) {
            Write-Host "Unused volumes:" -ForegroundColor Cyan
            $unusedVolumes | ForEach-Object { Write-Host "  - $_" }
        }
        Write-Host ""
        Write-Host "Build cache will be removed" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "[INFO] This was a dry run. Nothing was removed." -ForegroundColor Blue
    }
    "8" {
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

# Show what's still running
Write-Host "✅ Protected Resources (Still Running):" -ForegroundColor Green
$running = docker ps --format "{{.Names}}"
if ($running) {
    $running | ForEach-Object { Write-Host "  - $_" -ForegroundColor Green }
} else {
    Write-Host "  No containers running" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[SUCCESS] Safe cleanup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Tip: Run this script regularly to keep Docker disk usage low" -ForegroundColor Cyan
