# PowerShell script to fix MLflow database and start UI

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FIXING MLFLOW DATABASE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Remove old database
if (Test-Path "mlflow.db") {
    Remove-Item "mlflow.db" -Force
    Write-Host "[OK] Removed old database" -ForegroundColor Green
}

# Initialize database
Write-Host "`n[INFO] Initializing database..." -ForegroundColor Yellow
python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); mlflow.set_experiment('movie_sentiment_analysis'); print('Database initialized')"

# Re-log models
Write-Host "`n[INFO] Re-logging models..." -ForegroundColor Yellow
python scripts/relog_models_to_mlflow.py

# Start MLflow UI
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STARTING MLFLOW UI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Access at: http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
