#!/usr/bin/env python
"""Script to start MLflow UI with proper database initialization."""

import os
import subprocess
import sys

import mlflow

# Ensure database is clean and initialized
db_path = "mlflow.db"
if os.path.exists(db_path):
    # Remove old database
    os.remove(db_path)
    print(f"Removed old database: {db_path}")

# Initialize database first
print("Initializing MLflow database...")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("movie_sentiment_analysis")
print("Database initialized successfully!")

# Start MLflow UI
print("\nStarting MLflow UI...")
print("Access at: http://localhost:5000")
print("Press Ctrl+C to stop\n")

try:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--port",
            "5000",
        ]
    )
except KeyboardInterrupt:
    print("\nMLflow UI stopped.")
