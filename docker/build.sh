#!/bin/bash

echo "🐳 Building Movie Sentiment Analysis Docker Image..."

# Build the Docker image
docker build -f docker/Dockerfile -t movie-sentiment-api:latest .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo "📊 Image size:"
    docker images movie-sentiment-api:latest
else
    echo "❌ Docker build failed!"
    exit 1
fi
