#!/bin/bash

echo "🚀 Starting Movie Sentiment Analysis API..."

# Stop existing container if running
docker stop movie-sentiment-api 2>/dev/null
docker rm movie-sentiment-api 2>/dev/null

# Run the container
docker run -d \
    --name movie-sentiment-api \
    -p 8000:8000 \
    -v $(pwd)/logs:/app/logs \
    movie-sentiment-api:latest

# Check if container is running
sleep 5
if [ "$(docker ps -q -f name=movie-sentiment-api)" ]; then
    echo "✅ Container is running!"
    echo "🌐 API available at: http://localhost:8000"
    echo "📖 API docs at: http://localhost:8000/docs"
    echo "❤️ Health check: http://localhost:8000/health"

    # Show container logs
    echo "📋 Container logs:"
    docker logs movie-sentiment-api --tail 10
else
    echo "❌ Container failed to start!"
    docker logs movie-sentiment-api
    exit 1
fi
