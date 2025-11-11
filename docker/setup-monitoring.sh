#!/bin/bash

echo "🔧 Setting up Movie Sentiment Analysis Monitoring Stack..."

# Create necessary directories
mkdir -p docker/prometheus
mkdir -p docker/grafana/provisioning/dashboards
mkdir -p docker/grafana/provisioning/datasources

# Check if configuration files exist
if [ ! -f "docker/prometheus/prometheus.yml" ]; then
    echo "❌ Prometheus config not found!"
    exit 1
fi

# Build the API image first
echo "🐳 Building API image..."
docker build -f docker/Dockerfile -t movie-sentiment-api:latest .

# Start the monitoring stack
echo "🚀 Starting monitoring stack..."
docker-compose -f docker/docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
fi

# Check Prometheus
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana is not responding"
fi

echo ""
echo "🎉 Monitoring stack is ready!"
echo "📊 API: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "📊 Prometheus: http://localhost:9090"
echo "📊 Grafana: http://localhost:3000 (admin/admin123)"
echo ""
echo "🧪 Test the API to generate metrics:"
echo "curl -X POST 'http://localhost:8000/predict' -H 'Content-Type: application/json' -d '{\"text\": \"Great movie!\"}'"
