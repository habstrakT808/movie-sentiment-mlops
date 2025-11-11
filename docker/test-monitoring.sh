#!/bin/bash

echo "🧪 Testing Movie Sentiment Analysis API and Monitoring..."

API_URL="http://localhost:8000"

# Test API endpoints and generate metrics
echo "1. Testing single prediction..."
curl -X POST "$API_URL/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic!"}'

echo -e "\n2. Testing batch prediction..."
curl -X POST "$API_URL/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Amazing film with great acting!",
         "Terrible movie, waste of time.",
         "It was okay, nothing special.",
         "Brilliant cinematography and story!",
         "Boring and predictable plot."
       ],
       "include_probabilities": true
     }'

echo -e "\n3. Testing health check..."
curl "$API_URL/health"

echo -e "\n4. Testing model info..."
curl "$API_URL/model/info" | jq '.model_name, .device, .parameters'

echo -e "\n5. Checking metrics..."
curl "$API_URL/metrics" | grep -E "(http_requests_total|predictions_total|prediction_duration)" | head -10

echo -e "\n6. Testing error handling..."
curl -X POST "$API_URL/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": ""}'

echo -e "\n✅ Monitoring test complete!"
echo "📊 Check Prometheus: http://localhost:9090/targets"
echo "📊 Check Grafana: http://localhost:3000"
