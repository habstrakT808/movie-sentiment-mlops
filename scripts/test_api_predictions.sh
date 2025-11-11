#!/bin/bash
# Test script for API prediction endpoints

API_URL="http://localhost:8000"

echo "=========================================="
echo "Testing Movie Sentiment Analysis API"
echo "=========================================="
echo ""

# Test 1: Single prediction - positive text
echo "Test 1: Single prediction (positive text)"
echo "------------------------------------------"
curl -X POST "${API_URL}/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely amazing! Best film ever!", "include_probabilities": true}' \
     -w "\n\n"
echo ""

# Test 2: Single prediction - negative text
echo "Test 2: Single prediction (negative text)"
echo "------------------------------------------"
curl -X POST "${API_URL}/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Terrible movie, waste of time. Very disappointing.", "include_probabilities": true}' \
     -w "\n\n"
echo ""

# Test 3: Batch prediction
echo "Test 3: Batch prediction"
echo "------------------------------------------"
curl -X POST "${API_URL}/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "This movie was fantastic!",
         "Terrible film, very boring.",
         "It was okay, nothing special."
       ],
       "include_probabilities": true
     }' \
     -w "\n\n"
echo ""

# Test 4: Empty text (should fail validation)
echo "Test 4: Empty text validation (should fail)"
echo "------------------------------------------"
curl -X POST "${API_URL}/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": ""}' \
     -w "\n\n"
echo ""

# Test 5: Very long text (should work but truncated)
echo "Test 5: Very long text (should work)"
echo "------------------------------------------"
LONG_TEXT=$(python -c "print('This movie is great! ' * 100)")
curl -X POST "${API_URL}/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"${LONG_TEXT}\"}" \
     -w "\n\n"
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
