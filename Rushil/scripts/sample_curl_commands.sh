#!/bin/bash

# Sample cURL commands for testing the Gemma VLM API

echo "=== Gemma VLM API Test Commands ==="

echo "1. Health Check:"
echo "curl http://localhost:8000/health"
echo ""


echo "2. Metrics:"
echo "curl http://localhost:8000/metrics"
echo ""

echo "3. Analyze Images (requires base64 encoded image):"
echo "curl -X POST http://localhost:8000/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"images\": [\"$(base64 -w 0 sample_image.jpg)\"],"
echo "    \"query\": \"Find all cars in the image\""
echo "  }'"
echo ""

echo "4. Batch Analyze (multiple images):"
echo "curl -X POST http://localhost:8000/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"images\": ["
echo "      \"$(base64 -w 0 image1.jpg)\","
echo "      \"$(base64 -w 0 image2.jpg)\""
echo "    ],"
echo "    \"query\": \"Identify all people wearing red clothing\""
echo "  }'"
echo ""

echo "5. Security Surveillance Query:"
echo "curl -X POST http://localhost:8000/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"images\": [\"$(base64 -w 0 security_frame.jpg)\"],"
echo "    \"query\": \"Identify suspicious handoffs near the blue sedan\""
echo "  }'"
echo ""

echo "6. Vehicle Detection Query:"
echo "curl -X POST http://localhost:8000/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"images\": [\"$(base64 -w 0 parking_lot.jpg)\"],"
echo "    \"query\": \"Detect any vehicles entering the parking lot\""
echo "  }'"
echo ""


if command -v convert &> /dev/null; then
    echo "Creating sample test image..."
    convert -size 640x480 xc:white -fill black -pointsize 72 -gravity center -annotate +0+0 "Test Image" sample_image.jpg
    echo "Sample image created: sample_image.jpg"
    echo ""
fi

echo "Note: Replace the base64 encoded image strings with actual base64 data from your images"
echo "To encode an image: base64 -w 0 your_image.jpg"
