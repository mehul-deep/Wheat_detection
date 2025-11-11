#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set environment variables for local development
export STORAGE_BACKEND="local"
export MODEL_PATH="model.pth"
export YOLO_MODEL_PATH="best.pt"
export IMG_SIZE="1024"
export DEVICE="cpu"
export YOLO_DISEASE_CLASS="2"
export YOLO_WHEAT_CLASSES="1,2"
export LOCAL_UPLOAD_DIR="uploads"
export LOCAL_RESULTS_DIR="static/results"
export FLASK_SECRET="local-development-secret-key"
export PORT="5000"

# Create directories if they don't exist
mkdir -p uploads
mkdir -p static/results

echo "=================================="
echo "Starting Wheat Detection App (NO AUTH MODE)"
echo "=================================="
echo "⚠️  WARNING: Running without authentication!"
echo "This will fail when accessing Firebase."
echo "Use this only to test if the app starts."
echo "=================================="
echo "URL: http://localhost:5000"
echo "=================================="
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python app.py
