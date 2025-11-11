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
export FIREBASE_PROJECT_ID="wheat-detection-cb988"
export PORT="5000"

# Fix for PyTorch/OpenCV memory crash (free(): double free)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONMALLOC=malloc
export OPENCV_IO_MAX_IMAGE_PIXELS=1099511627776

# Create directories if they don't exist
mkdir -p uploads
mkdir -p static/results

# Set Google Application Default Credentials
# This tells Firebase Admin SDK and Firestore to use your local gcloud credentials
# Unset any existing GOOGLE_APPLICATION_CREDENTIALS to force use of default creds
unset GOOGLE_APPLICATION_CREDENTIALS

# Verify gcloud authentication
echo "Checking gcloud authentication..."
if ! gcloud auth application-default print-access-token &>/dev/null; then
    echo "⚠️  ERROR: Not authenticated with gcloud!"
    echo "Please run: gcloud auth application-default login"
    exit 1
fi
echo "✓ Authenticated with gcloud"
echo ""

echo "=================================="
echo "Starting Wheat Detection App (Local Mode)"
echo "=================================="
echo "Storage: Local filesystem"
echo "Models: model.pth, best.pt"
echo "Firebase: wheat-detection-cb988"
echo "URL: http://localhost:5000"
echo "=================================="
echo ""
echo "Make sure you're logged in to gcloud:"
echo "  gcloud auth application-default login"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="
echo ""

# Run the Flask app
python app.py
