# CPU image: adjust python version as needed
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libglib2.0-0 libgl1 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy code
COPY . /app

# Install Python deps (use PyTorch CPU wheels)
# Note: this will install a CPU-only torch. For GPU, see notes below.
RUN pip install --upgrade pip
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0.0" "torchvision>=0.15.0" \
    && pip install -r requirements.txt

# Create results folder
RUN mkdir -p /app/uploads /app/static/results

ENV FLASK_ENV=production
ENV PORT=5000

EXPOSE 5000

# Default command
CMD ["python", "app.py"]
