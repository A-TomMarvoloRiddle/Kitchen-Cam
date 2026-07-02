FROM python:3.10-slim

# Install system dependencies for OpenCV and other utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p input output logs models

# Copy application source code
COPY src/ src/
COPY dashboard/ dashboard/
COPY main.py .
COPY dashboard_server.py .
COPY config/ config/

# Copy models (if they are bundled, though typically they are downloaded or mounted)
# COPY models/ models/

# Expose backend port
EXPOSE 8000

# Run the backend server
CMD ["python", "dashboard_server.py"]
