# Use Python 3.12 as base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for vector database
RUN mkdir -p ./chroma_db

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "waitress_server.py"] 