FROM python:3.11-slim

# Install FFmpeg and system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create empty __init__.py files for packages
RUN touch agents/__init__.py tools/__init__.py utils/__init__.py config/__init__.py

# Verify FFmpeg is available
RUN ffmpeg -version | head -1

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python3", "job_worker.py"]
