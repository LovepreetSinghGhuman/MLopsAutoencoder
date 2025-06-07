# Use a slim Python base image
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Install system dependencies for pandas, openpyxl, and TensorFlow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libopenblas-dev \
        libpq-dev \
        libffi-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        libjpeg-dev \
        zlib1g-dev \
        git \
        curl \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set a working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY deployment/requirements.txt ./requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy scoring script and model artifacts
COPY deployment/score.py ./score.py
COPY models/ ./models/

# Expose port (match uvicorn port)
EXPOSE 8000

# Healthcheck (optional, for container orchestrators)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Run the app with uvicorn
CMD ["uvicorn", "score:app", "--host", "0.0.0.0", "--port", "8000"]