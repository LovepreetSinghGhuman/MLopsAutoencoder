# Use a slim Python base image
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Install system dependencies for pandas, openpyxl, and TensorFlow
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set a working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY deployment/requirements.txt ./requirements.txt

# Upgrade pip and install dependencies, then clean up cache
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy source code
COPY src/ ./src/

# Ensure models directory exists (even if empty)
RUN mkdir -p models

# (Optional) Use a non-root user for security
RUN useradd -m appuser
USER appuser

# (Optional) Add a healthcheck for production
HEALTHCHECK CMD curl --fail http://localhost:8000/docs || exit 1

# Set entrypoint
CMD ["uvicorn", "src.score:app", "--host", "0.0.0.0", "--port", "8000"]