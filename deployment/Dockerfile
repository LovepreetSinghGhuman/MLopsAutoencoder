# Use a slim Python base image
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for pandas, openpyxl, and TensorFlow
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set a working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY deployment/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Set entrypoint
CMD ["uvicorn", "src.score:app", "--host", "0.0.0.0", "--port", "8000"]