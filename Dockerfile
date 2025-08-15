# Dockerfile for Data Analyst Agent API
# Build with: docker build -t data-analyst-agent .
# Run with: docker run -p 8000:8000 --env-file .env data-analyst-agent

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Use environment variables from .env if present
ENV PYTHONUNBUFFERED=1

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
