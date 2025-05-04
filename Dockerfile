FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p temp_docs

# Expose the port
EXPOSE 8501

# Wait for MongoDB and Ollama to be ready
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Run the application
CMD ["/bin/bash", "-c", "/wait-for-it.sh mongodb:27017 -t 60 && /wait-for-it.sh ollama:11434 -t 120 && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]