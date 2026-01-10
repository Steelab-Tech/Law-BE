FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    python-dotenv \
    pymongo \
    sentence-transformers \
    FlagEmbedding

# Copy source code
COPY src/ ./src/
COPY entrypoint.sh .

# Convert CRLF to LF (Windows compatibility) and make executable
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

EXPOSE 8002

CMD ["sh", "entrypoint.sh"]
