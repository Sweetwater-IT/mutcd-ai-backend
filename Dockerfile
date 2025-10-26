FROM python:3.13-slim

# Install system deps (Tesseract for OCR)
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 10000

# Run server (Render binds to $PORT)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
