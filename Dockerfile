# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verify numpy installation
RUN python -c "import numpy; print(numpy.__version__)"

# Copy the application code
COPY . .

# Create weights directory if it doesn't exist
RUN mkdir -p weights

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
