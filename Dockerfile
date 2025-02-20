# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY src/ src/
COPY app/ app/
COPY checkpoints/ checkpoints/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/checkpoints

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]