# python-ml-service/Dockerfile
# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the models directory
COPY models/ ./models/

# Copy the application code
COPY src/ ./src/

# Expose the port your FastAPI app listens on
EXPOSE 8000

# Command to run your application using Uvicorn (production ready)
# 'src.main:app' means 'app' object inside 'main.py' located in 'src' directory
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]