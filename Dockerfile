# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /src

# Copy files
COPY . .

# Install system dependencies (optional: improve compatibility)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Streamlit specific config (to prevent browser launch)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS false

# Expose the port Streamlit runs on
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
