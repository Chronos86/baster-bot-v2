# Dockerfile for Trading Bot on Render.com

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for TA-Lib and potentially other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# --- Install TA-Lib C Library --- 
# Download and install TA-Lib from source
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install
# Clean up TA-Lib source files
RUN rm -rf /tmp/*
# --- End TA-Lib C Library Install --- 

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Ensure TA-Lib wrapper is installed *after* the C library
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Streamlit will run on (Render uses $PORT)
# EXPOSE $PORT # Not strictly necessary as Render injects $PORT

# Command to run the application using Honcho
# Honcho will read the Procfile and start the 'web' and 'bot' processes
CMD ["honcho", "start"]

