# Dockerfile for Trading Bot on Render.com

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for TA-Lib and potentially other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    unzip \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# --- Install TA-Lib C Library --- 
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install

# Set the library path for TA-Lib
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# Clean up TA-Lib source files
RUN rm -rf /tmp/*
# --- End TA-Lib C Library Install --- 

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
# Ensure TA-Lib wrapper is installed *after* the C library and LD_LIBRARY_PATH is set
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["honcho", "start"]

