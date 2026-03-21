FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/appuser
ENV STREAMLIT_HOME=/home/appuser/.streamlit
ENV XDG_CONFIG_HOME=/home/appuser/.config

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Provide a real passwd/group entry for uid/gid 1000 (used by docker-compose).
RUN getent group 1000 || groupadd -g 1000 appuser && \
    getent passwd 1000 || useradd -m -u 1000 -g 1000 -s /bin/bash appuser && \
    mkdir -p /home/appuser/.streamlit /home/appuser/.config && \
    chown -R 1000:1000 /home/appuser

# Set python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit UI
CMD ["streamlit", "run", "ui/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--", "--config", "configs/default.yaml"]
