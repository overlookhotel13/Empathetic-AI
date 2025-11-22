# Dockerfile - CPU-optimized for PyTorch + Transformers
FROM python:3.10-slim

# work dir
WORKDIR /app

# system deps commonly required for pip builds and for ffmpeg/hf
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        git-lfs \
        curl \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --upgrade pip setuptools wheel

# copy only requirements first for Docker caching
COPY requirements.txt /app/requirements.txt

# install CPU PyTorch from official PyTorch wheel index, then other requirements
# (This avoids slow/failed builds that try to compile from source)
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.6.0" && \
    pip install -r /app/requirements.txt

# copy the rest of the code
COPY . /app

# expose uvicorn port
EXPOSE 8000

# default command to run uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
