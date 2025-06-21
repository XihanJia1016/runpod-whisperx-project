FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 确保Python和pip可用
RUN python --version && pip --version

# 升级pip
RUN pip install --upgrade pip setuptools wheel

# PyTorch already included in base image

# 复制requirements并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载模型（避免运行时下载）
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-large-v3')"

# 复制应用文件
COPY . .

# 创建目录
RUN mkdir -p /workspace/input /workspace/output /workspace/cache

# 设置环境变量
ENV TRANSFORMERS_CACHE=/workspace/cache
ENV HF_HOME=/workspace/cache
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/workspace

# 设置权限
RUN chmod +x *.py

# 默认命令
CMD ["python", "run_batch.py"]
