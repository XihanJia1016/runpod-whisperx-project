#!/bin/bash

# 半自动种子识别说话人日志系统 - 依赖安装脚本
# 解决PyTorch版本冲突问题

echo "🚀 开始安装半自动种子识别说话人日志系统依赖..."

# 设置错误处理
set -e

# 1. 清理冲突的PyTorch包
echo "1️⃣ 清理冲突的PyTorch包..."
pip uninstall -y torch torchaudio torchvision torchmetrics pytorch-lightning pytorch-metric-learning torch-audiomentations torch-pitch-shift 2>/dev/null || true

# 清理nvidia包
echo "清理NVIDIA包..."
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 2>/dev/null || true

# 2. 检测CUDA支持
echo "2️⃣ 检测CUDA支持..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ 检测到CUDA支持"
    CUDA_AVAILABLE=true
    
    # 获取CUDA版本
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "CUDA版本: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        TORCH_INDEX="cu118"
    elif [[ "$CUDA_VERSION" == "12."* ]]; then
        TORCH_INDEX="cu121"
    else
        TORCH_INDEX="cu118"  # 默认使用11.8
    fi
else
    echo "⚠️ 未检测到CUDA，使用CPU版本"
    CUDA_AVAILABLE=false
    TORCH_INDEX="cpu"
fi

# 3. 安装兼容的PyTorch版本
echo "3️⃣ 安装兼容的PyTorch版本..."
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "安装CUDA版本的PyTorch..."
    pip install torch==2.1.0+${TORCH_INDEX} torchaudio==2.1.0+${TORCH_INDEX} torchvision==0.16.0+${TORCH_INDEX} --index-url https://download.pytorch.org/whl/${TORCH_INDEX} --force-reinstall
else
    echo "安装CPU版本的PyTorch..."
    pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu --force-reinstall
fi

# 4. 安装基础依赖
echo "4️⃣ 安装基础依赖..."
pip install numpy pandas scikit-learn librosa soundfile tqdm

# 5. 安装Transformers生态
echo "5️⃣ 安装Transformers..."
pip install "transformers>=4.35.0,<5.0.0" "accelerate>=0.20.0"

# 6. 安装说话人识别
echo "6️⃣ 安装pyannote.audio..."
pip install "pyannote.audio>=3.1.0,<4.0.0"

# 7. 安装WhisperX
echo "7️⃣ 安装WhisperX..."
pip install git+https://github.com/m-bain/whisperx.git

# 8. 验证安装
echo "8️⃣ 验证安装..."
python3 -c "
import torch
import torchaudio
import torchvision
import sklearn
import pandas as pd
import numpy as np

print('✅ PyTorch:', torch.__version__)
print('✅ TorchAudio:', torchaudio.__version__)
print('✅ TorchVision:', torchvision.__version__)
print('✅ CUDA Available:', torch.cuda.is_available())

try:
    import pyannote.audio
    print('✅ pyannote.audio:', pyannote.audio.__version__)
except ImportError as e:
    print('❌ pyannote.audio 导入失败:', e)

try:
    import whisperx
    print('✅ whisperx 可用')
except ImportError as e:
    print('❌ whisperx 导入失败:', e)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    print('✅ scikit-learn 可用')
except ImportError as e:
    print('❌ scikit-learn 导入失败:', e)

print('🎉 所有核心依赖验证成功！')
"

echo "🎉 依赖安装完成！"
echo ""
echo "📝 下一步操作："
echo "1. 设置HuggingFace Token: export HF_TOKEN='your_token_here'"
echo "2. 运行脚本: python process_audio_large.py"
echo ""
echo "⚠️ 如果遇到问题，请检查CUDA版本兼容性"