#!/bin/bash

# RunPod WhisperX 半自动种子识别系统 - 快速设置
# 使用: bash runpod_setup_guide.sh

# 1. 进入工作目录
cd /workspace

# 2. 克隆/更新项目
if [ -d "runpod-whisperx-project" ]; then
    cd runpod-whisperx-project && git pull origin main
else
    git clone https://github.com/XihanJia1016/runpod-whisperx-project.git
    cd runpod-whisperx-project
fi

# 3. 升级pip并安装依赖
python -m pip install --upgrade pip

# 自动处理NumPy冲突
if ! pip install -r requirements.txt 2>/dev/null; then
    echo "修复NumPy版本冲突..."
    python fix_numpy_conflict.py
fi

# 4. 验证HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ 需要设置HF_TOKEN"
    echo "运行: export HF_TOKEN='your_token_here'"
    echo "获取Token: https://huggingface.co/settings/tokens"
    exit 1
fi

# 5. 设置环境
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
mkdir -p /workspace/input /workspace/output

# 6. 验证安装
python -c "
import whisperx, pyannote.audio, torch, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
print('✅ 环境验证成功')
print(f'NumPy: {np.__version__}, PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"

echo "🎉 设置完成！"
echo "下一步: python process_audio_large.py"