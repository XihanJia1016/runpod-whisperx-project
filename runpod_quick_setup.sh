#!/bin/bash

# RunPod 极简设置脚本 - 无验证，快速启动
echo "🚀 RunPod 极简设置开始..."

# 1. 进入工作目录
cd /workspace

# 2. 克隆/更新项目
if [ -d "runpod-whisperx-project" ]; then
    echo "📁 更新现有项目..."
    cd runpod-whisperx-project && git pull origin main
else
    echo "📁 克隆新项目..."
    git clone https://github.com/XihanJia1016/runpod-whisperx-project.git
    cd runpod-whisperx-project
fi

# 3. 快速安装依赖（忽略错误）
echo "📦 安装依赖..."
python -m pip install --upgrade pip >/dev/null 2>&1
pip install -r requirements.txt >/dev/null 2>&1 || echo "⚠️ 依赖安装可能有警告，继续..."

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

echo "✅ 设置完成！"
echo "📋 使用步骤:"
echo "1. 将音频文件(.wav)放在 /workspace/input/"
echo "2. 将 text_data_output.csv 放在 /workspace/input/"
echo "3. 运行: python process_audio_large.py"