#!/bin/bash

# ================================================================
# RunPod WhisperX 自动设置和运行脚本
# ================================================================
# 
# 使用说明：
# 1. 在RunPod终端中运行：bash runpod_setup_guide.sh
# 2. 或者按照下面的步骤手动执行
#
# ================================================================

echo "🚀 RunPod WhisperX Large-v3 自动设置开始"
echo "================================================================"

# 第一步：进入工作目录
echo "📁 第一步：进入工作目录"
cd /workspace
pwd

# 第二步：克隆项目（如果不存在）
echo -e "\n📦 第二步：获取项目代码"
if [ -d "runpod-whisperx-project" ]; then
    echo "✅ 项目目录已存在，更新代码..."
    cd runpod-whisperx-project
    git pull origin main
else
    echo "🔄 克隆项目..."
    git clone https://github.com/XihanJia1016/runpod-whisperx-project.git
    cd runpod-whisperx-project
fi

# 第三步：升级pip
echo -e "\n🔧 第三步：升级pip"
python -m pip install --upgrade pip

# 第四步：安装依赖
echo -e "\n📚 第四步：安装Python依赖"
echo "⏰ 这可能需要几分钟时间..."
pip install -r requirements.txt

# 第五步：设置环境变量
echo -e "\n🔧 第五步：设置环境变量"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "✅ HuggingFace Token 已设置"
echo "✅ CUDA库路径已设置"

# 第六步：创建必要目录
echo -e "\n📂 第六步：创建输入输出目录"
mkdir -p /workspace/input /workspace/output
echo "✅ 输入目录：/workspace/input"
echo "✅ 输出目录：/workspace/output"

# 第七步：检查输入文件
echo -e "\n🎵 第七步：检查音频文件"
if [ "$(ls -A /workspace/input/ 2>/dev/null)" ]; then
    echo "✅ 发现音频文件："
    ls -la /workspace/input/
else
    echo "⚠️  请将音频文件上传到 /workspace/input/ 目录"
    echo "💡 支持格式：.mp3, .wav, .m4a, .flac, .ogg"
    echo "💡 建议文件名格式：dyad.conversation.mp3 (如: 19.4.mp3, 33.4.mp3)"
    echo ""
    echo "📋 上传文件后，运行以下命令开始转录："
    echo "   python run_batch.py"
    echo ""
    echo "================================================================"
    exit 0
fi

# 第八步：运行转录
echo -e "\n🎯 第八步：开始音频转录"
echo "⏰ 这将需要一些时间，请耐心等待..."
echo "================================================================"
python run_batch.py

# 完成
echo ""
echo "🎉 转录完成！"
echo "📁 结果文件保存在：/workspace/output/"
echo "💾 主要文件："
echo "   - *_combined_transcription.csv - 完整转录结果"
echo "   - *_summary_stats.txt - 统计报告"
echo ""
echo "📋 下载文件方法："
echo "   1. 通过Jupyter界面：导航到output文件夹右键下载"
echo "   2. 通过终端打包：tar -czf results.tar.gz output/"
echo "================================================================"