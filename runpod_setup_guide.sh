#!/bin/bash

# ================================================================
# RunPod WhisperX 半自动种子识别说话人日志系统 - 自动设置脚本
# ================================================================
# 
# 使用说明：
# 1. 在RunPod终端中运行：bash runpod_setup_guide.sh
# 2. 或者按照下面的步骤手动执行
# 3. 确保已设置HF_TOKEN环境变量
#
# ================================================================

echo "🚀 RunPod WhisperX 半自动种子识别说话人日志系统 - 自动设置开始"
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

# 第四步：安装依赖（处理NumPy版本冲突）
echo -e "\n📚 第四步：安装Python依赖"
echo "⏰ 这可能需要几分钟时间..."

# 检查是否存在NumPy版本冲突
echo "🔍 检查依赖兼容性..."
if ! pip install -r requirements.txt 2>/dev/null; then
    echo "⚠️ 检测到依赖冲突，运行自动修复..."
    
    # 运行NumPy冲突修复脚本
    if [ -f "fix_numpy_conflict.py" ]; then
        python fix_numpy_conflict.py
    else
        echo "🔧 手动修复NumPy版本冲突..."
        pip uninstall -y numpy
        pip install 'numpy>=2.0.2'
        pip install -r requirements.txt
    fi
else
    echo "✅ 依赖安装成功"
fi

# 第五步：设置环境变量
echo -e "\n🔧 第五步：设置环境变量"

# 检查HF_TOKEN是否已设置
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN未设置！"
    echo "📝 请先设置HuggingFace Token："
    echo "   export HF_TOKEN='hf_your_token_here'"
    echo ""
    echo "💡 获取Token方法："
    echo "   1. 访问 https://huggingface.co/"
    echo "   2. 登录/注册账户"
    echo "   3. 进入 Settings -> Access Tokens"
    echo "   4. 创建新Token（Read权限即可）"
    echo ""
    echo "⚠️ 请设置Token后重新运行脚本"
    exit 1
else
    echo "✅ HuggingFace Token 已设置: ${HF_TOKEN:0:10}..."
fi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "✅ CUDA库路径已设置"

# 第六步：创建必要目录
echo -e "\n📂 第六步：创建输入输出目录"
mkdir -p /workspace/input /workspace/output
echo "✅ 输入目录：/workspace/input"
echo "✅ 输出目录：/workspace/output"

# 第七步：验证环境
echo -e "\n🔍 第七步：验证环境安装"
python -c "
try:
    import whisperx
    import pyannote.audio
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    print('✅ 所有依赖验证成功')
    print(f'✅ NumPy: {np.__version__}')
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ 依赖验证失败: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 环境验证失败，请检查依赖安装"
    exit 1
fi

# 第八步：检查输入文件和黄金文本
echo -e "\n🎵 第八步：检查音频文件和黄金文本"
if [ "$(ls -A /workspace/input/ 2>/dev/null)" ]; then
    echo "✅ 发现音频文件："
    ls -la /workspace/input/
else
    echo "⚠️ 请将音频文件上传到 /workspace/input/ 目录"
    echo "💡 支持格式：.mp3, .wav, .m4a, .flac, .ogg"
    echo "💡 建议文件名格式：dyad_X_conversation_Y.mp3 (如: dyad_19_conversation_4.mp3)"
    echo ""
fi

echo -e "\n📋 使用说明："
echo "1. 上传音频文件到 /workspace/input/"
echo "2. 确保黄金标准文本CSV文件可访问"
echo "3. 修改 process_audio_large.py 中的路径配置"
echo "4. 运行: python process_audio_large.py"
echo ""
echo "💡 半自动种子识别系统特点："
echo "   - 基于黄金文本的说话人种子识别"
echo "   - 更高的说话人识别准确率"
echo "   - 支持S/L说话人标识系统"
echo ""

# 第九步：显示运行命令
echo -e "\n🎯 第九步：运行种子识别转录"
echo "📝 请根据您的需求修改以下路径后运行："
echo ""
echo "python process_audio_large.py"
echo ""
echo "或者使用批量处理函数："
echo "python -c \""
echo "from process_audio_large import process_conversations_with_golden_text"
echo "process_conversations_with_golden_text("
echo "    audio_dir='/workspace/input',"
echo "    golden_text_path='/path/to/your/golden_text.csv',"
echo "    output_dir='/workspace/output'"
echo ")\""

# 完成
echo ""
echo "🎉 RunPod环境设置完成！"
echo "📁 目录结构："
echo "   - /workspace/input/ - 音频文件目录"
echo "   - /workspace/output/ - 输出结果目录"
echo ""
echo "📝 下一步操作："
echo "   1. 设置HF_TOKEN: export HF_TOKEN='your_token'"
echo "   2. 上传音频文件到 /workspace/input/"
echo "   3. 配置黄金文本路径在 process_audio_large.py"
echo "   4. 运行: python process_audio_large.py"
echo ""
echo "💾 输出文件："
echo "   - combined_transcription_with_seed_diarization.csv - 种子识别转录结果"
echo "   - 包含说话人识别、时间戳和质量评分"
echo ""
echo "📋 下载方法："
echo "   1. Jupyter界面：导航到output文件夹右键下载"
echo "   2. 终端打包：tar -czf results.tar.gz output/"
echo "================================================================"