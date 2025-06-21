# RunPod WhisperX 使用指南

## 🚀 快速开始

### 必要的RunPod配置
- **GPU**: RTX 4090 (24GB VRAM)
- **RAM**: 最少16GB，推荐32GB+
- **存储**: 30GB+

### 自动运行（推荐）
```bash
# 在RunPod终端中运行一条命令完成所有设置
curl -s https://raw.githubusercontent.com/XihanJia1016/runpod-whisperx-project/main/runpod_setup_guide.sh | bash
```

### 手动运行步骤

#### 1. 进入工作目录
```bash
cd /workspace
```

#### 2. 获取项目代码
```bash
# 首次运行
git clone https://github.com/XihanJia1016/runpod-whisperx-project.git
cd runpod-whisperx-project

# 或更新现有项目
cd runpod-whisperx-project
git pull origin main
```

#### 3. 安装依赖
```bash
# 升级pip
python -m pip install --upgrade pip

# 安装所需包（需要几分钟）
pip install -r requirements.txt
```

#### 4. 设置环境变量
```bash
# 设置HuggingFace Token（用于说话人识别）
export HF_TOKEN="YOUR_HF_TOKEN_HERE"

# 设置CUDA库路径
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 5. 准备音频文件
```bash
# 创建输入目录
mkdir -p /workspace/input /workspace/output

# 上传音频文件到 /workspace/input/
# 支持格式：.mp3, .wav, .m4a, .flac, .ogg
# 建议文件名：dyad.conversation.mp3 (如: 19.4.mp3, 33.4.mp3)
```

#### 6. 运行转录
```bash
python run_batch.py
```

## 📋 输出文件

转录完成后，结果保存在 `/workspace/output/` 目录：

### 主要文件
- `*_combined_transcription.csv` - 完整转录结果
- `*_summary_stats.txt` - 统计报告

### CSV文件包含的列
- `dyad` - 对话组编号（从文件名提取）
- `conversation` - 对话编号（从文件名提取）
- `segment_id` - 片段序号
- `start_time` - 开始时间 (HH:MM:SS,mmm)
- `finish_time` - 结束时间 (HH:MM:SS,mmm)
- `duration` - 持续时间（秒）
- `speaker` - 说话人标识
- `speaker_raw` - 原始说话人信息
- `text` - 转录文本
- `confidence` - 置信度
- `word_count` - 词数
- `language` - 语言（nl-荷兰语）
- `model_used` - 使用的模型
- `device_used` - 使用的设备
- `has_ai_speaker_detection` - 是否成功进行说话人识别

## 💾 下载结果文件

### 方法1：Jupyter界面（推荐）
1. 在RunPod连接页面选择 "Jupyter Lab"
2. 导航到 `workspace/output/` 文件夹
3. 右键点击文件选择 "Download"

### 方法2：打包下载
```bash
# 打包所有结果
cd /workspace
tar -czf transcription_results.tar.gz output/

# 然后通过Jupyter下载 transcription_results.tar.gz
```

### 方法3：Web服务器
```bash
# 启动简单HTTP服务器
cd /workspace/output
python -m http.server 8000

# 在浏览器访问：你的RunPod外部IP:8000
```

## 🔧 故障排除

### 内存不足错误
- 确保选择了足够大的RAM（16GB+）
- 不要选择512MB RAM的配置

### cuDNN库错误
- 脚本会自动尝试修复
- 如果仍有问题，重启RunPod实例

### 模型下载慢
- 第一次运行会下载大模型文件
- 后续运行会使用缓存，速度较快

### 说话人识别失败
- 检查HF_TOKEN环境变量是否正确设置
- 确保网络连接正常

## 📊 性能指标

### 预期处理时间
- RTX 4090: 约2-4倍实时速度
- 8分钟音频约需要2-4分钟处理

### 预期成本（RunPod）
- RTX 4090: ~$0.34/小时
- 处理3个8分钟音频约$1-3

## 🎯 支持的音频格式

- MP3 (.mp3)
- WAV (.wav) 
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)

## 📝 文件命名建议

为了正确提取dyad和conversation信息，建议使用以下格式：
- `19.4.mp3` → dyad=19, conversation=4
- `33.4.mp3` → dyad=33, conversation=4
- `35.3.mp3` → dyad=35, conversation=3

如果文件名不符合此格式，脚本会使用默认值并给出警告。