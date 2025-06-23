# RunPod WhisperX 半自动种子识别说话人日志系统

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/XihanJia1016/runpod-whisperx-project.git
cd runpod-whisperx-project
```

### 2. 自动环境设置（推荐）
```bash
python setup_environment.py
```

### 3. 设置HuggingFace Token
```bash
export HF_TOKEN='your_huggingface_token_here'
```

### 4. 运行脚本
```bash
python process_audio_large.py
```

## 🔧 环境问题解决

如果遇到依赖冲突或版本问题，脚本会自动检测并提供修复选项：

### 自动修复
- 脚本启动时会自动检查环境
- 发现问题时会提示是否自动修复
- 选择 `y` 进行自动修复

### 手动修复
```bash
# 使用稳定版本依赖
pip install -r requirements_stable.txt

# 或运行环境设置脚本
python setup_environment.py
```

### 清理重装
```bash
# 清理冲突包
pip uninstall -y torch torchaudio torchvision pyannote.audio whisperx

# 重新安装
python setup_environment.py
```

## 📝 使用方法

### 批量处理（推荐）
修改 `process_audio_large.py` 中的路径配置：
```python
def main():
    audio_directory = "/path/to/your/audio/files"
    golden_text_file = "/path/to/golden_text.csv" 
    output_directory = "/path/to/output"
```

### 单文件处理
```python
from process_audio_large import HighPrecisionAudioProcessor
processor = HighPrecisionAudioProcessor()
processor.load_models()
segments = processor.process_single_file("audio.wav", 19, 4, golden_turns_df)
```

## 🎯 核心特性

- **半自动种子识别**: 基于黄金文本的说话人识别
- **环境自检**: 自动检测和修复依赖冲突
- **版本管理**: 使用稳定的版本组合避免冲突
- **批量处理**: 支持多文件批量转录
- **错误恢复**: 单文件失败不影响整体处理

## 📋 文件说明

- `process_audio_large.py` - 主要处理脚本
- `setup_environment.py` - 环境设置和依赖修复
- `requirements_stable.txt` - 稳定版本依赖
- `README_SEED_DIARIZATION.md` - 详细技术文档

## ⚡ 常见问题

### PyTorch版本冲突
```
错误: torchvision 0.16.0+cu118 requires torch==2.1.0
解决: python setup_environment.py
```

### 导入错误
```
错误: No module named 'pyannote.audio'
解决: 脚本会自动检测并提示修复
```

### CUDA问题
```
错误: CUDA out of memory
解决: 脚本会自动选择CPU版本作为备选
```

## 🔄 版本历史

- v3.0: 添加环境自检和自动修复
- v2.0: 实现半自动种子识别说话人日志
- v1.0: 原始WhisperX全自动处理系统
