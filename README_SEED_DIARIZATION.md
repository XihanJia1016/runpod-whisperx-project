# 半自动种子识别说话人日志系统

## 概述

这个修改版本的WhisperX处理脚本实现了基于"种子"片段的半自动说话人识别功能，通过利用黄金标准文本来提高说话人识别的准确性。

## 主要改进

### 1. 替换模型架构
- **旧系统**: 使用 `pyannote/speaker-diarization-3.1` 进行全自动说话人识别
- **新系统**: 使用 `pyannote/embedding` 模型进行基于种子的说话人识别

### 2. 种子识别流程
1. **种子定位**: 根据黄金文本中的说话人标识（S/L），从AI转录中找到最匹配的片段作为"种子"
2. **特征提取**: 为每个说话人的种子片段生成声音嵌入向量
3. **说话人识别**: 对所有其他片段计算与种子的相似度，分配说话人标签

## 安装依赖

```bash
pip install pyannote.audio scikit-learn pandas torch pydub
```

确保设置HuggingFace访问令牌：
```bash
export HF_TOKEN='your_huggingface_token_here'
```

## 使用方法

### 方法1: 使用主函数（推荐）

修改 `main()` 函数中的路径配置：

```python
def main():
    # 配置路径
    audio_directory = "/path/to/your/audio/files"
    golden_text_file = "/Users/xihanjia/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/project/4 text mining/text_data_output.csv"
    output_directory = "/path/to/output"
    
    # 运行处理
    process_conversations_with_golden_text(
        audio_dir=audio_directory,
        golden_text_path=golden_text_file,
        output_dir=output_directory
    )
```

然后运行：
```bash
python process_audio_large.py
```

### 方法2: 手动调用函数

```python
from process_audio_large import process_conversations_with_golden_text

# 批量处理
process_conversations_with_golden_text(
    audio_dir="/path/to/audio",
    golden_text_path="/path/to/golden_text.csv",
    output_dir="/path/to/output"
)
```

### 方法3: 单文件处理

```python
from process_audio_large import HighPrecisionAudioProcessor
import pandas as pd

# 初始化处理器
processor = HighPrecisionAudioProcessor()
processor.load_models()

# 加载黄金文本
golden_df = pd.read_csv("golden_text.csv")
golden_turns = golden_df[
    (golden_df['dyad'] == 19) & 
    (golden_df['conversation'] == 4)
]

# 处理单个文件
segments = processor.process_single_file(
    audio_path="audio.wav",
    dyad_id=19,
    conversation_id=4,
    golden_turns_df=golden_turns
)
```

## 文件名约定

脚本会自动从音频文件名解析dyad和conversation信息。支持的格式：
- `dyad_19_conversation_4.wav`
- `audio_dyad_19_conversation_4.mp3`

或者使用自定义映射：
```python
conversation_mapping = {
    "audio1.wav": (19, 4),
    "audio2.wav": (33, 4),
    "custom_name.mp3": (35, 3)
}
```

## 黄金文本格式要求

CSV文件必须包含以下列：
- `dyad`: 对话组ID
- `conversation`: 对话编号  
- `role`: 说话人标识（'S' 或 'L'）
- `text`: 说话人轮次的文本内容

示例：
```csv
dyad,conversation,role,text
19,4,S,"你好，我想讨论一下..."
19,4,L,"是的，我同意你的观点..."
```

## 输出格式

生成的CSV文件包含以下字段：
- `dyad`, `conversation`: 对话标识
- `segment_id`: 片段序号
- `start_time`, `finish_time`: 时间戳（HH:MM:SS,mmm格式）
- `speaker`: 说话人（Speaker_A/Speaker_B）
- `text`: 转录文本
- `confidence`: 转录置信度
- `has_ai_speaker_detection`: 是否成功进行说话人识别

## 算法细节

### 种子片段查找
1. 从黄金文本中获取每个说话人（S/L）的第一个轮次作为目标文本
2. 使用贪心算法在AI转录片段中寻找最佳匹配组合
3. 基于文本相似度（difflib.SequenceMatcher）选择种子片段

### 说话人识别
1. 为种子片段生成嵌入向量（使用pyannote/embedding模型）
2. 计算所有片段与种子的余弦相似度
3. 根据相似度分配说话人标签

## 性能优化

- GPU加速：自动检测并使用CUDA
- 批量处理：支持多文件批量转录
- 内存管理：自动清理GPU内存
- 错误处理：单文件失败不影响整体处理

## 故障排除

### 1. 模型加载失败
```
错误: 模型加载失败
解决: 检查HF_TOKEN环境变量，确保网络连接正常
```

### 2. 找不到种子片段
```
警告: 未能找到S和L的种子
解决: 检查黄金文本格式，确保包含S和L的轮次
```

### 3. 内存不足
```
错误: CUDA out of memory
解决: 降低batch_size或使用CPU模式
```

## 上传到GitHub

修改完成后，将整个项目目录上传到：
https://github.com/XihanJia1016/runpod-whisperx-project

## 版本历史

- v2.0: 实现半自动种子识别说话人日志
- v1.0: 原始WhisperX全自动处理系统