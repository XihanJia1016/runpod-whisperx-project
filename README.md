# RunPod WhisperX Large-v3 高精度转录项目

## 项目说明
使用WhisperX Large-v3模型对3个8分钟音频进行高精度转录，包含说话人识别和精确时间戳。

## 使用步骤

### 1. 构建Docker镜像
```bash
docker build -t your-username/whisperx-large:latest .
```

### 2. 推送到Docker Hub
```bash
docker push your-username/whisperx-large:latest
```

### 3. 在RunPod部署
- Container Image: your-username/whisperx-large:latest
- GPU: RTX 4090 (推荐)
- Volume: 30GB

### 4. 上传音频文件
将音频文件上传到 `/workspace/input/`
- 35.1.mp3
- 35.2.mp3  
- 35.3.mp3

### 5. 运行处理
```bash
python run_batch.py
```

### 6. 下载结果
主要输出文件：
- `dyad_35_combined_transcription.csv` - 合并的转录结果
- `dyad_35_summary_stats.txt` - 统计报告

## 输出格式
CSV包含以下列：
- dyad, conversation, segment_id
- start_time, finish_time (格式: HH:MM:SS,mmm)
- duration, speaker, speaker_raw, text
- confidence, word_count, language
- model_used, device_used, has_ai_speaker_detection

## 预期成本
- RTX 4090: ~$0.34/小时
- 处理时间: 1-2小时
- 总成本: $1-3
