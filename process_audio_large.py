"""
RunPod WhisperX Large-v3 高精度音频处理脚本
优化说话人识别和时间戳精度
"""

import whisperx
import pandas as pd
import torch
import gc
import os
import time
import logging
from pathlib import Path
import numpy as np
from pyannote.audio import Pipeline
import subprocess
import shutil

# 禁用TF32以避免精度和兼容性问题
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_ffmpeg():
    """确保ffmpeg已安装"""
    if shutil.which('ffmpeg') is None:
        logger.info("🔧 未找到ffmpeg，正在自动安装...")
        try:
            # 尝试使用apt安装
            subprocess.run(['apt', 'update'], check=True, capture_output=True)
            subprocess.run(['apt', 'install', '-y', 'ffmpeg'], check=True, capture_output=True)
            logger.info("✅ ffmpeg安装成功")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # 如果apt失败，尝试conda
                subprocess.run(['conda', 'install', '-c', 'conda-forge', 'ffmpeg', '-y'], check=True, capture_output=True)
                logger.info("✅ ffmpeg通过conda安装成功")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error("❌ 无法自动安装ffmpeg，请手动安装")
                logger.info("💡 手动安装命令: apt install -y ffmpeg 或 conda install -c conda-forge ffmpeg")
                raise RuntimeError("ffmpeg未安装且无法自动安装")
    else:
        logger.info("✅ ffmpeg已安装")

class HighPrecisionAudioProcessor:
    def __init__(self):
        # 确保ffmpeg已安装
        ensure_ffmpeg()
        self.device = self._setup_device()
        self.model = None
        self.align_model = None
        self.metadata = None
        self.diarize_model = None
        
        # 高精度配置
        self.config = {
            'model_size': 'large-v3',
            'batch_size': 8,  # GPU优化
            'chunk_length': 30,  # 30秒块，平衡精度和内存
            'return_attention': True,
            'word_timestamps': True,
            'vad_filter': True,
            'temperature': 0.0,  # 确定性输出
        }
        
        logger.info(f"初始化高精度处理器: 设备={self.device}")
    
    def _setup_device(self):
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            device = "cpu"
            logger.warning("未检测到GPU，使用CPU（会很慢）")
        return device
    
    def load_models(self):
        """加载所有必需模型"""
        try:
            # 1. 加载Whisper Large-v3
            logger.info("加载Whisper Large-v3模型...")
            self.model = whisperx.load_model(
                self.config['model_size'], 
                device=self.device,
                compute_type="float16",
                download_root="/workspace/cache"
            )
            logger.info("✅ Whisper Large-v3加载完成")
            
            # 2. 加载对齐模型（提高时间戳精度）
            logger.info("加载强制对齐模型...")
            self.align_model, self.metadata = whisperx.load_align_model(
                language_code="nl",  # 荷兰语
                device=self.device
            )
            logger.info("✅ 对齐模型加载完成")
            
            # 3. 加载说话人识别模型
            logger.info("加载说话人识别模型...")
            # 获取HuggingFace token (需要设置环境变量 HF_TOKEN)
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                logger.warning("⚠️ 未设置HF_TOKEN环境变量，说话人识别可能失败")
                logger.info("💡 请运行: export HF_TOKEN='your_token_here'")
            
            self.diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=hf_token
            )
            if self.device == "cuda":
                self.diarize_model.to(torch.device("cuda"))
            logger.info("✅ 说话人识别模型加载完成")
            
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def _seconds_to_timestamp(self, seconds):
        """将秒数转换为 HH:MM:SS,mmm 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def process_single_file(self, audio_path, dyad_id, conversation_id):
        """处理单个音频文件"""
        logger.info(f"开始处理: {Path(audio_path).name}")
        start_time = time.time()
        
        try:
            # 1. 加载音频
            logger.info("加载音频...")
            audio = whisperx.load_audio(audio_path)
            duration = len(audio) / 16000
            logger.info(f"音频时长: {duration:.1f}秒")
            
            # 2. 转录（Large-v3高精度）
            logger.info("开始高精度转录...")
            result = self.model.transcribe(
                audio,
                batch_size=self.config['batch_size']
            )
            
            segments = result.get("segments", [])
            logger.info(f"转录完成: {len(segments)}个片段")
            
            # 3. 强制对齐（提高时间戳精度）
            if self.align_model and segments:
                logger.info("进行强制对齐...")
                result = whisperx.align(
                    result["segments"], 
                    self.align_model, 
                    self.metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=False
                )
                logger.info("✅ 强制对齐完成")
            
            # 4. 说话人识别
            speaker_success = False
            if self.diarize_model:
                logger.info("开始说话人识别...")
                try:
                    # 使用正确的API调用方式
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(audio_path)
                    diarization = self.diarize_model({"waveform": waveform, "sample_rate": sample_rate})
                    
                    # 转换diarization结果为WhisperX格式
                    diarize_segments = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        diarize_segments.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        })
                    
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    speaker_success = True
                    logger.info("✅ 说话人识别完成")
                except Exception as e:
                    logger.warning(f"说话人识别失败: {e}")
                    speaker_success = False
            else:
                speaker_success = False
            
            # 5. 处理和格式化结果
            processed_segments = self._format_results(
                result.get("segments", []), 
                dyad_id, 
                conversation_id, 
                speaker_success,
                duration
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 处理完成: {processing_time:.1f}秒")
            
            # 清理GPU内存
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            return processed_segments
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            raise
    
    def _format_results(self, segments, dyad_id, conversation_id, speaker_success, total_duration):
        """格式化转录结果"""
        processed = []
        
        for i, seg in enumerate(segments):
            # 处理说话人信息
            if speaker_success and "speaker" in seg:
                speaker_raw = seg["speaker"]
                # 简化说话人标识
                if "SPEAKER_00" in speaker_raw:
                    speaker_name = "Speaker_A"
                elif "SPEAKER_01" in speaker_raw:
                    speaker_name = "Speaker_B"
                else:
                    speaker_name = f"Speaker_{speaker_raw.split('_')[-1]}"
            else:
                # 简单交替分配
                speaker_name = f"Speaker_{'A' if i % 2 == 0 else 'B'}"
                speaker_raw = f"SPEAKER_{i % 2:02d}"
            
            # 获取时间戳
            start_seconds = seg.get('start', 0)
            end_seconds = seg.get('end', 0)
            duration = end_seconds - start_seconds
            
            # 转换为指定格式
            start_time = self._seconds_to_timestamp(start_seconds)
            finish_time = self._seconds_to_timestamp(end_seconds)
            
            # 获取文本和置信度
            text = seg.get('text', '').strip()
            confidence = seg.get('confidence', 0.0)
            
            # 统计词级信息
            words = seg.get('words', [])
            word_count = len(text.split()) if text else 0
            
            processed.append({
                'dyad': dyad_id,
                'conversation': conversation_id,
                'segment_id': i + 1,
                'start_time': start_time,  # 新格式：HH:MM:SS,mmm
                'finish_time': finish_time,  # 改名为finish_time
                'duration': round(duration, 3),
                'speaker': speaker_name,
                'speaker_raw': speaker_raw,
                'text': text,
                'confidence': round(confidence, 4),
                'word_count': word_count,
                'language': 'nl',  # 荷兰语
                'model_used': 'large-v3',
                'device_used': self.device,
                'has_ai_speaker_detection': speaker_success
            })
        
        return processed
    
    def cleanup(self):
        """清理模型和内存"""
        if self.model:
            del self.model
        if self.align_model:
            del self.align_model
        if self.diarize_model:
            del self.diarize_model
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("内存清理完成")
