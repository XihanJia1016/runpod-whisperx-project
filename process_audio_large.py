"""
RunPod WhisperX Large-v3 高精度音频处理脚本
半自动种子识别说话人日志系统
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
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# 环境检查和自动修复
def check_and_fix_environment():
    """检查环境并在需要时自动修复"""
    
    missing_packages = []
    version_conflicts = []
    
    try:
        import torch
        import torchvision
        
        torch_version = torch.__version__
        torchvision_version = torchvision.__version__
        
        # 检查版本兼容性
        if not (torch_version.startswith('2.1.') and torchvision_version.startswith('0.16.')):
            version_conflicts.append(f"PyTorch版本冲突: torch={torch_version}, torchvision={torchvision_version}")
    except ImportError as e:
        missing_packages.append(f"PyTorch: {e}")
    
    try:
        import pyannote.audio
    except ImportError as e:
        missing_packages.append(f"pyannote.audio: {e}")
    
    try:
        import whisperx
    except ImportError as e:
        missing_packages.append(f"whisperx: {e}")
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        missing_packages.append(f"scikit-learn: {e}")
    
    if missing_packages or version_conflicts:
        print("❌ 环境检查发现问题:")
        for issue in missing_packages + version_conflicts:
            print(f"  - {issue}")
        
        print("\n🔧 自动修复建议:")
        print("运行以下命令修复环境:")
        print("  python setup_environment.py")
        print("\n或者手动修复:")
        print("  pip install -r requirements_stable.txt")
        
        # 询问是否自动修复
        response = input("\n是否现在自动修复？(y/N): ")
        if response.lower() in ['y', 'yes']:
            print("🔄 开始自动修复...")
            try:
                import setup_environment
                setup_environment.main()
                print("✅ 环境修复完成，请重新运行脚本")
                exit(0)
            except Exception as e:
                print(f"❌ 自动修复失败: {e}")
                print("请手动运行: python setup_environment.py")
                exit(1)
        else:
            print("请先修复环境问题后再运行脚本")
            exit(1)
    else:
        print("✅ 环境检查通过")

# 移除自动环境检查，避免在RunPod中卡住
# check_and_fix_environment()  # 已禁用，防止交互式输入导致卡死

# 禁用TF32以避免精度和兼容性问题
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_cudnn_issue():
    """修复cuDNN库问题"""
    logger.info("🔧 尝试修复cuDNN库问题...")
    
    try:
        # 尝试安装缺失的cuDNN库
        logger.info("🔧 安装cuDNN库...")
        subprocess.run(['apt', 'update'], check=True, capture_output=True)
        subprocess.run(['apt', 'install', '-y', 'libcudnn8', 'libcudnn8-dev'], check=True, capture_output=True)
        logger.info("✅ cuDNN库安装完成")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"⚠️ 无法通过apt安装cuDNN: {e}")
        
        # 尝试使用conda安装
        try:
            logger.info("🔧 尝试通过conda安装cudnn...")
            subprocess.run(['conda', 'install', '-c', 'conda-forge', 'cudnn', '-y'], check=True, capture_output=True)
            logger.info("✅ cuDNN通过conda安装完成")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"⚠️ conda安装也失败: {e}")

def setup_cuda_environment():
    """设置CUDA环境变量修复cuDNN问题"""
    # 先尝试修复cuDNN
    fix_cudnn_issue()
    
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-11.8/lib64", 
        "/usr/local/cuda-11.7/lib64",
        "/usr/local/cuda-11.6/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/opt/conda/lib",
        "/opt/conda/pkgs/cudnn*/lib",
        "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib"
    ]
    
    # 检查PyTorch信息
    logger.info(f"🔧 PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"🔧 CUDA版本: {torch.version.cuda}")
        logger.info(f"🔧 cuDNN版本: {torch.backends.cudnn.version()}")
    
    # 设置LD_LIBRARY_PATH
    current_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_paths = []
    
    for path in cuda_paths:
        if os.path.exists(path):
            new_paths.append(path)
            logger.info(f"✅ 找到CUDA库路径: {path}")
    
    if new_paths:
        if current_path:
            os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths) + ':' + current_path
        else:
            os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)
        logger.info(f"🔧 已设置LD_LIBRARY_PATH")
    
    # 设置额外的环境变量来稳定cuDNN
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # 强制重新加载动态库
    try:
        import ctypes
        ctypes.CDLL("libcudnn.so.8", mode=ctypes.RTLD_GLOBAL)
        logger.info("✅ 成功加载libcudnn.so.8")
    except Exception as e:
        logger.warning(f"⚠️ 无法加载libcudnn.so.8: {e}")
    
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
        # 设置CUDA环境修复cuDNN问题
        setup_cuda_environment()
        # 确保ffmpeg已安装
        ensure_ffmpeg()
        self.device = self._setup_device()
        self.model = None
        self.align_model = None
        self.metadata = None
        self.diarize_model = None
        self.embedding_model = None
        
        # 高精度配置
        self.config = {
            'model_size': 'large-v3',
            'batch_size': 8,  # GPU优化
            'chunk_length': 30,  # 30秒块，平衡精度和内存
            'return_attention': True,
            'word_timestamps': True,
            'vad_filter': True,
            'temperature': 0.0,  # 确定性输出
            'enable_speaker_diarization': True,  # 启用说话人识别
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
            try:
                self.align_model, self.metadata = whisperx.load_align_model(
                    language_code="nl",  # 荷兰语
                    device=self.device
                )
                logger.info("✅ 对齐模型加载完成")
            except Exception as e:
                logger.warning(f"荷兰语对齐模型加载失败: {e}")
                logger.info("尝试使用英语对齐模型作为备选...")
                try:
                    self.align_model, self.metadata = whisperx.load_align_model(
                        language_code="en",  # 英语作为备选
                        device=self.device
                    )
                    logger.info("✅ 英语对齐模型加载完成（备选方案）")
                except Exception as e2:
                    logger.warning(f"对齐模型加载完全失败: {e2}")
                    logger.info("将跳过对齐步骤，使用原始时间戳")
                    self.align_model = None
                    self.metadata = None
            
            # 3. 说话人嵌入模型将采用"用时加载，用完即毁"策略
            logger.info("✅ 说话人嵌入模型将动态加载（用时加载，用完即毁）")
            
            # 验证HuggingFace token
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                logger.warning("⚠️ 未设置HF_TOKEN环境变量，说话人嵌入可能失败")
                logger.info("💡 请运行: export HF_TOKEN='your_token_here'")
            else:
                logger.info(f"🔑 HF_TOKEN已设置: {hf_token[:20]}...")
            
            # 不再预加载embedding模型，改为动态加载
            self.embedding_model = None
            self.diarize_model = None
            
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
    
    def _clean_text_for_comparison(self, text):
        """清理文本以便更好地进行比较"""
        if pd.isna(text):
            return ""
        import re
        return re.sub(r'\s+', ' ', str(text).strip().lower())
    
    def _force_align_initial_turns(self, golden_turns_df, all_segments, num_turns=3):
        """
        阶段一：强制分配开场轮次的说话人
        
        Args:
            golden_turns_df: 黄金文本DataFrame
            all_segments: 所有AI转录片段列表
            num_turns: 强制分配的轮次数，默认3轮
            
        Returns:
            tuple: (processed_segments, remaining_segments, success)
                - processed_segments: 已强制分配好说话人的片段列表
                - remaining_segments: 剩余未处理的片段列表  
                - success: 此阶段是否成功
        """
        logger.info(f"🎯 强制分配前{num_turns}轮说话人...")
        
        processed_segments = []
        cursor = 0  # AI片段的指针
        success = True
        
        try:
            # 确保有足够的轮次可以处理
            actual_turns = min(num_turns, len(golden_turns_df))
            logger.info(f"实际处理轮次: {actual_turns}")
            
            # 循环处理每一轮
            for i in range(actual_turns):
                golden_turn = golden_turns_df.iloc[i]
                speaker = golden_turn['role']
                text = golden_turn['text']
                
                logger.info(f"处理第{i+1}轮 - 说话人: {speaker}")
                
                # 从当前cursor位置开始寻找匹配的AI片段
                available_segments = all_segments[cursor:]
                
                if not available_segments:
                    logger.warning(f"第{i+1}轮: 没有剩余的AI片段可匹配")
                    success = False
                    break
                
                # 使用现有的贪心对齐逻辑找到最匹配的片段组合
                matched_segments = self._find_best_matching_segments(text, available_segments)
                
                if not matched_segments:
                    logger.warning(f"第{i+1}轮: 未找到匹配的AI片段")
                    success = False
                    break
                
                # 强制分配说话人
                for segment in matched_segments:
                    segment['speaker'] = speaker
                    segment['confidence'] = 1.0  # 基于黄金标准，置信度最高
                
                # 添加到已处理列表
                processed_segments.extend(matched_segments)
                
                # 更新cursor到最后一个匹配片段的下一个位置
                last_matched_idx = None
                for j, seg in enumerate(all_segments):
                    if seg in matched_segments:
                        last_matched_idx = j
                
                if last_matched_idx is not None:
                    cursor = last_matched_idx + 1
                else:
                    # 如果没找到索引，保守地只移动1位
                    cursor += len(matched_segments)
                
                logger.info(f"✅ 第{i+1}轮完成: 分配{len(matched_segments)}个片段给{speaker}, cursor移至{cursor}")
            
            # 计算剩余片段
            remaining_segments = all_segments[cursor:] if cursor < len(all_segments) else []
            
            logger.info(f"🎯 强制分配阶段完成: 处理了{len(processed_segments)}个片段, 剩余{len(remaining_segments)}个片段")
            return processed_segments, remaining_segments, success
            
        except Exception as e:
            logger.error(f"强制分配阶段失败: {e}")
            return [], all_segments, False

    def _find_seed_segments(self, golden_turns_df, ai_segments):
        """
        使用质量优先策略根据黄金文本找到每个说话人的种子片段
        
        Args:
            golden_turns_df: 已筛选的黄金文本DataFrame (当前dyad和conversation)
            ai_segments: 当前对话的所有AI转录片段列表
            
        Returns:
            dict: {'S': [segment1, segment2, ...], 'L': [segment3, segment4, ...]}
        """
        logger.info("开始寻找说话人种子片段（质量优先策略）...")
        
        seed_map = {'S': [], 'L': []}
        
        try:
            # 找到S和L的轮次
            s_turns = golden_turns_df[golden_turns_df['role'] == 'S']
            l_turns = golden_turns_df[golden_turns_df['role'] == 'L']
            
            if s_turns.empty or l_turns.empty:
                logger.warning("未找到S或L的黄金文本，无法生成种子")
                return seed_map
            
            # 为S找最佳质量种子
            logger.info("🔍 分析S说话人的候选轮次...")
            s_best_segments = self._find_best_quality_seed(s_turns, ai_segments, 'S')
            if s_best_segments:
                seed_map['S'] = s_best_segments
            
            # 为L找最佳质量种子 (排除已用于S的片段)
            logger.info("🔍 分析L说话人的候选轮次...")
            remaining_segments = [seg for seg in ai_segments if seg not in s_best_segments]
            l_best_segments = self._find_best_quality_seed(l_turns, remaining_segments, 'L')
            if l_best_segments:
                seed_map['L'] = l_best_segments
            
            logger.info(f"🌱 最终种子选择结果: S={len(seed_map['S'])}个片段, L={len(seed_map['L'])}个片段")
            return seed_map
            
        except Exception as e:
            logger.error(f"寻找种子片段失败: {e}")
            return {'S': [], 'L': []}
    
    def _find_best_quality_seed(self, speaker_turns, available_segments, speaker_name):
        """
        使用质量优先策略为指定说话人找到最佳种子片段
        
        Args:
            speaker_turns: 该说话人的黄金文本轮次
            available_segments: 可用的AI片段
            speaker_name: 说话人名称（用于日志）
            
        Returns:
            list: 最佳质量的种子片段列表
        """
        # 定义候选范围：前5个轮次
        MAX_CANDIDATES = min(5, len(speaker_turns))
        candidate_turns = speaker_turns.head(MAX_CANDIDATES)
        
        logger.info(f"   {speaker_name}说话人有{len(speaker_turns)}个轮次，分析前{MAX_CANDIDATES}个候选")
        
        best_quality_score = -1
        best_segments = []
        best_turn_text = ""
        
        # 遍历每个候选轮次
        for idx, (_, turn) in enumerate(candidate_turns.iterrows()):
            turn_text = self._clean_text_for_comparison(turn['text'])
            turn_length = len(turn_text)
            
            # 使用贪心对齐找到对应的AI片段
            matched_segments = self._find_best_matching_segments(turn_text, available_segments)
            
            if matched_segments:
                # 计算质量分：平均单词置信度
                quality_score = self._calculate_quality_score(matched_segments)
                
                logger.info(f"   候选{idx+1}: 文本长度={turn_length}, 匹配片段={len(matched_segments)}个, "
                           f"质量分={quality_score:.4f}")
                logger.info(f"     文本: '{turn_text[:50]}...'")
                
                # 选择质量分最高的
                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_segments = matched_segments
                    best_turn_text = turn_text
                    best_candidate_idx = idx + 1
            else:
                logger.info(f"   候选{idx+1}: 文本长度={turn_length}, 匹配片段=0个, 质量分=0.0000")
                logger.info(f"     文本: '{turn_text[:50]}...'")
        
        if best_segments:
            logger.info(f"✅ {speaker_name}最佳种子选择: 候选{best_candidate_idx}, "
                       f"质量分={best_quality_score:.4f}, 片段数={len(best_segments)}")
            logger.info(f"   最佳种子文本: '{best_turn_text[:50]}...'")
            for i, seg in enumerate(best_segments[:3]):  # 显示前3个片段
                logger.info(f"   种子片段{i+1}: '{seg.get('text', '')[:30]}...'")
        else:
            logger.warning(f"❌ {speaker_name}未找到有效的种子片段")
        
        return best_segments
    
    def _calculate_quality_score(self, segments):
        """
        计算AI片段组合的平均单词置信度质量分
        
        Args:
            segments: AI片段列表
            
        Returns:
            float: 平均单词置信度 (0-1之间)
        """
        total_score = 0.0
        total_words = 0
        
        for segment in segments:
            words = segment.get('words', [])
            for word in words:
                # 尝试多种可能的置信度字段名
                score = (word.get('score') or 
                        word.get('probability') or 
                        word.get('confidence') or 
                        0.0)
                total_score += score
                total_words += 1
        
        if total_words == 0:
            # 如果没有单词级信息，使用片段级置信度
            segment_scores = []
            for segment in segments:
                seg_score = (segment.get('avg_logprob') or 
                           segment.get('confidence') or 
                           0.0)
                if seg_score < 0:  # logprob转换为概率
                    seg_score = max(0.0, min(1.0, (seg_score + 1.0)))
                segment_scores.append(seg_score)
            
            return sum(segment_scores) / len(segment_scores) if segment_scores else 0.0
        
        return total_score / total_words
    
    def _get_embedding_with_fresh_model(self, audio_data, segments_to_embed):
        """
        使用"用时加载，用完即毁"策略生成嵌入向量
        
        Args:
            audio_data: Numpy数组格式的完整音频
            segments_to_embed: 包含一个或多个片段字典的列表
            
        Returns:
            numpy.ndarray: 平均嵌入向量，失败时返回None
        """
        embedding_model = None
        
        try:
            # 动态加载现代嵌入模型
            logger.info("🔄 动态加载SpeechBrain嵌入模型...")
            
            # 从 speechbrain 加载一个强大的、兼容性好的说话人嵌入模型
            from speechbrain.inference.speaker import EncoderClassifier
            
            embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join('/workspace/cache', 'speechbrain_models'),
                run_opts={"device": self.device}  # 直接在加载时指定设备
            )
            
            logger.info(f"✅ 嵌入模型动态加载完成，处理{len(segments_to_embed)}个片段")
            
            # 生成嵌入向量
            embeddings = []
            sample_rate = 16000
            
            for i, segment in enumerate(segments_to_embed):
                try:
                    # 提取音频片段
                    start_sample = int(segment.get('start', 0) * sample_rate)
                    end_sample = int(segment.get('end', 0) * sample_rate)
                    
                    # 确保索引有效
                    start_sample = max(0, start_sample)
                    end_sample = min(len(audio_data), end_sample)
                    
                    if start_sample >= end_sample:
                        continue
                    
                    audio_segment = audio_data[start_sample:end_sample]
                    
                    # 确保音频长度足够 (至少0.1秒)
                    min_length = int(0.1 * sample_rate)
                    if len(audio_segment) < min_length:
                        audio_segment = np.pad(audio_segment, (0, min_length - len(audio_segment)))
                    
                    # 转换为PyTorch tensor并立即发送到正确的设备
                    audio_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0).to(self.device)
                    
                    # 生成嵌入 - speechbrain 模型直接接收音频张量和其相对长度
                    with torch.no_grad():
                        wav_lens = torch.tensor([1.0], device=self.device)  # 1.0 表示使用完整长度
                        embedding = embedding_model.encode_batch(audio_tensor, wav_lens=wav_lens)
                    
                    # 转换为numpy - 移除所有大小为1的维度，然后展平
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.squeeze().cpu().numpy()
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                        
                except Exception as e:
                    logger.warning(f"片段{i}嵌入生成失败: {e}")
                    continue
            
            if embeddings:
                # 计算平均嵌入
                mean_embedding = np.mean(embeddings, axis=0)
                logger.info(f"✅ 成功生成平均嵌入向量，形状: {mean_embedding.shape}")
                return mean_embedding
            else:
                logger.warning("❌ 没有成功生成任何嵌入向量")
                return None
                
        except Exception as e:
            logger.error(f"动态加载嵌入模型失败: {e}")
            return None
            
        finally:
            # 无论成功或失败都要清理模型
            if embedding_model is not None:
                logger.info("🗑️ 清理嵌入模型...")
                del embedding_model
                
                # 清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("✅ 嵌入模型已卸载和清理")
    
    def _generate_single_embedding_with_model(self, audio_segment, embedding_model):
        """
        使用已加载的模型为单个音频片段生成嵌入向量
        
        Args:
            audio_segment: 音频片段 (numpy array)
            embedding_model: 已加载的嵌入模型
            
        Returns:
            numpy.ndarray: 嵌入向量，如果失败则返回None
        """
        try:
            # 确保音频长度足够 (至少0.1秒)
            min_length = int(0.1 * 16000)
            if len(audio_segment) < min_length:
                # 如果太短，用零填充
                audio_segment = np.pad(audio_segment, (0, min_length - len(audio_segment)))
            
            # 转换为PyTorch tensor并立即发送到正确的设备
            audio_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0).to(self.device)
            
            # 生成嵌入 - speechbrain 模型调用方式
            with torch.no_grad():
                wav_lens = torch.tensor([1.0], device=self.device)  # 1.0 表示使用完整长度
                embedding = embedding_model.encode_batch(audio_tensor, wav_lens=wav_lens)
            
            # 转换为numpy - 移除所有大小为1的维度，然后展平
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            logger.warning(f"单个嵌入生成失败: {e}")
            return None
    
    def _find_best_matching_segments(self, target_text, ai_segments):
        """
        使用贪心算法找到与目标文本最匹配的AI片段组合
        
        Args:
            target_text: 清理后的目标文本
            ai_segments: 可用的AI片段列表
            
        Returns:
            list: 最佳匹配的片段列表
        """
        if not target_text or not ai_segments:
            return []
        
        best_match_ratio = 0
        best_match_segments = []
        
        # 贪心搜索：尝试不同的片段组合
        for start_idx in range(len(ai_segments)):
            temp_text = ""
            temp_segments = []
            
            for end_idx in range(start_idx, min(start_idx + 5, len(ai_segments))):  # 最多组合5个片段
                segment = ai_segments[end_idx]
                temp_segments.append(segment)
                temp_text += " " + self._clean_text_for_comparison(segment.get('text', ''))
                temp_text = temp_text.strip()
                
                # 计算相似度
                if temp_text:
                    similarity = difflib.SequenceMatcher(None, temp_text, target_text).ratio()
                    
                    if similarity > best_match_ratio:
                        best_match_ratio = similarity
                        best_match_segments = temp_segments.copy()
                    
                    # 如果相似度开始下降，提前停止
                    if len(temp_segments) > 1 and similarity < best_match_ratio * 0.8:
                        break
        
        logger.info(f"最佳匹配相似度: {best_match_ratio:.3f}")
        return best_match_segments
    
    def perform_seed_based_diarization(self, audio_data, all_ai_segments, seed_map):
        """
        基于种子片段进行说话人识别的核心函数
        
        Args:
            audio_data: whisperx.load_audio()返回的numpy数组
            all_ai_segments: 当前对话的所有AI转录片段
            seed_map: 种子字典 {'S': [...], 'L': [...]}
            
        Returns:
            tuple: (更新了speaker字段的all_ai_segments, 成功标志boolean)
        """
        logger.info("开始基于种子的说话人识别...")
        
        try:
            # 1. 使用动态加载策略生成种子指纹
            logger.info("步骤1: 生成S说话人的种子指纹...")
            s_seed_embedding = self._get_embedding_with_fresh_model(audio_data, seed_map['S'])
            
            logger.info("步骤2: 生成L说话人的种子指纹...")
            l_seed_embedding = self._get_embedding_with_fresh_model(audio_data, seed_map['L'])
            
            if s_seed_embedding is None or l_seed_embedding is None:
                logger.error("无法生成一个或两个种子嵌入，跳过说话人识别")
                return all_ai_segments, False
            
            # --- 种子自检逻辑 ---
            seeds_similarity = cosine_similarity(
                s_seed_embedding.reshape(1, -1),
                l_seed_embedding.reshape(1, -1)
            )[0][0]
            
            logger.info(f"🔍 种子自检：S和L的种子指纹相似度为 {seeds_similarity:.4f}")
            
            # 如果两个种子过于相似，则没有继续下去的意义
            SIMILARITY_THRESHOLD = 0.85  # 这是一个可以调整的阈值
            if seeds_similarity > SIMILARITY_THRESHOLD:
                logger.error(f"❌ 种子过于相似 (相似度>{SIMILARITY_THRESHOLD})，无法区分说话人。请检查种子选择逻辑或音频质量。")
                # 直接返回，标记所有为UNKNOWN并设置失败
                for segment in all_ai_segments:
                    segment['speaker'] = 'UNKNOWN'
                    segment['confidence'] = 0.0
                return all_ai_segments, False
            # --- 自检逻辑结束 ---
            
            logger.info("✅ 种子嵌入生成完成，种子差异充足")
            
            # 3. 为主要识别流程预加载一个"干净"的模型实例
            logger.info("步骤3: 为主要识别流程动态加载模型...")
            main_embedding_model = None
            
            try:
                # 使用 speechbrain 现代嵌入模型
                from speechbrain.inference.speaker import EncoderClassifier
                
                main_embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=os.path.join('/workspace/cache', 'speechbrain_models'),
                    run_opts={"device": self.device}  # 直接在加载时指定设备
                )
                
                logger.info("✅ 主要识别模型加载完成")
                
                # 4. 识别所有片段
                sample_rate = 16000  # WhisperX使用16kHz
                
                for i, segment in enumerate(all_ai_segments):
                    try:
                        # 提取音频片段
                        start_sample = int(segment.get('start', 0) * sample_rate)
                        end_sample = int(segment.get('end', 0) * sample_rate)
                        
                        # 确保索引有效
                        start_sample = max(0, start_sample)
                        end_sample = min(len(audio_data), end_sample)
                        
                        if start_sample >= end_sample:
                            logger.warning(f"片段{i}时间戳无效，跳过")
                            segment['speaker'] = 'UNKNOWN'
                            continue
                        
                        audio_segment = audio_data[start_sample:end_sample]
                        
                        # 使用预加载的模型生成片段嵌入
                        segment_embedding = self._generate_single_embedding_with_model(audio_segment, main_embedding_model)
                        
                        if segment_embedding is not None:
                            # 计算与种子的相似度
                            s_similarity = cosine_similarity(
                                segment_embedding.reshape(1, -1), 
                                s_seed_embedding.reshape(1, -1)
                            )[0][0]
                            
                            l_similarity = cosine_similarity(
                                segment_embedding.reshape(1, -1), 
                                l_seed_embedding.reshape(1, -1)
                            )[0][0]
                            
                            # 重构的说话人分配逻辑 - 分步决策流程
                            
                            # 1. 定义阈值
                            MIN_CONFIDENCE_THRESHOLD = 0.45  # 稍微降低最小置信度
                            MIN_DIFFERENCE_THRESHOLD = 0.08  # 稍微降低差距要求
                            
                            confidence = 0.0
                            assigned_speaker = 'UNKNOWN'
                            
                            # 3. 确定胜出方
                            if s_similarity > l_similarity:
                                winner = 'S'
                                winner_score = s_similarity
                                loser_score = l_similarity
                            else:
                                winner = 'L'
                                winner_score = l_similarity
                                loser_score = s_similarity
                            
                            # 4. 分步验证结果是否可信
                            # 条件1: 胜出方的分数是否达到了最低要求？
                            is_confident_enough = winner_score >= MIN_CONFIDENCE_THRESHOLD
                            
                            # 条件2: 胜出方和失败方的分数差距是否足够大？
                            is_distinct_enough = (winner_score - loser_score) >= MIN_DIFFERENCE_THRESHOLD
                            
                            # 只有当两个条件都满足时，我们才接受这个结果
                            if is_confident_enough and is_distinct_enough:
                                assigned_speaker = winner
                                confidence = winner_score
                            else:
                                # 否则，即使有一方分数更高，我们依然认为结果不可靠
                                assigned_speaker = 'UNKNOWN'
                                confidence = winner_score  # 依然可以记录最高分，但标签是UNKNOWN
                            
                            segment['speaker'] = assigned_speaker
                            segment['confidence'] = float(confidence)
                            
                            # 调试日志 - 显示前几个片段的详细信息
                            if i < 5:
                                if assigned_speaker != 'UNKNOWN':
                                    logger.info(f"✅ 片段 {i}: S={s_similarity:.3f}, L={l_similarity:.3f}, "
                                               f"胜出={winner}({winner_score:.3f}), 差距={winner_score-loser_score:.3f}, "
                                               f"分配={assigned_speaker}")
                                else:
                                    reason = []
                                    if not is_confident_enough:
                                        reason.append(f"置信度不足({winner_score:.3f}<{MIN_CONFIDENCE_THRESHOLD})")
                                    if not is_distinct_enough:
                                        reason.append(f"差距不够({winner_score-loser_score:.3f}<{MIN_DIFFERENCE_THRESHOLD})")
                                    
                                    logger.warning(f"❌ 片段 {i}: S={s_similarity:.3f}, L={l_similarity:.3f}, "
                                                 f"胜出={winner}({winner_score:.3f}), 标记=UNKNOWN, "
                                                 f"原因: {', '.join(reason)}")
                            
                        else:
                            logger.warning(f"片段{i}无法生成嵌入，使用默认标记")
                            segment['speaker'] = 'UNKNOWN'
                            segment['confidence'] = None
                        
                    except Exception as e:
                        logger.warning(f"处理片段{i}失败: {e}")
                        segment['speaker'] = 'UNKNOWN'
                        segment['confidence'] = None
                
                # 统计结果和相似度分布
                s_count = sum(1 for seg in all_ai_segments if seg.get('speaker') == 'S')
                l_count = sum(1 for seg in all_ai_segments if seg.get('speaker') == 'L')
                unknown_count = sum(1 for seg in all_ai_segments if seg.get('speaker') == 'UNKNOWN')
                
                # 统计置信度分布（用于调试门槛设置）
                confidences = [seg.get('confidence', 0) for seg in all_ai_segments if seg.get('confidence') is not None]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    max_confidence = max(confidences)
                    min_confidence = min(confidences)
                    logger.info(f"置信度统计: 平均={avg_confidence:.3f}, 最高={max_confidence:.3f}, 最低={min_confidence:.3f}")
                
                logger.info(f"说话人识别结果: S={s_count}, L={l_count}, Unknown={unknown_count}")
                
                # 检查是否有足够的成功识别
                success_rate = (s_count + l_count) / len(all_ai_segments) if all_ai_segments else 0
                success = success_rate > 0.5  # 超过50%成功才算成功
                
                logger.info(f"识别成功率: {success_rate:.2%}, 整体状态: {'成功' if success else '失败'}")
                return all_ai_segments, success
                
            finally:
                # 确保主要识别模型被清理
                if main_embedding_model is not None:
                    logger.info("🗑️ 清理主要识别模型...")
                    del main_embedding_model
                    
                    # 清理CUDA缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info("✅ 主要识别模型已卸载和清理")
                
        except Exception as e:
            logger.error(f"基于种子的说话人识别失败: {e}")
            return all_ai_segments, False
    
    def _generate_seed_embedding(self, audio_data, seed_segments):
        """
        为种子片段生成平均嵌入向量
        
        Args:
            audio_data: 完整音频数据
            seed_segments: 种子片段列表
            
        Returns:
            numpy.ndarray: 平均嵌入向量，如果失败则返回None
        """
        if not seed_segments:
            return None
        
        sample_rate = 16000
        embeddings = []
        
        for segment in seed_segments:
            try:
                start_sample = int(segment.get('start', 0) * sample_rate)
                end_sample = int(segment.get('end', 0) * sample_rate)
                
                # 确保索引有效
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                if start_sample >= end_sample:
                    continue
                
                audio_segment = audio_data[start_sample:end_sample]
                embedding = self._generate_single_embedding(audio_segment)
                
                if embedding is not None:
                    embeddings.append(embedding)
                    
            except Exception as e:
                logger.warning(f"种子片段嵌入生成失败: {e}")
                continue
        
        if embeddings:
            # 计算平均嵌入
            mean_embedding = np.mean(embeddings, axis=0)
            return mean_embedding
        else:
            return None
    
    def _generate_single_embedding(self, audio_segment):
        """
        为单个音频片段生成嵌入向量
        
        Args:
            audio_segment: 音频片段 (numpy array)
            
        Returns:
            numpy.ndarray: 嵌入向量，如果失败则返回None
        """
        try:
            # 确保音频长度足够 (至少0.1秒)
            min_length = int(0.1 * 16000)
            if len(audio_segment) < min_length:
                # 如果太短，用零填充
                audio_segment = np.pad(audio_segment, (0, min_length - len(audio_segment)))
            
            # 转换为PyTorch tensor
            audio_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0)
            
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # 生成嵌入 - 使用Model的正确调用方式
            with torch.no_grad():
                # 直接传递tensor给模型，不使用dict格式
                # pyannote Model期望直接接收waveform tensor
                embedding = self.embedding_model(audio_tensor)
            
            # 转换为numpy
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            return embedding.flatten()
            
        except Exception as e:
            logger.warning(f"生成单个嵌入失败: {e}")
            return None
    
    def process_single_file(self, audio_path, dyad_id, conversation_id, golden_turns_df=None):
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
            
            # 4. 新混合策略三阶段说话人识别流程
            speaker_success = False
            if golden_turns_df is not None and not golden_turns_df.empty:
                
                # --- 阶段一：强制分配开场（前3轮） ---
                logger.info(">> 阶段一：强制分配前3轮说话人...")
                # 调用一个新函数来处理这个逻辑，它会返回已被分配好说话人的片段，以及剩余未分配的片段
                all_segments = result["segments"]
                processed_segments, remaining_segments, success_stage1 = self._force_align_initial_turns(
                    golden_turns_df, 
                    all_segments,
                    num_turns=3  # 指定强制分配的轮次数
                )

                # --- 阶段二 和 阶段三 ---
                if success_stage1 and remaining_segments:
                    logger.info(">> 阶段二：从后续轮次中智能选择种子...")

                    # 种子选择范围从第4轮开始 (因为前3轮已用掉)
                    seed_candidate_turns = golden_turns_df.iloc[3:]

                    # 调用 _find_seed_segments，但只在候选轮次和剩余片段中寻找
                    seed_map = self._find_seed_segments(seed_candidate_turns, remaining_segments)

                    if seed_map.get('S') and seed_map.get('L'):
                        logger.info(">> 阶段三：对剩余片段进行种子识别...")

                        # 调用 perform_seed_based_diarization，但只处理剩余的片段
                        diarized_remaining_segments, success_stage3 = self.perform_seed_based_diarization(
                            audio,
                            remaining_segments,
                            seed_map
                        )

                        # 合并结果
                        final_segments = processed_segments + diarized_remaining_segments
                        speaker_success = True
                    else:
                        logger.warning("未能从后续轮次中找到足够的种子，剩余片段将使用回退方案。")
                        # 对剩余部分使用回退方案
                        for i, seg in enumerate(remaining_segments):
                            seg['speaker'] = 'UNKNOWN'  # 或 A/B 轮换
                        final_segments = processed_segments + remaining_segments
                        speaker_success = False  # 整体不算完全成功
                else:
                    logger.warning("阶段一失败或没有剩余片段，直接使用阶段一的结果。")
                    final_segments = processed_segments
                    speaker_success = success_stage1

                result["segments"] = final_segments

            else:
                logger.warning("未提供黄金文本，跳过说话人识别")
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
        
        # 调试：记录第一个segment的结构
        if segments and len(segments) > 0:
            logger.info(f"🔍 第一个segment包含的字段: {list(segments[0].keys())}")
        
        for i, seg in enumerate(segments):
            # 处理说话人信息
            if speaker_success and "speaker" in seg:
                speaker_raw = seg["speaker"]
                # 处理新的S/L标识系统
                if speaker_raw == 'S':
                    speaker_name = "Speaker_A"
                    speaker_raw = "SPEAKER_00"
                elif speaker_raw == 'L':
                    speaker_name = "Speaker_B"
                    speaker_raw = "SPEAKER_01"
                elif "SPEAKER_00" in str(speaker_raw):
                    speaker_name = "Speaker_A"
                elif "SPEAKER_01" in str(speaker_raw):
                    speaker_name = "Speaker_B"
                else:
                    # 处理其他情况
                    speaker_name = f"Speaker_{str(speaker_raw).split('_')[-1]}"
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
            
            # 计算置信度（从多个可能的源）
            confidence = 0.0
            if 'confidence' in seg and seg['confidence'] is not None:
                confidence = seg['confidence']
            elif 'avg_logprob' in seg:
                # 将logprob转换为置信度近似值
                confidence = min(1.0, max(0.0, (seg['avg_logprob'] + 1.0)))
            elif 'words' in seg and seg['words']:
                # 从词级置信度计算平均值
                word_confidences = [w.get('probability', 0.0) for w in seg['words'] if 'probability' in w]
                if word_confidences:
                    confidence = sum(word_confidences) / len(word_confidences)
                else:
                    # 如果没有任何置信度信息，根据无停顿时间估算
                    confidence = 0.85 if seg.get('no_speech_prob', 1.0) < 0.5 else 0.3
            else:
                # 默认合理置信度（而非0）
                confidence = 0.8
            
            # 确保confidence是有效数值
            if confidence is None:
                confidence = 0.5  # 设置合理的默认值
            
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
        if self.embedding_model:
            del self.embedding_model
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("内存清理完成")


def load_golden_text_data(golden_text_path):
    """加载黄金标准文本数据"""
    try:
        logger.info(f"加载黄金文本数据: {golden_text_path}")
        df = pd.read_csv(golden_text_path)
        df.columns = df.columns.str.strip()  # 清理列名
        
        # 确保必要的列存在
        required_cols = ['dyad', 'conversation', 'role', 'text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"黄金文本缺少必要列: {missing_cols}")
            logger.info(f"可用列: {list(df.columns)}")
            return None
        
        logger.info(f"✅ 黄金文本加载完成: {len(df)}行数据")
        logger.info(f"包含对话: {df['dyad'].nunique()}个dyad, {df['conversation'].nunique()}个conversation")
        
        return df
        
    except Exception as e:
        logger.error(f"加载黄金文本失败: {e}")
        return None


def process_conversations_with_golden_text(
    audio_dir, 
    golden_text_path, 
    output_dir, 
    conversation_mapping=None
):
    """
    使用黄金文本数据批量处理对话
    
    Args:
        audio_dir: 音频文件目录
        golden_text_path: 黄金文本CSV文件路径
        output_dir: 输出目录
        conversation_mapping: 音频文件名到(dyad, conversation)的映射字典
                            如果为None，将尝试从文件名解析
    """
    
    # 加载黄金文本
    golden_df = load_golden_text_data(golden_text_path)
    if golden_df is None:
        logger.error("无法加载黄金文本，终止处理")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化处理器
    processor = HighPrecisionAudioProcessor()
    
    try:
        # 加载模型
        if not processor.load_models():
            logger.error("模型加载失败，终止处理")
            return
        
        # 获取音频文件列表
        audio_files = []
        for ext in ['.wav', '.mp3', '.m4a', '.flac']:
            audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        all_results = []
        processed_count = 0
        
        for audio_file in audio_files:
            try:
                # 解析dyad和conversation
                if conversation_mapping:
                    if audio_file.name in conversation_mapping:
                        dyad_id, conversation_id = conversation_mapping[audio_file.name]
                    else:
                        logger.warning(f"文件 {audio_file.name} 不在映射中，跳过")
                        continue
                else:
                    # 尝试从文件名解析，支持多种格式
                    try:
                        # 格式1: dyad_X_conversation_Y.mp3
                        if '_' in audio_file.stem:
                            parts = audio_file.stem.split('_')
                            dyad_id = int(parts[1])
                            conversation_id = int(parts[3])
                        # 格式2: X.Y.mp3 (dyad.conversation.mp3)
                        elif '.' in audio_file.stem:
                            parts = audio_file.stem.split('.')
                            dyad_id = int(parts[0])
                            conversation_id = int(parts[1])
                        else:
                            raise ValueError("无法识别的文件名格式")
                            
                        logger.info(f"📁 解析文件 {audio_file.name} -> dyad:{dyad_id}, conversation:{conversation_id}")
                    except (ValueError, IndexError):
                        logger.warning(f"无法从文件名解析dyad和conversation: {audio_file.name}")
                        logger.info("支持的格式: dyad_X_conversation_Y.mp3 或 X.Y.mp3")
                        continue
                
                # 过滤对应的黄金文本
                golden_turns_df = golden_df[
                    (golden_df['dyad'] == dyad_id) & 
                    (golden_df['conversation'] == conversation_id)
                ].copy()
                
                if golden_turns_df.empty:
                    logger.warning(f"对话 {dyad_id}-{conversation_id} 没有对应的黄金文本，跳过")
                    continue
                
                logger.info(f"处理对话 {dyad_id}-{conversation_id}: {audio_file.name}")
                logger.info(f"黄金文本轮次: {len(golden_turns_df)}")
                
                # 处理音频文件
                segments = processor.process_single_file(
                    str(audio_file), 
                    dyad_id, 
                    conversation_id, 
                    golden_turns_df
                )
                
                all_results.extend(segments)
                processed_count += 1
                
                logger.info(f"✅ 完成处理 {dyad_id}-{conversation_id}: {len(segments)}个片段")
                
            except Exception as e:
                logger.error(f"❌ 处理文件 {audio_file.name} 失败: {e}")
                continue
        
        # 保存结果
        if all_results:
            output_file = os.path.join(output_dir, "combined_transcription_with_seed_diarization.csv")
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"✅ 所有结果已保存到: {output_file}")
            logger.info(f"总计处理: {processed_count}个对话, {len(all_results)}个片段")
            
            # 输出统计信息
            if 'has_ai_speaker_detection' in results_df.columns:
                success_count = results_df['has_ai_speaker_detection'].sum()
                logger.info(f"说话人识别成功率: {success_count}/{processed_count} ({success_count/processed_count*100:.1f}%)")
        else:
            logger.warning("没有成功处理任何文件")
    
    finally:
        # 清理
        processor.cleanup()


def main():
    """主函数 - RunPod使用示例"""
    
    print("🚀 WhisperX半自动种子识别说话人日志系统")
    print("=" * 50)
    
    # 跳过耗时的环境检查，直接开始
    print("⚡ 开始处理...")
    
    # 默认RunPod路径配置
    audio_directory = "/workspace/input"
    golden_text_file = "/workspace/input/text_data_output.csv"
    output_directory = "/workspace/output"
    
    print(f"📁 音频文件目录: {audio_directory}")
    print(f"📄 黄金文本文件: {golden_text_file}")
    print(f"📤 输出目录: {output_directory}")
    
    # 检查必要文件和目录
    if not os.path.exists(audio_directory):
        print(f"❌ 音频目录不存在: {audio_directory}")
        print("请将音频文件(.wav 或 .mp3)放在 /workspace/input/ 目录")
        return
    
    if not os.path.exists(golden_text_file):
        print(f"❌ 黄金文本文件不存在: {golden_text_file}")
        print("请将 text_data_output.csv 文件放在 /workspace/input/ 目录")
        return
    
    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)
    
    # 检查音频文件 (支持wav和mp3格式)
    wav_files = list(Path(audio_directory).glob("*.wav"))
    mp3_files = list(Path(audio_directory).glob("*.mp3"))
    audio_files = wav_files + mp3_files
    
    if not audio_files:
        print(f"❌ 在 {audio_directory} 中没有找到音频文件(.wav 或 .mp3)")
        return
    
    print(f"✅ 找到音频文件: {len(wav_files)} 个.wav, {len(mp3_files)} 个.mp3 (总计 {len(audio_files)} 个)")
    
    logger.info("=== 开始批量处理音频文件 ===")
    
    # 开始处理
    process_conversations_with_golden_text(
        audio_dir=audio_directory,
        golden_text_path=golden_text_file,
        output_dir=output_directory,
        conversation_mapping=None  # 使用自动解析
    )
    
    logger.info("=== 处理完成 ===")
    print("🎉 处理完成！结果保存在 /workspace/output/")


if __name__ == "__main__":
    main()
