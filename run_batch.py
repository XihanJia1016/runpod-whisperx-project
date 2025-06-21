"""
批量处理3个音频文件并合并为单个CSV的主脚本
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from process_audio_large import HighPrecisionAudioProcessor

def main():
    print("🚀 RunPod WhisperX Large-v3 批量处理开始")
    print("="*60)
    
    # 音频文件配置 - 自动检测所有音频文件
    input_dir = Path("/workspace/input")
    # 支持常见音频格式
    audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(ext))
    
    if not audio_files:
        print("❌ 未找到音频文件")
        print("💡 支持的格式: .mp3, .wav, .m4a, .flac, .ogg")
        print(f"💡 请将音频文件放在: {input_dir}")
        sys.exit(1)
    
    # 解析文件配置
    audio_configs = []
    for i, file_path in enumerate(sorted(audio_files), 1):
        filename = file_path.name
        
        # 从文件名提取dyad和conversation信息
        # 支持格式: 19.4.mp3 -> dyad=19, conversation=4
        # 支持格式: 33.4.mp3 -> dyad=33, conversation=4  
        # 支持格式: 35.3.mp3 -> dyad=35, conversation=3
        try:
            # 移除文件扩展名
            name_without_ext = filename.rsplit('.', 1)[0]
            # 按点分割
            parts = name_without_ext.split('.')
            
            if len(parts) >= 2:
                dyad_id = int(parts[0])
                conversation_id = int(parts[1])
            else:
                # 如果无法解析，使用文件序号
                dyad_id = 35  # 默认dyad
                conversation_id = i
                print(f"⚠️ 无法从文件名 {filename} 解析dyad.conversation，使用默认值: dyad={dyad_id}, conversation={conversation_id}")
                
        except (ValueError, IndexError):
            # 解析失败，使用默认值
            dyad_id = 35
            conversation_id = i
            print(f"⚠️ 文件名 {filename} 格式不标准，使用默认值: dyad={dyad_id}, conversation={conversation_id}")
        
        audio_configs.append({
            "file": filename,
            "path": file_path,
            "dyad": dyad_id,
            "conversation": conversation_id
        })
    
    if not audio_configs:
        print("❌ 没有有效的音频文件")
        sys.exit(1)
    
    print(f"📁 发现 {len(audio_configs)} 个音频文件:")
    for config in audio_configs:
        print(f"   - {config['file']} (dyad={config['dyad']}, conversation={config['conversation']})")
    
    # 初始化处理器
    processor = HighPrecisionAudioProcessor()
    
    total_start = time.time()
    all_segments = []  # 存储所有片段数据
    
    try:
        # 加载模型（一次性加载）
        print("\n📦 加载模型...")
        if not processor.load_models():
            print("❌ 模型加载失败")
            sys.exit(1)
        
        print("✅ 所有模型加载完成")
        
        # 处理所有文件
        results = []
        total_segments_count = 0
        
        for i, config in enumerate(audio_configs, 1):
            print(f"\n🎵 处理文件 {i}/{len(audio_configs)}: {config['file']}")
            print("-" * 40)
            
            try:
                segments = processor.process_single_file(
                    str(config['path']),
                    config["dyad"],
                    config["conversation"]
                )
                
                # 添加到总列表
                all_segments.extend(segments)
                
                results.append({
                    "file": config["file"],
                    "segments": len(segments),
                    "status": "success"
                })
                
                total_segments_count += len(segments)
                print(f"✅ {config['file']} -> {len(segments)} 片段")
                
            except Exception as e:
                print(f"❌ {config['file']} 处理失败: {e}")
                results.append({
                    "file": config["file"],
                    "segments": 0,
                    "status": "failed",
                    "error": str(e)
                })
        
        # 合并所有结果到单个CSV
        if all_segments:
            print(f"\n📊 合并所有结果...")
            
            # 创建DataFrame
            df_combined = pd.DataFrame(all_segments)
            
            # 重新排序segment_id（全局连续）
            df_combined['segment_id'] = range(1, len(df_combined) + 1)
            
            # 根据实际dyad生成文件名
            dyad_ids = df_combined['dyad'].unique()
            if len(dyad_ids) == 1:
                # 单个dyad
                dyad_name = f"dyad_{dyad_ids[0]}"
            else:
                # 多个dyad
                dyad_name = f"dyads_{'_'.join(map(str, sorted(dyad_ids)))}"
            
            # 保存合并的CSV
            combined_file = f"/workspace/output/{dyad_name}_combined_transcription.csv"
            df_combined.to_csv(combined_file, index=False, encoding='utf-8')
            
            print(f"✅ 合并文件已保存: {combined_file}")
            print(f"📈 总片段数: {len(df_combined)}")
            
            # 保存分组统计
            stats_file = f"/workspace/output/{dyad_name}_summary_stats.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"{dyad_name.replace('_', ' ').title()} 转录统计汇总\n")
                f.write(f"{'='*50}\n\n")
                
                # 按conversation分组统计
                conv_stats = df_combined.groupby('conversation').agg({
                    'duration': 'sum',
                    'segment_id': 'count',
                    'word_count': 'sum',
                    'confidence': 'mean'
                }).round(3)
                
                f.write(f"按conversation分组:\n")
                for conv_id in sorted(df_combined['conversation'].unique()):
                    conv_data = df_combined[df_combined['conversation'] == conv_id]
                    f.write(f"  Conversation {conv_id}:\n")
                    f.write(f"    片段数: {len(conv_data)}\n")
                    f.write(f"    总时长: {conv_data['duration'].sum():.2f}秒\n")
                    f.write(f"    词数: {conv_data['word_count'].sum()}\n")
                    f.write(f"    平均置信度: {conv_data['confidence'].mean():.4f}\n\n")
            
            print(f"📋 统计报告已保存: {stats_file}")
        
        # 显示总结
        total_time = time.time() - total_start
        successful = sum(1 for r in results if r["status"] == "success")
        
        print(f"\n🎉 批量处理完成!")
        print("="*60)
        print(f"📊 处理统计:")
        print(f"   总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"   成功文件: {successful}/{len(audio_configs)}")
        print(f"   总片段数: {total_segments_count}")
        print(f"   平均速度: {total_segments_count/total_time:.1f} 片段/秒")
        
        print(f"\n📁 输出文件:")
        output_files = list(Path("/workspace/output").glob("*"))
        for file in sorted(output_files):
            print(f"   {file.name}")
        
    finally:
        # 清理资源
        processor.cleanup()
        print("\n🧹 资源清理完成")

if __name__ == "__main__":
    main()
