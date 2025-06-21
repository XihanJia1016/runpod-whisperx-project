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
    
    # 音频文件配置 - 自动检测35.x.mp3格式文件
    input_dir = Path("/workspace/input")
    audio_files = list(input_dir.glob("35.*.mp3"))
    
    if not audio_files:
        print("❌ 未找到35.*.mp3格式的文件")
        print("💡 请确保音频文件命名为: 35.1.mp3, 35.2.mp3, 35.3.mp3")
        sys.exit(1)
    
    # 解析文件配置
    audio_configs = []
    for file_path in sorted(audio_files):
        filename = file_path.name
        # 从35.X.mp3提取conversation ID
        try:
            conversation_id = int(filename.split('.')[1])
            audio_configs.append({
                "file": filename,
                "path": file_path,
                "dyad": 35,
                "conversation": conversation_id
            })
        except (IndexError, ValueError):
            print(f"⚠️ 跳过文件: {filename} (格式不正确)")
    
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
            
            # 保存合并的CSV
            combined_file = "/workspace/output/dyad_35_combined_transcription.csv"
            df_combined.to_csv(combined_file, index=False, encoding='utf-8')
            
            print(f"✅ 合并文件已保存: {combined_file}")
            print(f"📈 总片段数: {len(df_combined)}")
            
            # 保存分组统计
            stats_file = "/workspace/output/dyad_35_summary_stats.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"Dyad 35 转录统计汇总\n")
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
