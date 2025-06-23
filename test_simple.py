#!/usr/bin/env python3
"""
简单测试脚本 - 快速验证环境
不会导入复杂的库，避免卡死
"""

import os
import sys
from pathlib import Path

def main():
    print("🧪 简单环境测试")
    print("=" * 30)
    
    # 基本信息
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    
    # 检查目录
    input_dir = "/workspace/input"
    output_dir = "/workspace/output"
    
    print(f"\n📁 检查目录:")
    print(f"输入目录存在: {os.path.exists(input_dir)}")
    print(f"输出目录存在: {os.path.exists(output_dir)}")
    
    if os.path.exists(input_dir):
        files = list(Path(input_dir).glob("*"))
        print(f"输入目录文件数: {len(files)}")
        for f in files[:5]:  # 只显示前5个
            print(f"  - {f.name}")
    
    # 测试导入
    print(f"\n🔍 测试导入:")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy: {e}")
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except Exception as e:
        print(f"❌ Pandas: {e}")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
    
    print("\n如果以上都正常，可以运行: python process_audio_large.py")

if __name__ == "__main__":
    main()