#!/usr/bin/env python3
"""
快速修复NumPy版本冲突问题
解决WhisperX 3.3.4与NumPy版本要求的冲突
"""

import subprocess
import sys

def run_command(cmd, description, ignore_errors=False):
    """运行命令"""
    print(f"🔧 {description}")
    print(f"执行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"⚠️ {description} - 忽略错误")
            return True
        print(f"❌ {description} - 失败")
        print(f"错误: {e.stderr[:200]}...")
        return False

def fix_numpy_conflict():
    """修复NumPy版本冲突"""
    print("🚀 修复WhisperX NumPy版本冲突")
    print("=" * 50)
    
    # 1. 卸载冲突的numpy版本
    print("\n1️⃣ 卸载现有NumPy...")
    run_command("pip uninstall -y numpy", "卸载NumPy", ignore_errors=True)
    
    # 2. 安装兼容的numpy版本
    print("\n2️⃣ 安装兼容的NumPy版本...")
    if not run_command("pip install 'numpy>=2.0.2'", "安装NumPy 2.0.2+"):
        print("❌ NumPy安装失败")
        return False
    
    # 3. 安装其他核心依赖
    print("\n3️⃣ 安装核心依赖...")
    core_deps = [
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0", 
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "tqdm>=4.65.0"
    ]
    
    for dep in core_deps:
        run_command(f"pip install '{dep}'", f"安装 {dep.split('>=')[0]}")
    
    # 4. 安装PyTorch (如果需要)
    print("\n4️⃣ 检查PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch已安装: {torch.__version__}")
    except ImportError:
        print("安装PyTorch...")
        run_command(
            "pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118",
            "安装PyTorch CUDA版本"
        )
    
    # 5. 安装transformers
    print("\n5️⃣ 安装Transformers...")
    run_command("pip install 'transformers>=4.35.0,<5.0.0' 'accelerate>=0.20.0'", "安装Transformers")
    
    # 6. 安装pyannote
    print("\n6️⃣ 安装pyannote.audio...")
    run_command("pip install pyannote.audio==3.1.1", "安装pyannote.audio")
    
    # 7. 最后安装WhisperX
    print("\n7️⃣ 安装WhisperX...")
    if not run_command("pip install whisperx==3.3.4", "安装WhisperX 3.3.4"):
        print("❌ WhisperX安装失败")
        return False
    
    # 8. 验证安装
    print("\n8️⃣ 验证安装...")
    try:
        import numpy as np
        import pandas as pd
        import torch
        import whisperx
        import pyannote.audio
        from sklearn.metrics.pairwise import cosine_similarity
        
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Pandas: {pd.__version__}")
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ WhisperX: 可用")
        print(f"✅ pyannote.audio: {pyannote.audio.__version__}")
        print(f"✅ scikit-learn: 可用")
        
        print("\n🎉 所有依赖安装成功！")
        return True
        
    except ImportError as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    print("WhisperX NumPy冲突修复工具")
    print("==========================")
    
    if fix_numpy_conflict():
        print("\n✅ 修复完成！现在可以运行:")
        print("export HF_TOKEN='your_token'")
        print("python process_audio_large.py")
    else:
        print("\n❌ 修复失败，请查看错误信息")
        sys.exit(1)