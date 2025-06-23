#!/usr/bin/env python3
"""
半自动种子识别说话人日志系统 - 环境设置脚本
解决依赖冲突，确保环境可重复使用
"""

import subprocess
import sys
import os
import importlib.util

def check_package_installed(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def run_pip_command(command, description, ignore_errors=False):
    """运行pip命令"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"⚠️ {description} - 忽略错误: {e.stderr[:100]}...")
            return True
        else:
            print(f"❌ {description} - 失败: {e.stderr[:100]}...")
            return False

def check_cuda_available():
    """检查CUDA是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_from_requirements():
    """使用requirements.txt安装所有依赖"""
    print("\n=== 使用requirements.txt安装依赖 ===")
    
    # 检查requirements.txt是否存在
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt 文件不存在")
        return False
    
    # 首先尝试直接安装
    print("🔄 尝试直接安装所有依赖...")
    if run_pip_command("pip install -r requirements.txt", "安装requirements.txt", ignore_errors=True):
        print("✅ 依赖安装成功")
        return True
    
    print("⚠️ 直接安装失败，尝试分步安装...")
    
    # 如果失败，分步安装关键组件
    return install_pytorch_compatible()

def install_pytorch_compatible():
    """安装兼容的PyTorch版本"""
    print("\n=== 安装PyTorch生态系统 ===")
    
    # 检查是否已有兼容版本
    try:
        import torch
        import torchvision
        import torchaudio
        
        torch_version = torch.__version__
        torchvision_version = torchvision.__version__
        
        print(f"检测到现有版本: torch={torch_version}, torchvision={torchvision_version}")
        
        # 检查版本兼容性
        if torch_version.startswith('2.1.') and torchvision_version.startswith('0.16.'):
            print("✅ 现有PyTorch版本兼容，跳过重装")
            return True
        else:
            print("⚠️ 版本不兼容，需要重新安装")
    except ImportError:
        print("未检测到PyTorch，开始安装...")
    
    # 卸载冲突版本
    packages_to_uninstall = [
        "torch", "torchaudio", "torchvision", 
        "torchmetrics", "pytorch-lightning", "pytorch-metric-learning"
    ]
    
    for package in packages_to_uninstall:
        run_pip_command(f"pip uninstall -y {package}", f"卸载 {package}", ignore_errors=True)
    
    # 安装兼容版本
    # 优先尝试CUDA版本，失败则使用CPU版本
    cuda_install_cmd = (
        "pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    
    cpu_install_cmd = (
        "pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 "
        "--index-url https://download.pytorch.org/whl/cpu"
    )
    
    # 尝试CUDA版本
    if run_pip_command(cuda_install_cmd, "安装PyTorch (CUDA版本)", ignore_errors=True):
        print("✅ CUDA版本安装成功")
        return True
    else:
        print("⚠️ CUDA版本安装失败，尝试CPU版本...")
        return run_pip_command(cpu_install_cmd, "安装PyTorch (CPU版本)")

def install_core_dependencies():
    """安装核心依赖"""
    print("\n=== 安装核心依赖 ===")
    
    core_packages = [
        "numpy>=2.0.2",  # 更新以兼容WhisperX 3.3.4
        "pandas>=2.0.0,<3.0.0", 
        "scikit-learn>=1.3.0,<2.0.0",
        "librosa>=0.10.0,<1.0.0",
        "soundfile>=0.12.0,<1.0.0",
        "tqdm>=4.65.0"
    ]
    
    for package in core_packages:
        if not run_pip_command(f"pip install '{package}'", f"安装 {package.split('>=')[0]}"):
            return False
    
    return True

def install_transformers():
    """安装Transformers生态"""
    print("\n=== 安装Transformers ===")
    
    transformers_packages = [
        "transformers>=4.35.0,<5.0.0",
        "accelerate>=0.20.0,<1.0.0"
    ]
    
    for package in transformers_packages:
        if not run_pip_command(f"pip install '{package}'", f"安装 {package.split('>=')[0]}"):
            return False
    
    return True

def install_pyannote():
    """安装pyannote.audio"""
    print("\n=== 安装pyannote.audio ===")
    
    # 检查是否已安装
    if check_package_installed("pyannote.audio", "pyannote.audio"):
        try:
            import pyannote.audio
            print(f"✅ pyannote.audio 已安装: {pyannote.audio.__version__}")
            return True
        except Exception as e:
            print(f"⚠️ pyannote.audio 导入失败: {e}")
    
    return run_pip_command("pip install pyannote.audio==3.1.1", "安装 pyannote.audio")

def install_whisperx():
    """安装WhisperX"""
    print("\n=== 安装WhisperX ===")
    
    # 检查是否已安装
    if check_package_installed("whisperx"):
        try:
            import whisperx
            print("✅ WhisperX 已安装")
            return True
        except Exception as e:
            print(f"⚠️ WhisperX 导入失败: {e}")
    
    # 使用固定版本避免依赖冲突
    return run_pip_command(
        "pip install whisperx==3.3.4", 
        "安装 WhisperX 3.3.4"
    )

def verify_installation():
    """验证安装"""
    print("\n=== 验证安装 ===")
    
    try:
        # 测试PyTorch
        import torch
        import torchaudio
        import torchvision
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ TorchAudio: {torchaudio.__version__}")
        print(f"✅ TorchVision: {torchvision.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        
        # 测试科学计算库
        import numpy as np
        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Pandas: {pd.__version__}")
        print("✅ Scikit-learn: 可用")
        
        # 测试音频库
        import librosa
        import soundfile
        print(f"✅ Librosa: {librosa.__version__}")
        print("✅ SoundFile: 可用")
        
        # 测试pyannote
        import pyannote.audio
        print(f"✅ pyannote.audio: {pyannote.audio.__version__}")
        
        # 测试WhisperX
        import whisperx
        print("✅ WhisperX: 可用")
        
        # 测试Transformers
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        print("\n🎉 所有依赖验证成功！")
        return True
        
    except ImportError as e:
        print(f"❌ 验证失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 半自动种子识别说话人日志系统 - 环境设置")
    print("=" * 50)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        sys.exit(1)
    
    print(f"✅ Python版本: {sys.version}")
    
    # 逐步安装
    steps = [
        ("从requirements.txt安装", install_from_requirements),
        ("验证安装", verify_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"❌ {step_name} 失败，终止安装")
            sys.exit(1)
    
    print("\n🎉 环境设置完成！")
    print("\n📝 下一步:")
    print("1. 设置HuggingFace Token: export HF_TOKEN='your_token_here'")
    print("2. 运行脚本: python process_audio_large.py")

if __name__ == "__main__":
    main()