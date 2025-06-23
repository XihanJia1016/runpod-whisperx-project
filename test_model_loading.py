#!/usr/bin/env python3
"""
测试 pyannote/embedding 模型加载
专门用于诊断模型加载问题
"""

import os
import sys
import traceback
from pyannote.audio import Pipeline

def test_model_loading():
    print("🧪 测试 pyannote/embedding 模型加载")
    print("=" * 50)
    
    # 检查Token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ 未设置 HF_TOKEN 环境变量")
        return False
        
    print(f"🔑 Token: {hf_token[:20]}...")
    
    try:
        print("⏳ 开始加载模型...")
        
        # 清除缓存，重新下载
        import shutil
        cache_dirs = [
            os.path.expanduser("~/.cache/huggingface"),
            "/tmp/huggingface_cache"
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                print(f"🗑️ 清除缓存: {cache_dir}")
                shutil.rmtree(cache_dir)
        
        # 手动下载模型
        print("📥 下载模型文件...")
        from huggingface_hub import snapshot_download
        
        model_path = snapshot_download(
            "pyannote/embedding",
            token=hf_token,
            cache_dir="/tmp/huggingface_cache"
        )
        print(f"✅ 模型下载完成: {model_path}")
        
        # 加载Pipeline
        print("🔄 初始化Pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/embedding",
            use_auth_token=hf_token,
            cache_dir="/tmp/huggingface_cache"
        )
        
        print(f"✅ Pipeline加载成功!")
        print(f"📊 模型类型: {type(pipeline)}")
        print(f"📊 Pipeline属性: {dir(pipeline)}")
        
        # 测试移动到CUDA（如果可用）
        import torch
        if torch.cuda.is_available():
            print("🚀 测试CUDA...")
            pipeline = pipeline.to("cuda")
            print("✅ CUDA移动成功")
        else:
            print("💻 使用CPU模式")
            
        print("🎉 模型加载测试完全成功!")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("详细错误:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)