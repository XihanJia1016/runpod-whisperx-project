# HuggingFace Token 设置说明

## RunPod快速设置

1. 手动设置环境变量：
```bash
export HF_TOKEN='your_huggingface_token_here'
bash runpod_setup_guide.sh
```

## 其他环境设置

如需在其他环境使用，请设置环境变量：

```bash
export HF_TOKEN='your_huggingface_token_here'
python process_audio_large.py
```

## 获取Token

1. 访问 https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择 "Read" 权限
4. 复制生成的Token

## 安全提醒

- 不要将Token提交到GitHub
- 使用环境变量设置Token
- 在RunPod中手动设置环境变量