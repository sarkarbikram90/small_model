# Deployment Guide for Streamlit Cloud

## Important Notes About Streamlit Cloud Deployment

### Resource Limitations

Streamlit Cloud (free tier) has significant limitations:
- **1GB RAM limit** (your model needs 4-6GB)
- **1 CPU core**
- Limited storage

**Reality check**: Your fine-tuned model is too large for Streamlit Cloud's free tier. You have better options:

## Alternative Deployment Options

### Option 1: Hugging Face Spaces (Recommended)
✅ **2GB RAM** (more than Streamlit Cloud)
✅ Free GPU option available
✅ Better for ML models
✅ Easy deployment

**Steps**:
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (choose Streamlit)
3. Push your code to the Space
4. Upload model to Hugging Face Model Hub
5. Load model from Hub in your app

**Example code to load from Hub**:
```python
from huggingface_hub import hf_hub_download

# Upload your model first:
# huggingface-cli upload your-username/your-model ./models/finetuned-tinyllama

# In streamlit_app.py:
model_path = "your-username/your-model"
```

### Option 2: Render.com
✅ 512MB-1GB RAM (free tier)
✅ Can upgrade for more resources
✅ Supports custom Docker images

### Option 3: Railway.app
✅ $5/month credit (free)
✅ Better resources than Streamlit Cloud
✅ Easy deployment

### Option 4: Local Network Only
If deployment is too complex, run locally and share:
1. Run `streamlit run streamlit_app.py`
2. Use ngrok to expose: `ngrok http 8501`
3. Share the temporary URL

## If You Still Want to Try Streamlit Cloud

You'll need extreme optimizations:

### 1. Use the Smallest Possible Model
```python
# In config.py, change to:
MODEL_CONFIG = {
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Smallest option
    "load_in_4bit": True,
}
```

### 2. Create .streamlit/config.toml
```toml
[server]
maxUploadSize = 500
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### 3. Optimize Model Loading
```python
# In streamlit_app.py, add memory optimization:
import gc
import torch

@st.cache_resource
def load_model():
    # Clear any existing models
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load with aggressive quantization
    # ... rest of loading code
```

### 4. Create .slugignore
```
data/
*.pyc
__pycache__/
.git/
tests/
docs/
```

## Deployment Steps for Streamlit Cloud (If Attempting)

1. **Prepare Your Repository**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

2. **Important Files**
- Ensure `requirements.txt` is present
- Create `packages.txt` for system dependencies (if needed):
```
build-essential
```

3. **Deploy**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click "New app"
- Select your repository
- Set main file: `streamlit_app.py`
- Deploy!

4. **Monitor Resources**
- Watch the deployment logs
- If you see "Killed" or "Out of memory", the app is too large

## Best Practice Recommendation

**For a real deployment**, I strongly recommend:

1. **Hugging Face Spaces** for hosting your Streamlit app
2. Upload your model to **Hugging Face Model Hub**
3. Use their free GPU inference for better performance

Would you like a detailed guide for Hugging Face Spaces deployment instead?

## Cost Comparison

| Platform | RAM | GPU | Cost | Best For |
|----------|-----|-----|------|----------|
| Streamlit Cloud | 1GB | No | Free | Simple apps |
| HF Spaces | 2GB+ | Yes (paid) | Free/Paid | ML models ✅ |
| Render | 512MB-1GB | No | Free/Paid | Web apps |
| Railway | Better | No | $5 credit | Full apps |
| AWS/GCP | Unlimited | Yes | Pay-as-go | Production |

## Troubleshooting Deployment Issues

### Error: "Killed" or "Out of Memory"
- Your model is too large for the platform
- Solution: Use smaller model or upgrade tier

### Error: "Module not found"
- Missing dependency in requirements.txt
- Solution: Add the missing package

### Slow Loading
- Model loading takes time on CPU
- Solution: Use GPU-enabled platform (HF Spaces)

### Model Not Found
- Model path incorrect
- Solution: Check that model files are included or use Model Hub

## Final Recommendation

For your 8GB laptop setup with a fine-tuned model:
1. ✅ **Develop and test locally** (works great!)
2. ✅ **Deploy to Hugging Face Spaces** (best free option)
3. ❌ **Avoid Streamlit Cloud** (insufficient resources)

Good luck with your deployment! 🚀
