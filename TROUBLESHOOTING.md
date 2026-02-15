# Troubleshooting Guide

Common issues and solutions when building your SLM.

## Installation Issues

### Issue: `pip install` fails with bitsandbytes
**Error**: `Could not find a version that satisfies the requirement bitsandbytes`

**Solution**:
```bash
# For Windows (bitsandbytes doesn't officially support Windows)
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl

# For macOS/Linux
pip install bitsandbytes --break-system-packages
```

### Issue: PyTorch installation fails
**Solution**:
```bash
# Install PyTorch CPU version (for systems without GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Training Issues

### Issue: Out of Memory (OOM) during training
**Error**: `RuntimeError: [enforce fail at alloc_cpu.cpp:...] . DefaultCPUAllocator: can't allocate memory`

**Solutions**:
1. Reduce batch size in `config.py`:
```python
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,  # Already at minimum
    "gradient_accumulation_steps": 8,  # Increase this instead
}
```

2. Reduce sequence length:
```python
DATA_CONFIG = {
    "max_length": 256,  # Reduce from 512
}
```

3. Close other applications to free RAM

4. Use swap space (not recommended, will be very slow):
```bash
# Linux: Check swap
free -h

# Windows: Check virtual memory in Task Manager
```

### Issue: Training is extremely slow
**Cause**: CPU training is inherently slow (4-8 hours is normal)

**Solutions**:
1. Reduce dataset size for testing:
```python
# In 1_data_preparation.py, use fewer examples
sample_data = create_sample_dataset()[:50]  # Only 50 examples for quick test
```

2. Use Google Colab (free GPU):
   - Upload your code to Google Drive
   - Open Colab notebook
   - Run training there (10-20x faster)

3. Reduce epochs:
```python
TRAINING_CONFIG = {
    "num_train_epochs": 1,  # Instead of 3
}
```

### Issue: Loss not decreasing
**Possible causes**:
1. Learning rate too high/low
2. Dataset too small
3. Data quality issues

**Solutions**:
```python
# Try different learning rates
TRAINING_CONFIG = {
    "learning_rate": 1e-4,  # Try 1e-4, 2e-4, 5e-5
}

# Ensure dataset has at least 100+ examples
# Check data quality - responses should be helpful and accurate
```

## Model Loading Issues

### Issue: Model not found error
**Error**: `OSError: ./models/finetuned-tinyllama does not appear to have a file named config.json`

**Solution**:
- Make sure you completed training successfully
- Check that files exist in `./models/finetuned-tinyllama/`
- Re-run training if files are missing

### Issue: Slow model loading in Streamlit
**Cause**: Normal - first load takes 30-60 seconds on CPU

**Solution**:
- This is expected behavior
- Model is cached after first load
- Subsequent loads are instant

## Streamlit App Issues

### Issue: Import errors when running Streamlit
**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Streamlit shows "Model loading failed"
**Solution**:
1. Verify model exists: `ls ./models/finetuned-tinyllama/`
2. Check training completed successfully
3. Ensure all model files are present:
   - adapter_config.json
   - adapter_model.bin
   - tokenizer files

### Issue: Responses are gibberish or repetitive
**Possible causes**:
1. Insufficient training
2. Poor quality training data
3. Wrong inference parameters

**Solutions**:
1. Train for more epochs
2. Improve training data quality
3. Adjust inference settings in Streamlit sidebar:
   - Increase `repetition_penalty` (1.2-1.5)
   - Adjust `temperature` (0.5-0.9)
   - Try different `top_p` values

## Data Preparation Issues

### Issue: No training data found
**Error**: `❌ Error: Training data not found!`

**Solution**:
```bash
# Run data preparation first
python 1_data_preparation.py

# Verify file exists
ls data/training_data.json
```

### Issue: Dataset is too small
**Problem**: Only 10-20 examples won't train a good model

**Solution**:
1. Add more examples manually in `1_data_preparation.py`
2. Use data augmentation
3. Collect from Stack Overflow API
4. Generate synthetic examples with GPT-4 (then review them)

**Minimum recommendations**:
- For testing: 50-100 examples
- For decent results: 1,000+ examples
- For good results: 5,000+ examples

## Hardware-Specific Issues

### Issue: Computer freezes during training
**Cause**: Using all available RAM + swap

**Solutions**:
1. Close all other applications
2. Reduce batch size and sequence length
3. Use cloud computing instead (Colab)
4. Consider training in stages with breaks

### Issue: No GPU detected (and you have one)
**Solution**:
```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU present

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Deployment Issues

### Issue: Streamlit Cloud deployment fails
**Error**: `Killed` or `Out of memory`

**Solution**:
- See `DEPLOYMENT.md` for detailed solutions
- Consider Hugging Face Spaces instead
- Use smaller model or quantization

### Issue: Model too large for deployment
**Solutions**:
1. Use more aggressive quantization (4-bit)
2. Deploy to platform with more resources
3. Use model pruning techniques
4. Consider distillation to smaller model

## Quality Issues

### Issue: Model gives wrong or unhelpful answers
**Solutions**:

1. **Improve training data**:
   - Add more diverse examples
   - Ensure answers are accurate and complete
   - Remove low-quality Q&A pairs

2. **Train longer**:
   - Increase epochs from 3 to 5-10
   - Monitor validation loss

3. **Try different base models**:
   - Switch from TinyLlama to Phi-2
   - Try Qwen2-1.5B

4. **Use retrieval-augmented generation (RAG)**:
   - Add a vector database
   - Retrieve relevant docs before generating
   - This is an advanced technique

### Issue: Model refuses to answer or says "I don't know"
**Cause**: Base model's safety training carries over

**Solution**:
- Add more confident, direct examples to training data
- Adjust system prompt
- This is partly intentional (better to admit uncertainty than hallucinate)

## Getting Help

If you encounter issues not listed here:

1. **Check error messages carefully** - they often indicate the exact problem
2. **Google the error** - someone likely had the same issue
3. **Check GitHub issues** for transformers, peft, streamlit
4. **Ask in communities**:
   - Hugging Face forums
   - r/MachineLearning subreddit
   - Stack Overflow

## Prevention Best Practices

1. **Start small**: Test with 50 examples and 1 epoch first
2. **Monitor resources**: Keep Task Manager/Activity Monitor open
3. **Save frequently**: Training script auto-saves, but keep backups
4. **Version control**: Use git to track changes
5. **Document changes**: Note what works and what doesn't

## Quick Diagnostic Commands

```bash
# Check Python version
python --version  # Should be 3.8+

# Check installed packages
pip list | grep torch
pip list | grep transformers

# Check available RAM
free -h  # Linux
vm_stat  # macOS

# Check GPU
nvidia-smi  # If you have NVIDIA GPU

# Test imports
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

## Still Having Issues?

Create an issue with:
1. Full error message
2. Your system specs (RAM, CPU, OS)
3. Steps you've tried
4. Relevant config settings

Good luck! 🚀
