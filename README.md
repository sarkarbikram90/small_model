# Building a Python/ML Question-Answering SLM

Complete guide to building, training, and deploying a Small Language Model for Python and Machine Learning questions on your laptop (8GB RAM, i5 processor).

## Project Overview

This project fine-tunes a small language model (TinyLlama-1.1B) on Python and ML Q&A data, then deploys it as a Streamlit chat interface.

## Hardware Requirements

- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, dedicated GPU
- **Your Setup**: Works with 8GB RAM, i5 processor (we'll use optimizations)

## Project Structure

```
slm-python-ml/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── 1_data_preparation.py        # Collect and prepare training data
├── 2_finetune_model.py         # Fine-tune the model
├── 3_test_model.py             # Test your trained model
├── streamlit_app.py            # Chat interface
├── config.py                   # Configuration settings
├── data/
│   └── training_data.json      # Your Q&A dataset
└── models/
    └── finetuned-tinyllama/    # Your trained model
```

## Step-by-Step Guide

### Step 1: Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Training Data

```bash
python 1_data_preparation.py
```

This script will:
- Download Python/ML Q&A pairs from Stack Overflow (via API)
- Format data for training
- Save to `data/training_data.json`

**Alternative**: Manually create your dataset using the format shown in the script.

### Step 3: Fine-tune the Model

```bash
python 2_finetune_model.py
```

**Training time**: 4-8 hours on your hardware (depending on dataset size)
**Memory usage**: ~6-7GB RAM with our optimizations

The script will:
- Load TinyLlama-1.1B model in 4-bit quantization
- Apply QLoRA fine-tuning
- Save the trained adapter weights

**Monitor progress**: The script shows loss and progress bars

### Step 4: Test Your Model

```bash
python 3_test_model.py
```

Test with example questions to verify your model works.

### Step 5: Run Streamlit App Locally

```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501` to chat with your model.

### Step 6: Deploy to Streamlit Cloud

1. Push your code to GitHub (exclude large model files)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

**Important**: Streamlit Cloud has 1GB memory limit. You'll need to:
- Use the smallest model possible
- Load model in 4-bit quantization
- Consider using Hugging Face Spaces instead (more resources)

## Configuration Options

Edit `config.py` to customize:
- Model selection (TinyLlama, Phi-2, Qwen2)
- Training parameters (epochs, batch size, learning rate)
- Maximum sequence length
- Quantization settings

## Tips for Better Results

1. **Data Quality > Quantity**: 5,000 high-quality Q&A pairs beat 50,000 poor ones
2. **Domain Focus**: Stick to Python/ML topics for better specialization
3. **Prompt Format**: Use consistent instruction format in training data
4. **Evaluation**: Test on held-out questions regularly
5. **Iterate**: Start small, evaluate, improve data, retrain

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size` in config
- Reduce `max_length` 
- Use gradient checkpointing (already enabled)
- Close other applications

### Slow Training
- Reduce dataset size for testing
- Use fewer epochs initially
- Consider cloud GPU (Google Colab free tier)

### Poor Model Responses
- Check training data quality
- Increase training epochs
- Add more diverse examples
- Try different base models

## Alternative Approaches

If laptop training is too slow:

1. **Google Colab** (Free GPU): Run fine-tuning there, download model
2. **Hugging Face Spaces**: Better deployment option than Streamlit Cloud
3. **Modal/Replicate**: Pay-per-use GPU training
4. **Pre-trained models**: Use existing Python-focused models (CodeLlama, StarCoder)

## Resources

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [TinyLlama GitHub](https://github.com/jzhang38/TinyLlama)
- [Streamlit Documentation](https://docs.streamlit.io)

## Next Steps

After getting the basic version working:
1. Expand your dataset with more examples
2. Experiment with different models (Phi-2, Qwen2)
3. Add retrieval-augmented generation (RAG) for better accuracy
4. Fine-tune on specific domains (MLOps, deep learning, etc.)
5. Implement evaluation metrics

## License

This project template is MIT licensed. Note that base models have their own licenses.
