"""
Configuration settings for the SLM project
Adjust these parameters based on your hardware and requirements
"""

# Model Configuration
MODEL_CONFIG = {
    # Base model to fine-tune
    # Options: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft/phi-2", "Qwen/Qwen2-1.5B"
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # Quantization (reduces memory usage)
    "load_in_4bit": True,  # Use 4-bit quantization (saves memory)
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
}

# LoRA Configuration (Parameter-Efficient Fine-Tuning)
LORA_CONFIG = {
    "r": 16,  # LoRA rank (higher = more parameters, better quality, more memory)
    "lora_alpha": 32,  # LoRA scaling factor
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers to apply LoRA
}

# Training Configuration
TRAINING_CONFIG = {
    "output_dir": "./models/finetuned-tinyllama",
    "num_train_epochs": 3,  # Number of training passes (increase for better quality)
    "per_device_train_batch_size": 1,  # Batch size (1 for 8GB RAM)
    "gradient_accumulation_steps": 4,  # Simulates larger batch size
    "learning_rate": 2e-4,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    # Memory optimizations
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",  # Memory-efficient optimizer
    "fp16": False,  # Use bf16 if your CPU supports it
    "bf16": False,
    
    # Logging and saving
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 2,
}

# Data Configuration
DATA_CONFIG = {
    "train_data_path": "./data/training_data.json",
    "max_length": 512,  # Maximum sequence length (reduce if OOM)
    "train_test_split": 0.95,  # 95% train, 5% validation
}

# Prompt Template
PROMPT_TEMPLATE = """Below is an instruction that describes a task related to Python or Machine Learning. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

# Inference Configuration (for Streamlit app)
INFERENCE_CONFIG = {
    "max_new_tokens": 256,  # Maximum length of generated response
    "temperature": 0.7,  # Creativity (0.0 = deterministic, 1.0 = creative)
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.15,
    "do_sample": True,
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Python/ML Assistant",
    "page_icon": "🐍",
    "initial_message": "Hello! I'm your Python and Machine Learning assistant. Ask me anything about Python programming, ML algorithms, MLOps, or data science!",
}
