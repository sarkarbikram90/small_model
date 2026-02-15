"""
Fine-tuning Script using QLoRA
Optimized for 8GB RAM systems without GPU
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import config


def load_and_prepare_data():
    """
    Load training data and prepare it for fine-tuning.
    """
    print("📚 Loading training data...")
    
    # Load data
    with open(config.DATA_CONFIG["train_data_path"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(data)
    
    # Split into train and validation
    split_ratio = config.DATA_CONFIG["train_test_split"]
    dataset = dataset.train_test_split(test_size=1-split_ratio, seed=42)
    
    print(f"📊 Train: {len(dataset['train'])} | Validation: {len(dataset['test'])}")
    
    return dataset


def format_prompt(example):
    """
    Format examples using the prompt template.
    """
    prompt = config.PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        response=example["response"]
    )
    return {"text": prompt}


def prepare_model_and_tokenizer():
    """
    Load model and tokenizer with 4-bit quantization for memory efficiency.
    """
    print("🤖 Loading model and tokenizer...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.MODEL_CONFIG["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=config.MODEL_CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_CONFIG["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_CONFIG["base_model"],
        trust_remote_code=True,
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("✅ Model and tokenizer loaded")
    
    return model, tokenizer


def apply_lora(model):
    """
    Apply LoRA adapters to the model for parameter-efficient fine-tuning.
    """
    print("🔧 Applying LoRA configuration...")
    
    lora_config = LoraConfig(
        r=config.LORA_CONFIG["r"],
        lora_alpha=config.LORA_CONFIG["lora_alpha"],
        lora_dropout=config.LORA_CONFIG["lora_dropout"],
        bias=config.LORA_CONFIG["bias"],
        task_type=config.LORA_CONFIG["task_type"],
        target_modules=config.LORA_CONFIG["target_modules"],
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ LoRA applied")
    print(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset.
    """
    print("✏️  Tokenizing dataset...")
    
    def tokenize_function(examples):
        # Format prompts
        formatted = format_prompt(examples)
        
        # Tokenize
        tokenized = tokenizer(
            formatted["text"],
            truncation=True,
            max_length=config.DATA_CONFIG["max_length"],
            padding="max_length",
        )
        
        # Set labels (for language modeling, labels = input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
    )
    
    print("✅ Tokenization complete")
    
    return tokenized_dataset


def train_model(model, tokenizer, tokenized_dataset):
    """
    Fine-tune the model using the prepared dataset.
    """
    print("🏋️  Starting training...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.TRAINING_CONFIG["output_dir"],
        num_train_epochs=config.TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=config.TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=config.TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=config.TRAINING_CONFIG["learning_rate"],
        max_grad_norm=config.TRAINING_CONFIG["max_grad_norm"],
        warmup_ratio=config.TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=config.TRAINING_CONFIG["lr_scheduler_type"],
        logging_steps=config.TRAINING_CONFIG["logging_steps"],
        save_strategy=config.TRAINING_CONFIG["save_strategy"],
        save_total_limit=config.TRAINING_CONFIG["save_total_limit"],
        optim=config.TRAINING_CONFIG["optim"],
        fp16=config.TRAINING_CONFIG["fp16"],
        bf16=config.TRAINING_CONFIG["bf16"],
        gradient_checkpointing=config.TRAINING_CONFIG["gradient_checkpointing"],
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # Train!
    print("\n⏳ Training in progress... This will take several hours on CPU.")
    print("💡 You can monitor progress in the logs below.\n")
    
    trainer.train()
    
    print("\n✅ Training complete!")
    
    return trainer


def save_model(model, tokenizer, output_dir):
    """
    Save the fine-tuned model and tokenizer.
    """
    print("💾 Saving model...")
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model saved to {output_dir}")
    print("\n🎉 Fine-tuning complete! You can now test your model with 3_test_model.py")


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("🚀 Starting SLM Fine-tuning Pipeline")
    print("=" * 60)
    print(f"\n📋 Configuration:")
    print(f"  - Base model: {config.MODEL_CONFIG['base_model']}")
    print(f"  - Training epochs: {config.TRAINING_CONFIG['num_train_epochs']}")
    print(f"  - Batch size: {config.TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  - Max length: {config.DATA_CONFIG['max_length']}")
    print(f"  - Output dir: {config.TRAINING_CONFIG['output_dir']}")
    print()
    
    # Check if data exists
    if not os.path.exists(config.DATA_CONFIG["train_data_path"]):
        print("❌ Error: Training data not found!")
        print(f"   Please run 1_data_preparation.py first to create {config.DATA_CONFIG['train_data_path']}")
        return
    
    # Load data
    dataset = load_and_prepare_data()
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Apply LoRA
    model = apply_lora(model)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Train model
    trainer = train_model(model, tokenizer, tokenized_dataset)
    
    # Save model
    save_model(model, tokenizer, config.TRAINING_CONFIG["output_dir"])
    
    print("\n" + "=" * 60)
    print("✨ All done! Next steps:")
    print("  1. Test your model: python 3_test_model.py")
    print("  2. Run chat interface: streamlit run streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
