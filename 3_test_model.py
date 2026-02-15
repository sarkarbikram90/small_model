"""
Test Script for Fine-tuned Model
Quick tests to verify your model is working correctly
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import config


def load_model():
    """
    Load the fine-tuned model for inference.
    """
    print("🤖 Loading fine-tuned model...")
    
    model_path = config.TRAINING_CONFIG["output_dir"]
    base_model_name = config.MODEL_CONFIG["base_model"]
    
    # Configure 4-bit quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("✅ Model loaded successfully!\n")
    
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str) -> str:
    """
    Generate a response for a given instruction.
    """
    # Format prompt
    prompt = f"""Below is an instruction that describes a task related to Python or Machine Learning. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.INFERENCE_CONFIG["max_new_tokens"],
            temperature=config.INFERENCE_CONFIG["temperature"],
            top_p=config.INFERENCE_CONFIG["top_p"],
            top_k=config.INFERENCE_CONFIG["top_k"],
            repetition_penalty=config.INFERENCE_CONFIG["repetition_penalty"],
            do_sample=config.INFERENCE_CONFIG["do_sample"],
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (after "### Response:")
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    return response


def run_tests(model, tokenizer):
    """
    Run a series of test questions to evaluate model performance.
    """
    test_questions = [
        "What is a Python decorator?",
        "How do you handle exceptions in Python?",
        "Explain the bias-variance tradeoff in machine learning.",
        "What is the difference between a list and a numpy array?",
        "How do you prevent overfitting in neural networks?",
    ]
    
    print("=" * 70)
    print("🧪 Running Test Questions")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 70)
        
        response = generate_response(model, tokenizer, question)
        print(f"🤖 Response:\n{response}")
        print("=" * 70)


def interactive_mode(model, tokenizer):
    """
    Interactive mode for testing custom questions.
    """
    print("\n" + "=" * 70)
    print("💬 Interactive Mode")
    print("=" * 70)
    print("Ask questions about Python or Machine Learning!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        question = input("❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🤖 Generating response...\n")
        response = generate_response(model, tokenizer, question)
        print(f"💡 Response:\n{response}\n")
        print("-" * 70)


def main():
    """
    Main test function.
    """
    print("=" * 70)
    print("🧪 Testing Fine-tuned Model")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model()
    
    # Run predefined tests
    run_tests(model, tokenizer)
    
    # Interactive mode
    try:
        interactive_mode(model, tokenizer)
    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted. Goodbye!")


if __name__ == "__main__":
    main()
