"""
Streamlit Chat Interface for Python/ML Assistant
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import config


@st.cache_resource
def load_model():
    """
    Load the fine-tuned model (cached for performance).
    """
    model_path = config.TRAINING_CONFIG["output_dir"]
    base_model_name = config.MODEL_CONFIG["base_model"]
    
    # Configure 4-bit quantization
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
    
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str) -> str:
    """
    Generate a response for the user's question.
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
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    return response


def main():
    """
    Main Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title=config.STREAMLIT_CONFIG["page_title"],
        page_icon=config.STREAMLIT_CONFIG["page_icon"],
        layout="wide",
    )
    
    # Title and description
    st.title(f"{config.STREAMLIT_CONFIG['page_icon']} {config.STREAMLIT_CONFIG['page_title']}")
    st.markdown("**Ask me anything about Python programming, Machine Learning, or MLOps!**")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            "This is a fine-tuned Small Language Model specialized in Python and Machine Learning topics. "
            "It has been trained on Python programming, ML algorithms, MLOps practices, and data science concepts."
        )
        
        st.header("Example Questions")
        example_questions = [
            "What is a Python decorator?",
            "Explain overfitting in machine learning",
            "How do I handle missing data in pandas?",
            "What is the difference between supervised and unsupervised learning?",
            "How do I create a virtual environment?",
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}"):
                st.session_state.example_question = q
        
        st.header("Settings")
        temperature = st.slider(
            "Temperature (creativity)",
            min_value=0.1,
            max_value=1.0,
            value=config.INFERENCE_CONFIG["temperature"],
            step=0.1,
            help="Higher values make output more creative but less focused"
        )
        
        max_tokens = st.slider(
            "Max response length",
            min_value=50,
            max_value=512,
            value=config.INFERENCE_CONFIG["max_new_tokens"],
            step=50,
        )
        
        # Update config with user settings
        config.INFERENCE_CONFIG["temperature"] = temperature
        config.INFERENCE_CONFIG["max_new_tokens"] = max_tokens
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Load model
    try:
        with st.spinner("Loading model... This may take a minute on first load."):
            model, tokenizer = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure you've run the training script first: `python 2_finetune_model.py`")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle example question from sidebar
    if "example_question" in st.session_state:
        user_input = st.session_state.example_question
        del st.session_state.example_question
    else:
        user_input = None
    
    # Chat input
    if prompt := (user_input or st.chat_input("Ask a question about Python or Machine Learning...")):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(model, tokenizer, prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
