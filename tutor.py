from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import torch

# Load the Llama 2 model and tokenizer
@st.cache_resource
def load_llama_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    hf_token = "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"  # Replace with your Hugging Face token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically uses GPU if available
        torch_dtype=torch.float16,  # Use half-precision for performance
        use_auth_token=hf_token,
        low_cpu_mem_usage=True,  # Reduce memory footprint on CPU
    )
    return tokenizer, model

# Generate response
def generate_response(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load model
tokenizer, model = load_llama_model()

# Streamlit app
st.title("A.I Tutor - Llama 2 Powered")
menu = st.sidebar.selectbox("Choose a Section", ["Home", "Chat"])

if menu == "Home":
    st.write("Welcome to the Llama 2 A.I Tutor! Choose 'Chat' to start asking questions.")

elif menu == "Chat":
    st.write("Ask any question to the A.I Tutor:")
    user_input = st.text_area("Your Question", "")
    if st.button("Get Answer") and user_input:
        with st.spinner("Generating response..."):
            response = generate_response(tokenizer, model, user_input)
        st.write("**Response:**")
        st.write(response)
