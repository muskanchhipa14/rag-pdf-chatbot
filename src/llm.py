import os
from langchain_groq import ChatGroq

SUPPORTED_MODELS = {
    "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
    "Llama 3 70B (High Quality)": "llama3-70b-8192",
    "Mixtral 8x7B (Balanced)": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it"
}

def get_llm(model_display_name="Llama 3.1 8B (Fast)", api_key=None, temperature=0.2):
    """
    Initializes and returns the Groq Chat model.
    Falls back to the environment variable GROQ_API_KEY if api_key is not provided.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Groq API Key is missing. Please set the GROQ_API_KEY environment variable or enter it in the sidebar.")
        
    model_id = SUPPORTED_MODELS.get(model_display_name, "llama-3.1-8b-instant")
    
    return ChatGroq(
        api_key=key,
        model=model_id,
        temperature=temperature
    )
