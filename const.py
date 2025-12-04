import os
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

from dotenv import load_dotenv

load_dotenv()

#This file defines constants to be used across the project.

_client = None
_embedding_model = None
_tokenizer = None
_llama_model = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        _client = OpenAI(api_key)
    return _client

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
    return _embedding_model

def get_llama_tokenizer_and_model():
    """
    Returns cached (tokenizer, model) for a HuggingFace Llama model.
    Loads lazily, using GPU if available.

    Environment variable: LLAMA_MODEL (defaults to meta-llama/Llama-3.2-3B-Instruct)
    """
    global _tokenizer, _llama_model

    if _tokenizer is None or _llama_model is None:
        llama_name = os.getenv("LLAMA_MODEL")

        _tokenizer = AutoTokenizer.from_pretrained(llama_name, local_files_only = True, padding_side = "left")
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        _llama_model = AutoModelForCausalLM.from_pretrained(llama_name, dtype = "auto", device_map = "auto", local_files_only = True)

        _llama_model = _llama_model.to("cuda" if torch.cuda.is_available() else "cpu")
        
    return _tokenizer, _llama_model
