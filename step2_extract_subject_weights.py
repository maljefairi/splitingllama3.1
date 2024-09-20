"""
Step 2: Extract Cryptography Weights and Create Cryptography-Specific Model

This script uses the outputs from step1_generate_embeddings.py to create a cryptography-specific model.

Dependencies:
- torch: PyTorch library for tensor computations
- transformers: Hugging Face's transformers library for pre-trained models
- numpy: Numerical computing library
- faiss: Facebook AI Similarity Search for efficient similarity search and clustering
- os: Operating system interface
- dotenv: Module for loading environment variables from a .env file

The script follows these main steps:
1. Load environment variables and API key
2. Initialize the pre-trained model and tokenizer
3. Load the FAISS index and metadata created in step 1
4. Extract cryptography-specific weights using the loaded index and metadata
5. Create and save a cryptography-specific model

Usage:
1. Ensure that step1_generate_embeddings.py has been run and its outputs are available.
2. Run this script to extract cryptography weights and create a cryptography-specific model.
3. The resulting cryptography weights will be saved in 'cryptography_weights.npy'.
4. The cryptography-specific model will be saved in a directory named 'cryptography_model'.

Note: This script assumes that the necessary environment variables and API keys are properly set up.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import faiss
import os
from dotenv import load_dotenv

# Set environment variable to handle OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load environment variables
load_dotenv()

# Retrieve the Hugging Face API key from environment variables
api_key = os.getenv("API_KEY_HUGGINGFACE")

# Initialize the pre-trained model and tokenizer
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=api_key,
    device_map="auto",  # Automatically distribute across available GPUs
    torch_dtype=torch.float16,  # Use half precision to reduce memory usage
    low_cpu_mem_usage=True,
    offload_folder="offload"  # Enable disk offloading
)

# Load the FAISS index and metadata from step 1
index = faiss.read_index("crypto_vector_database.index")
with open("crypto_metadata.txt", "r") as f:
    metadata = [line.strip() for line in f]

# Function to extract cryptography-specific weights
def extract_cryptography_weights():
    # Get all embeddings from the index
    total_vectors = index.ntotal
    d = index.d
    _, all_embeddings = index.search(np.zeros((1, d), dtype=np.float32), k=total_vectors)
    
    # Compute the centroid (average) of all embeddings
    crypto_centroid = np.mean(all_embeddings, axis=1)
    
    # Normalize the centroid
    crypto_weights = crypto_centroid / np.linalg.norm(crypto_centroid)
    
    return crypto_weights.squeeze()

# Extract weights for cryptography
print("Extracting weights for cryptography...")
crypto_weights = extract_cryptography_weights()

# Save cryptography-specific weights
np.save("cryptography_weights.npy", crypto_weights)
print("Cryptography-specific weights saved to cryptography_weights.npy")

# Function to create a cryptography-specific model
def create_cryptography_model(crypto_weights):
    # Define device map for efficient memory usage
    device_map = "auto"
    
    # Load the original model with gradient checkpointing enabled
    crypto_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=api_key, 
        device_map=device_map,
        offload_folder="offload",
        torch_dtype=torch.float16
    )
    crypto_model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    
    # Apply cryptography weights to the model's embedding layer
    with torch.no_grad():
        # Ensure crypto_weights has the correct shape
        crypto_weights = torch.tensor(crypto_weights, dtype=torch.float16)
        if crypto_weights.dim() == 0:  # If it's a scalar
            crypto_weights = crypto_weights.expand(crypto_model.model.embed_tokens.weight.shape)
        elif crypto_weights.dim() == 1:  # If it's a 1D tensor
            crypto_weights = crypto_weights.unsqueeze(1).expand(crypto_model.model.embed_tokens.weight.shape)
        
        print(f"crypto_weights shape: {crypto_weights.shape}")
        print(f"embed_tokens weight shape: {crypto_model.model.embed_tokens.weight.shape}")
        
        # Ensure the shapes match
        if crypto_weights.shape != crypto_model.model.embed_tokens.weight.shape:
            raise ValueError(f"Shape mismatch: crypto_weights {crypto_weights.shape} vs embed_tokens {crypto_model.model.embed_tokens.weight.shape}")
        
        # Apply the weights
        crypto_model.model.embed_tokens.weight *= crypto_weights.to(crypto_model.model.embed_tokens.weight.device)
    
    return crypto_model

# Create and save cryptography-specific model
print("Creating cryptography-specific model...")
crypto_model = create_cryptography_model(crypto_weights)
print(crypto_model)  # Print model structure for debugging

# Save the model in smaller chunks
print("Saving model...")
crypto_model.save_pretrained(
    "cryptography_model", 
    safe_serialization=True,
    max_shard_size="500MB"  # Adjust this value based on available memory
)
tokenizer.save_pretrained("cryptography_model")
print("Cryptography model saved.")

print("Cryptography-specific model has been created and saved.")
