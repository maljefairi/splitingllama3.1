"""
Step 2: Extract Subject Weights and Create Subject-Specific Model

This script uses the outputs from step1_generate_embeddings.py to create a subject-specific model.

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
4. Extract subject-specific weights using the loaded index and metadata
5. Create and save a subject-specific model

Usage:
1. Ensure that step1_generate_embeddings.py has been run and its outputs are available.
2. Run this script to extract subject weights and create a subject-specific model.
3. The resulting subject weights and model will be saved in the subject-specific folder.

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

# Load the topic from topic.txt
with open('topic.txt', 'r') as f:
    topic = f.readlines()[-1].strip()

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
index_path = os.path.join(topic, f'{topic}_vector_database.index')
index = faiss.read_index(index_path)
metadata_path = os.path.join(topic, f'{topic}_metadata.txt')
with open(metadata_path, "r") as f:
    metadata = [line.strip() for line in f]

# Function to extract subject-specific weights
def extract_subject_weights():
    # Get all embeddings from the index
    total_vectors = index.ntotal
    d = index.d
    _, all_embeddings = index.search(np.zeros((1, d), dtype=np.float32), k=total_vectors)
    
    # Compute the centroid (average) of all embeddings
    subject_centroid = np.mean(all_embeddings, axis=1)
    
    # Normalize the centroid
    subject_weights = subject_centroid / np.linalg.norm(subject_centroid)
    
    return subject_weights.squeeze()

# Extract weights for the subject
print(f"Extracting weights for {topic}...")
subject_weights = extract_subject_weights()

# Save subject-specific weights
weights_path = os.path.join(topic, f'{topic}_weights.npy')
np.save(weights_path, subject_weights)
print(f"{topic}-specific weights saved to {weights_path}")

# Function to create a subject-specific model
def create_subject_model(subject_weights):
    # Define device map for efficient memory usage
    device_map = "auto"
    
    # Load the original model with gradient checkpointing enabled
    subject_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=api_key, 
        device_map=device_map,
        offload_folder="offload",
        torch_dtype=torch.float16
    )
    subject_model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    
    # Apply subject weights to the model's embedding layer
    with torch.no_grad():
        # Ensure subject_weights has the correct shape
        subject_weights = torch.tensor(subject_weights, dtype=torch.float16)
        if subject_weights.dim() == 0:  # If it's a scalar
            subject_weights = subject_weights.expand(subject_model.model.embed_tokens.weight.shape)
        elif subject_weights.dim() == 1:  # If it's a 1D tensor
            subject_weights = subject_weights.unsqueeze(1).expand(subject_model.model.embed_tokens.weight.shape)
        
        print(f"subject_weights shape: {subject_weights.shape}")
        print(f"embed_tokens weight shape: {subject_model.model.embed_tokens.weight.shape}")
        
        # Ensure the shapes match
        if subject_weights.shape != subject_model.model.embed_tokens.weight.shape:
            raise ValueError(f"Shape mismatch: subject_weights {subject_weights.shape} vs embed_tokens {subject_model.model.embed_tokens.weight.shape}")
        
        # Apply the weights
        subject_model.model.embed_tokens.weight *= subject_weights.to(subject_model.model.embed_tokens.weight.device)
    
    return subject_model

# Create and save subject-specific model
print(f"Creating {topic}-specific model...")
subject_model = create_subject_model(subject_weights)
print(subject_model)  # Print model structure for debugging

# Save the model in smaller chunks
print("Saving model...")
model_path = os.path.join(topic, f'{topic}_model')
subject_model.save_pretrained(
    model_path, 
    safe_serialization=True,
    max_shard_size="500MB"  # Adjust this value based on available memory
)
tokenizer.save_pretrained(model_path)
print(f"{topic} model saved to {model_path}")

print(f"{topic}-specific model has been created and saved.")
