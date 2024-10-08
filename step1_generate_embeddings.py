"""
This script generates embeddings for a set of topic-related text samples using a pre-trained language model,
creates a FAISS index for efficient similarity search, and saves both the index and metadata.

Dependencies:
- torch: PyTorch library for tensor computations
- transformers: Hugging Face's transformers library for pre-trained models
- numpy: Numerical computing library
- faiss: Facebook AI Similarity Search for efficient similarity search and clustering
- os: Operating system interface
- dotenv: Module for loading environment variables from a .env file

The script performs the following steps:
1. Load environment variables and API key
2. Initialize the pre-trained model and tokenizer
3. Define a function to generate embeddings for given text
4. Prepare sample data (topic-related texts)
5. Generate embeddings for the sample texts
6. Create and populate a FAISS index with the generated embeddings
7. Save the FAISS index and metadata to files
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face API key from environment variables
api_key = os.getenv("API_KEY_HUGGINGFACE")

# Load the topic from topic.txt
with open('topic.txt', 'r') as f:
    topic = f.readlines()[-1].strip()

# Create a folder with the topic name
os.makedirs(topic, exist_ok=True)

# Initialize the pre-trained model and tokenizer
model_name = 'meta-llama/Meta-Llama-3.1-405B'  # Using Llama 2 70B chat model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
model = AutoModel.from_pretrained(model_name, token=api_key)

# Set the model to evaluation mode
model.eval()

def get_embedding(text):
    """
    Generate an embedding for the given text using the pre-trained model.

    Args:
        text (str): The input text to embed.

    Returns:
        numpy.ndarray: The embedding vector for the input text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding

# Sample data - topic-related texts
texts = [
    f"{topic} is a field with various important concepts and applications.",
    f"There are several key algorithms and techniques used in {topic}.",
    f"Security and privacy are crucial considerations in {topic}.",
    f"{topic} has seen significant advancements in recent years.",
    f"Understanding the fundamentals of {topic} is essential for professionals in the field."
]

# Generate embeddings for each text in the sample data
embeddings = []
for text in texts:
    embedding = get_embedding(text)
    embeddings.append(embedding)

# Stack the embeddings into a 2D numpy array and convert to float32
embeddings = np.vstack(embeddings).astype('float32')

# Initialize FAISS index
dimension = embeddings.shape[1]  # Get the dimensionality of the embeddings
index = faiss.IndexFlatL2(dimension)  # Create a FAISS index using L2 distance
index.add(embeddings)  # Add the embeddings to the index

# Save the FAISS index to a file in the topic folder
index_path = os.path.join(topic, f'{topic}_vector_database.index')
faiss.write_index(index, index_path)
print(f"FAISS index saved to {os.path.abspath(index_path)}")

# Save metadata (texts) to a text file in the topic folder
metadata_path = os.path.join(topic, f'{topic}_metadata.txt')
with open(metadata_path, 'w', encoding='utf-8') as f:
    for text in texts:
        f.write(f"{text}\n")
print(f"Metadata saved to {os.path.abspath(metadata_path)}")

print(f"{topic} embeddings and metadata have been saved successfully.")
print("Current working directory:", os.getcwd())
print(f"Files in the {topic} directory:", os.listdir(topic))
