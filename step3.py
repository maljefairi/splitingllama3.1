"""
AGI System for Topic-Specific Query Processing and Response Generation

This script implements an Artificial General Intelligence (AGI) system specialized in a specific topic,
capable of processing queries related to that topic, retrieving relevant information,
and generating appropriate responses. The system utilizes a pre-trained language model fine-tuned
for the specific topic, vector embeddings, and a FAISS index for efficient information retrieval.

Key Components:
1. AGISystem class: The main class that encapsulates the entire AGI system functionality.
2. Vector database: A FAISS index storing embeddings of topic-related text samples for efficient similarity search.
3. Topic-specific model: A pre-trained language model fine-tuned for the specific topic.
4. Embedding generation: Uses the same model as the topic-specific model for consistency.
5. Information retrieval: A process to fetch relevant context based on query similarity.
6. Response generation: Utilizes the topic-specific model to generate appropriate responses.

Dependencies:
- torch: PyTorch library for tensor computations and neural network operations.
- transformers: Hugging Face's transformers library for pre-trained models and tokenizers.
- faiss: Facebook AI Similarity Search for efficient similarity search and clustering.
- numpy: Numerical computing library for array operations.
- os: For file and directory operations.
- dotenv: For loading environment variables.

Class: AGISystem
    Attributes:
        - topic: The specific topic loaded from topic.txt
        - index: FAISS index loaded from '{topic}_vector_database.index'
        - metadata: List of metadata entries loaded from '{topic}_metadata.txt'
        - model: Topic-specific language model
        - tokenizer: Tokenizer corresponding to the topic-specific model
        - device: PyTorch device (CPU or CUDA) for model computations

    Methods:
        - __init__(): Initializes the AGI system, loading necessary models and data.
        - get_embedding(text): Generates an embedding for the input text.
        - retrieve_information(query_embedding, k=5): Retrieves relevant topic information from the vector database.
        - generate_response(query): Processes the query and generates a response using the topic-specific model.

Function: main()
    Implements the main interaction loop for the AGI system, allowing users to input topic-related queries
    and receive responses until they choose to quit.

Usage:
1. Ensure all required dependencies are installed.
2. Place the necessary model files, topic vector database, and metadata in the appropriate locations.
3. Run the script to start the AGI system.
4. Input topic-related queries when prompted and receive generated responses.
5. Enter 'quit' to exit the system.

Note: This system uses the outputs from step1_generate_embeddings.py and step2_extract_subject_weights.py.
Ensure these scripts have been run and their outputs are available before running this script.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
from dotenv import load_dotenv

class AGISystem:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("API_KEY_HUGGINGFACE")

        # Load the topic from topic.txt
        with open('topic.txt', 'r') as f:
            self.topic = f.readlines()[-1].strip()

        # Load topic vector database index
        index_path = os.path.join(self.topic, f'{self.topic}_vector_database.index')
        self.index = faiss.read_index(index_path)
        
        # Load topic metadata
        metadata_path = os.path.join(self.topic, f'{self.topic}_metadata.txt')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = [line.strip() for line in f]
        
        # Load topic-specific model and tokenizer
        model_path = os.path.join(self.topic, f'{self.topic}_model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        self.model.eval()
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        return embedding
    
    def retrieve_information(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        retrieved_texts = [self.metadata[i] for i in indices[0]]
        return ' '.join(retrieved_texts)
    
    def generate_response(self, query):
        query_embedding = self.get_embedding(query)
        context = self.retrieve_information(query_embedding)
        
        input_text = f"{context}\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1000) # was 512
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1000, # was 150
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if 'Answer:' in response:
            response = response.split('Answer:')[1].strip()
        return response

def main():
    print("Initializing Topic-Specific AGI System...")
    agi_system = AGISystem()
    print(f"{agi_system.topic.capitalize()} AGI System initialized successfully.")

    try:
        while True:
            query = input(f"\nEnter your {agi_system.topic}-related question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            print("Generating response...")
            response = agi_system.generate_response(query)
            print(f"Response: {response}")
    except KeyboardInterrupt:
        print("\nExiting the program...")
    finally:
        print(f"Thank you for using the {agi_system.topic.capitalize()} AGI System.")

if __name__ == "__main__":
    main()