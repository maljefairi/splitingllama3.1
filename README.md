# Cryptography AGI System

This repository contains a specialized Artificial General Intelligence (AGI) system for cryptography. The system processes queries related to cryptographic concepts, retrieves relevant information, and generates appropriate responses using a fine-tuned language model.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Dependencies](#dependencies)
6. [Contributing](#contributing)
7. [License](#license)

## Features

- Specialized cryptography language model based on Meta-Llama-3.1-8B-Instruct
- Efficient information retrieval using FAISS vector database
- Query processing and response generation
- Interactive command-line interface

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cryptography-agi-system.git
   cd cryptography-agi-system
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Hugging Face API key in a `.env` file:
   ```
   API_KEY_HUGGINGFACE=your_api_key_here
   ```

## Usage

The system is set up in three steps:

1. Generate embeddings:
   ```
   python step1_generate_embeddings.py
   ```

2. Extract subject weights and create the cryptography-specific model:
   ```
   python step2_extract_subject_weights.py
   ```

3. Run the AGI system:
   ```
   python step3.py
   ```

When running `step3.py`, you can interact with the system by entering cryptography-related questions. Type 'quit' to exit the program.

## Project Structure

- `step1_generate_embeddings.py`: Generates embeddings for cryptography-related texts and creates a FAISS index.
- `step2_extract_subject_weights.py`: Extracts cryptography-specific weights and creates a specialized model.
- `step3.py`: Main script for running the AGI system and interacting with it.
- `cryptography_model/`: Directory containing the saved cryptography-specific model.
- `crypto_vector_database.index`: FAISS index for efficient similarity search.
- `crypto_metadata.txt`: Metadata for the cryptography texts used in the vector database.
- `cryptography_weights.npy`: Saved cryptography-specific weights.

## Dependencies

python:requirements.txt
startLine: 1
endLine: 5

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.