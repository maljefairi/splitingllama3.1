# Topic-Specific AGI System

This repository contains a specialized Artificial General Intelligence (AGI) system designed to process queries related to a specific topic, retrieve relevant information, and generate appropriate responses using a fine-tuned language model.

## Features

- Specialized language model based on Meta-Llama-3.1-8B-Instruct
- Efficient information retrieval using FAISS vector database
- Query processing and response generation
- Interactive command-line interface
- Adaptable to different topics (currently set up for robotics and AAOIFI standards)

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/topic-specific-agi-system.git
   cd topic-specific-agi-system
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

2. Extract subject weights and create the topic-specific model:
   ```
   python step2_extract_subject_weights.py
   ```

3. Run the AGI system:
   ```
   python step3.py
   ```

When running `step3.py`, you can interact with the system by entering topic-related questions. Type 'quit' to exit the program.

## Project Structure

- `step1_generate_embeddings.py`: Generates embeddings for topic-related texts and creates a FAISS index.
- `step2_extract_subject_weights.py`: Extracts topic-specific weights and creates a specialized model.
- `step3.py`: Main script for running the AGI system and interacting with it.
- `<topic>_model/`: Directory containing the saved topic-specific model.
- `<topic>_vector_database.index`: FAISS index for efficient similarity search.
- `<topic>_metadata.txt`: Metadata for the topic-related texts used in the vector database.
- `<topic>_weights.npy`: Saved topic-specific weights.
- `topic.txt`: File containing the current topic for the AGI system.

## Dependencies
accelerate==0.34.2
certifi==2024.8.30
charset-normalizer==3.3.2
faiss-cpu==1.8.0.post1
filelock==3.16.1
fsspec==2024.9.0
huggingface-hub==0.25.0
idna==3.10
Jinja2==3.1.4
joblib==1.4.2
MarkupSafe==2.1.5
mpmath==1.3.0
networkx==3.3
numpy==1.26.4
packaging==24.1
pillow==10.4.0
psutil==6.0.0
python-dotenv==1.0.1
PyYAML==6.0.2
regex==2024.9.11
requests==2.32.3
safetensors==0.4.5
scikit-learn==1.5.2
scipy==1.14.1
sentence-transformers==3.1.0
setuptools==75.1.0
sympy==1.13.3
threadpoolctl==3.5.0
tokenizers==0.19.1
torch==2.4.1
tqdm==4.66.5
transformers==4.44.2
typing_extensions==4.12.2
urllib3==2.2.3

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.