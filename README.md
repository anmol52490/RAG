# RAG-Powered Chatbot with Chroma and Groq

**RAG-Powered Chatbot with Embeddings** is a project designed to build an intelligent chatbot that interacts with users by understanding and retrieving information from a large set of documents. The system utilizes advanced techniques to enhance the chatbot's responses based on the document content it has access to.

## Main Objectives

- **Compare Text Embeddings**: Use the Groq API to measure the similarity between different pieces of text.
- **Manage a Document Database**: Create and maintain a document database using Chroma, with text embedded by HuggingFace models.
- **Interactive Chatbot**: Allow users to ask questions, retrieve relevant information from the database, and get responses generated through Groq.

## Key Functionalities

- **Text Similarity**: Compare and analyze the similarity between different texts.
- **Database Creation**: Set up and manage a database for storing and retrieving documents.
- **Contextual Responses**: Query the database and provide answers based on the content of the documents.

## Overview

The project consists of three main scripts:

1. **`compare_embedding.py`**: Compares embeddings for text and measures similarity using the Groq API.
2. **`create_database.py`**: Creates and manages a document database with Chroma, using HuggingFace embeddings.
3. **`query_system.py`**: Queries the database and interacts with the Groq API to answer questions based on document context.

## Features

- Compare embeddings for text and measure similarity.
- Create and manage a document database with Chroma.
- Query the database and interact with the Groq API.
- Use HuggingFace embeddings for text representation.

## Technologies Used

- **Embeddings**: HuggingFace Transformers
- **Database**: Chroma
- **API**: Groq
- **Environment Variables**: dotenv

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- **Python version**: Ensure you have Python 3.8 or higher installed.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anmol52490/RAG.git
   cd your-repository
   ```

2. **Install Python dependencies:**
   Create a `requirements.txt` file with the following content:
   ```plaintext
   dotenv
   langchain
   transformers
   langchain-chroma
   langchain-huggingface
   groq
   ```
   Then, install the dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root with the following content:
   ```env
   GROQ_API_KEY=your_groq_api_key
   CACHE_DIR=./default_cache_path
   ```

4. **Run the scripts:**
   
   - **To compare embeddings:**
     ```bash
     python compare_embedding.py
     ```

   - **To create and manage the document database:**
     ```bash
     python create_database.py
     ```

   - **To query the database and interact with the Groq API:**
     ```bash
     python query_system.py
     ```

## Contributors

- Anmol Bhusal

## Acknowledgements

Thanks to HuggingFace for providing powerful models and to Groq for their API. Special thanks to LangChain for their extensive libraries and tools.
```

