import os
import shutil
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Define paths for Chroma database and data files
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Load pre-trained model and tokenizer
model_id = "distilbert-base-uncased"
model = AutoModel.from_pretrained(model_id, cache_dir="C:/Users/muska/Desktop/RAG_project/RAG")
tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False, cache_dir="C:/Users/muska/Desktop/RAG_project/RAG")

def main():
    try:
        generate_data_store()
    except Exception as e:
        print(f"Error: {e}")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return [document for document in documents]

def split_text(documents: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = []
    for document in documents:
        if hasattr(document, 'page_content'):
            chunked_text = text_splitter.split_text(document.page_content)
            for chunk in chunked_text:
                new_document = Document(page_content=chunk, metadata=document.metadata)
                chunks.append(new_document)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"Cleared existing Chroma database at {CHROMA_PATH}.")

        hf_embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

        db = Chroma.from_documents(chunks, hf_embeddings, persist_directory=CHROMA_PATH)
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
