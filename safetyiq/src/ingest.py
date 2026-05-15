"""
SafetyIQ Ingestion Pipeline
1. Load PDFs
2. Split into chunks
3. Embed using OpenAI
4. Store in ChromaDB
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Safety check
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found. Add it to your .env file."
    )

PDF_FOLDER = "data/pdfs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "safetyiq_docs"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200



# Load PDFs
def load_pdfs(folder_path):
    documents = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError("No PDF files found")

    print(f"Found {len(pdf_files)} PDFs")

    for file in pdf_files:
        path = os.path.join(folder_path, file)
        print(f"Loading {file}")

        loader = PyMuPDFLoader(path)
        docs = loader.load()

        print(f"Loaded {len(docs)} pages")
        documents.extend(docs)

    print(f"Total pages: {len(documents)}")
    return documents


# Chunking

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )

    return splitter.split_documents(documents)



# Embedding + Storage

def embed_and_store(chunks):

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    print(f"Embedding {len(chunks)} chunks...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )


    print("\nIngestion complete!")
  
    print(f"Vectors: {vectorstore._collection.count()}")





if __name__ == "__main__":
    print("===== SafetyIQ Ingestion =====")

    docs = load_pdfs(PDF_FOLDER)
    chunks = split_documents(docs)
    embed_and_store(chunks)