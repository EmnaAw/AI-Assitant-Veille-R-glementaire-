import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
DATA_DIR = "./data"
DB_DIR = "./db_vigogne_bge_m3"
EMB_MODEL = "BAAI/bge-m3"

def ingest_data():
    documents = []
    
    # 1. Load ALL files (PDF + TXT)
    print(f"📂 Scanning {DATA_DIR}...")
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        
        try:
            if file.endswith(".pdf"):
                print(f" Loading PDF: {file}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                print(f" Loading Text: {file}")
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            print(f" Could not load {file}: {e}")

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f" Created {len(chunks)} chunks.")

    # 3. Create Vector DB with BGE-M3
    print(f" Embedding with {EMB_MODEL} (Combo A)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f" Database created at {DB_DIR}")

if __name__ == "__main__":
    ingest_data()