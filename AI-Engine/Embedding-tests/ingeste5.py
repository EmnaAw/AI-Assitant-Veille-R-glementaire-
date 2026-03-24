import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- CONFIGURATION ---
DATA_DIR = "./data"
DB_DIR = "./db_vigogne_multilingual_e5"
EMB_MODEL = "intfloat/multilingual-e5-large"

def run_ingestion():
    # 1. Initialize E5 Embedding Model
    print(f"--- Loading Embedding Model: {EMB_MODEL} ---")
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    
    # 2. Setup Text Splitter (Legal text needs smaller chunks for precision)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    
    all_docs = []

    # 3. Process All Files in /data
    print(f"--- Scanning directory: {DATA_DIR} ---")
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        
        try:
            if filename.endswith(".txt"):
                print(f"📄 Processing Text: {filename}")
                loader = TextLoader(file_path, encoding="utf-8")
                all_docs.extend(loader.load())
            
            elif filename.endswith(".pdf"):
                print(f"📕 Processing PDF: {filename}")
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    # 4. Split into Chunks and Add E5 'passage: ' Prefix
    print("--- Splitting and Prefixing ---")
    final_chunks = []
    chunks = text_splitter.split_documents(all_docs)
    
    for chunk in chunks:
        # E5 MUST have 'passage: ' at the start of every chunk
        prefixed_content = f"passage: {chunk.page_content}"
        
        # Create a new Document object with the prefix
        new_doc = Document(
            page_content=prefixed_content,
            metadata=chunk.metadata
        )
        final_chunks.append(new_doc)

    # 5. Clear old DB and Create New One
    if os.path.exists(DB_DIR):
        import shutil
        print(f"🗑️ Deleting old database at {DB_DIR}...")
        shutil.rmtree(DB_DIR)

    print(f"🏗️ Building Vector Store with {len(final_chunks)} chunks...")
    db = Chroma.from_documents(
        documents=final_chunks,
        embedding=emb,
        persist_directory=DB_DIR
    )
    
    print(f"✅ Success! Database created at {DB_DIR}")

if __name__ == "__main__":
    run_ingestion()