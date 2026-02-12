import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- 1. DATA LOADING ---
print("--- Step 1: Loading Data ---")
try:
    # Look for the CSV in the current directory
    path = 'preventijsveille_clean.csv'
    if not os.path.exists(path):
        path = os.path.join('sample_data', 'preventijsveille_clean.csv')
    
    df = pd.read_csv(path)
    # Cleaning: Fill empty spots and force text type
    df['resume'] = df['resume'].fillna('').astype(str)
    df = df[df['resume'].str.strip() != '']
    print(f"✅ Loaded {len(df)} rows.")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")

# --- 2. TEXT CHUNKING ---
print("\n--- Step 2: Splitting Text ---")
# This stays the same to ensure legal context isn't lost
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

docs = []
if 'df' in locals() and not df.empty:
    for index, row in df.iterrows():
        chunks = text_splitter.split_text(row['resume'])
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "id": str(row['id']),
                    "titre": str(row['titre']),
                    "source": str(row['journal'])
                }
            ))
    print(f"✅ Created {len(docs)} chunks from your laws.")

# --- 3. VECTOR DB & FAST MODEL ---
print("\n--- Step 3: Building the Brain (Embeddings) ---")

# FAST MODEL: This is 4x faster on CPU than the E5-base model
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"Loading fast model: {model_name}...")
embeddings = HuggingFaceEmbeddings(model_name=model_name)

if 'docs' in locals() and docs:
    # This creates a local folder 'preventis_db' so you don't have to re-run this every time
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory="./preventis_db"
    )
    print("✅ Vector Database is ready.")
else:
    print("❌ No documents to process.")

# --- 4. THE SEARCH TEST ---
print("\n--- Step 4: Testing the POC ---")
if 'vector_db' in locals():
    query = "Quelles sont les trois catégories d'établissements ?"
    # Note: No "query: " prefix needed for this model
    results = vector_db.similarity_search(query, k=2)

    for i, res in enumerate(results):
        print(f"\n[MATCH {i+1}]")
        print(f"TITLE: {res.metadata['titre']} | SOURCE: {res.metadata['source']}")
        print(f"EXCERPT: {res.page_content[:300]}...")