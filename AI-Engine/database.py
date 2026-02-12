import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, DB_DIR, DATA_DIR

def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 1. LOAD IF EXISTS
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("🔍 Loading existing database from disk...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # 2. BUILD NEW (Fixed for Windows Encoding)
    print("🚀 Database not found. Creating new embeddings...")
    
    # We pass the encoding='utf-8' specifically to the TextLoader via loader_kwargs
    loader = DirectoryLoader(
        DATA_DIR, 
        glob="*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'} # <--- THIS FIXES THE CRASH
    )
    
    documents = loader.load()

    # Smart Chunking for Legal texts
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200,
        separators=["\nArticle", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )

def hybrid_search(query, vector_db):
    # Keyword Boost: Look for specific numbers (Articles)
    nums = re.findall(r'\d+', query)
    boosted = []
    if nums:
        # Scan documents for exact number match
        all_data = vector_db.get()
        for i, text in enumerate(all_data['documents']):
            if nums[0] in text:
                boosted.append(Document(page_content=text, metadata=all_data['metadatas'][i]))

    # Semantic Search: Look for the meaning of the question
    semantic = vector_db.similarity_search(query, k=3)
    
    # Merge and remove duplicates
    combined = {d.page_content: d for d in (boosted + semantic)}.values()
    return list(combined)