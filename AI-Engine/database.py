import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def hybrid_search(query, vector_db, k=3):
    """
    Combo A Logic: BGE-M3 Semantic Search + BM25 Keyword Search.
    """
    # --- 1. SEMANTIC SEARCH (BGE-M3) ---
    # No "query: " prefix needed for BGE-M3
    semantic_docs = vector_db.similarity_search(query, k=k)
    
    # --- 2. KEYWORD SEARCH (BM25) ---
    data = vector_db.get()
    
    if not data['documents']:
        return semantic_docs
        
    all_docs = [
        Document(page_content=d, metadata=m) 
        for d, m in zip(data['documents'], data['metadatas'])
    ]
    
    # Initialize BM25 with the original query
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    keyword_docs = bm25_retriever.invoke(query)[:k]
    
    # --- 3. MERGE & DE-DUPLICATE ---
    combined_docs = semantic_docs + keyword_docs
    seen_content = set()
    unique_docs = []
    
    for doc in combined_docs:
        content_key = doc.page_content.strip()
        if content_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_key)
            
    return unique_docs[:k]