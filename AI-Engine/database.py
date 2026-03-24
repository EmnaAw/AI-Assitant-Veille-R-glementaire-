import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def hybrid_search(query, vector_db, k=3):
    """
    Combines Semantic Search (E5) and Keyword Search (BM25).
    """
    # --- 1. SEMANTIC SEARCH (E5 Prefix Required) ---
    # We add 'query: ' only for the vector search
    e5_query = f"query: {query}"
    semantic_docs = vector_db.similarity_search(e5_query, k=k)
    
    # --- 2. KEYWORD SEARCH (BM25) ---
    # Get all raw documents from Chroma to build BM25 index
    data = vector_db.get()
    
    if not data['documents']:
        return semantic_docs # Fallback if DB is empty
        
    all_docs = [
        Document(page_content=d, metadata=m) 
        for d, m in zip(data['documents'], data['metadatas'])
    ]
    
    # Initialize BM25 on the fly (uses original query for keyword matching)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    keyword_docs = bm25_retriever.invoke(query)[:k]
    
    # --- 3. MERGE & DE-DUPLICATE ---
    combined_docs = semantic_docs + keyword_docs
    
    seen_content = set()
    unique_docs = []
    
    for doc in combined_docs:
        # Standardize content for comparison
        content_key = doc.page_content.strip()
        if content_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_key)
            
    # Return the top k unique results
    return unique_docs[:k]