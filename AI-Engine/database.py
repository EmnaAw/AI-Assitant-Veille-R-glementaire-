import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def hybrid_search(query, vector_db, k=3):
    # --- 1. SEMANTIC SEARCH (Chroma) ---
    # Get more than k results to allow for better merging later
    semantic_docs = vector_db.similarity_search(query, k=k)
    
    # --- 2. KEYWORD SEARCH (BM25) ---
    data = vector_db.get()
    all_docs = [
        Document(page_content=d, metadata=m) 
        for d, m in zip(data['documents'], data['metadatas'])
    ]
    
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    keyword_docs = bm25_retriever.invoke(query)[:k]
    
    # --- 3. MANUAL MERGE & DE-DUPLICATION ---
    # We combine both lists and keep unique docs based on page_content
    combined_docs = semantic_docs + keyword_docs
    
    seen_content = set()
    unique_docs = []
    
    for doc in combined_docs:
        # Simple cleaning to check for duplicates
        content_hash = doc.page_content.strip()
        if content_hash not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_hash)
            
    # Return the top k unique results
    return unique_docs[:k]