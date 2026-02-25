import re
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_core.documents import Document

def hybrid_search(query, vector_db, k=3):
    # 1. Extract data from Chroma
    data = vector_db.get()
    
    # 2. Reconstruct Document objects
    docs = [
        Document(page_content=d, metadata=m) 
        for d, m in zip(data['documents'], data['metadatas'])
    ]
    
    # 3. Keyword Retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    
    # 4. Semantic Retriever
    chroma_retriever = vector_db.as_retriever(search_kwargs={"k": k})
    
    # 5. Combine
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], 
        weights=[0.3, 0.7]
    )
    
    return ensemble.invoke(query)