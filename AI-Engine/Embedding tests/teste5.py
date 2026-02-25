import os, time, shutil, re
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from queries import TEST_QUERIES
from database import hybrid_search 

MODEL_NAME = "intfloat/multilingual-e5-base"
DB_DIR = "./db_e5"
FILES = ["data/Articles.txt", "data/Lois.txt", "data/Termesjuridiques.txt"]
BASE_PDF_PATH = "./data/pdfs/"

def run_test():
    if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
    
    all_docs = []
    for f_path in FILES:
        if not os.path.exists(f_path): continue
        with open(f_path, 'r', encoding='utf-8') as f:
            entries = f.read().split("────────────────────────────────────────────────────────────")
        for entry in entries:
            if not entry.strip(): continue
            id_v = (re.search(r"ID:\s*(\d+)", entry) or [None, "0"])[1]
            theme = (re.search(r"Application:\s*(.*)", entry) or [None, "General"])[1].strip()
            pdf = (re.search(r"PDF:\s*(.*)", entry) or [None, ""])[1].strip()
            resume = (re.search(r"Resume:\n(.*)", entry, re.DOTALL) or [None, ""])[1].strip()
            
            p_text = ""
            if pdf and pdf.lower() != "n/a":
                path = os.path.join(BASE_PDF_PATH, theme, pdf)
                if not os.path.exists(path):
                    for r, d, fs in os.walk(BASE_PDF_PATH):
                        # Filter out Thumbs.db and hidden files
                        fs = [f for f in fs if not f.startswith('.') and not f.lower().endswith('.db')]
                        if pdf in fs: path = os.path.join(r, pdf); break
                if os.path.exists(path):
                    try: p_text = " ".join([p.page_content for p in PyPDFLoader(path).load()])
                    except: pass
            all_docs.append(Document(page_content=f"{resume} {p_text}", metadata={"id": id_v, "source": f_path}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    emb = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={'normalize_embeddings': True})
    db = Chroma.from_documents(chunks, emb, persist_directory=DB_DIR, collection_metadata={"hnsw:space": "cosine"})

    hits, total_mrr, total_precision, start_time = 0, 0, 0, time.time()
    print(f"\n--- Detailed Audit for {MODEL_NAME} ---")
    for test in TEST_QUERIES:
        results = hybrid_search(f"query: {test['q']}", db)
        found_sources = [os.path.basename(doc.metadata.get('source', '')).lower() for doc in results]
        expected_filename = os.path.basename(test['expected'].split(' (')[0].strip().lower())

        is_hit, rank_score, relevant_count = False, 0, 0
        for i, s in enumerate(found_sources):
            if expected_filename in s:
                if not is_hit: is_hit, rank_score = True, 1/(i + 1)
                relevant_count += 1
        
        status = " MATCH" if is_hit else " FAIL "
        print(f"{status} | Query: {test['q'][:50]}...")
        if not is_hit:
            print(f"       Expected: {expected_filename} | AI Found: {found_sources}")

        if is_hit: hits += 1
        total_mrr += rank_score
        total_precision += (relevant_count / len(found_sources)) if found_sources else 0

    total_q = len(TEST_QUERIES)
    recall = hits / total_q
    precision = total_precision / total_q
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print(f" METRICS FOR {MODEL_NAME}")
    print(f" Recall@3:    {recall*100:.1f}%")
    print(f" Precision@3: {precision*100:.1f}%")
    print(f" F1-Score:    {f1*100:.1f}%")
    print(f" MRR:         {total_mrr/total_q:.3f}")
    print(f" Latency:     {(time.time()-start_time)/total_q:.3f}s")
    print("="*50)

if __name__ == "__main__":
    run_test()

