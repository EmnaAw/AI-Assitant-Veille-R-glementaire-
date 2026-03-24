import time
from langchain_ollama import OllamaLLM
from database import hybrid_search
from queries import TEST_QUERIES
from judge import evaluate_answer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

MODEL_TAG = "mistral:v0.3"
EMB_MODEL = "BAAI/bge-m3"
DB_PATH = "./db_bge_m3"

def run():
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    db = Chroma(persist_directory=DB_PATH, embedding_function=emb)
    llm = OllamaLLM(model=MODEL_TAG, temperature=0)
    
    scores_list = []
    print(f"\n--- ÉVALUATION EN COURS: {MODEL_TAG} + {EMB_MODEL} ---")

    for i, item in enumerate(TEST_QUERIES):
        start = time.time()
        docs = hybrid_search(item['question'], db, k=2)
        context = "\n".join([d.page_content for d in docs])
        
        # Mistral [INST] Format
        ans = llm.invoke(f"[INST] Contexte: {context}\nQuestion: {item['question']} [/INST]")
        lat = time.time() - start
        print(f"\n{'='*10} QUESTION {i+1} {'='*10}") 
        print(f"Q: {item['question']}") 
        print(f"A: {ans}")
        
        f, r, p, c = evaluate_answer(context, item['question'], ans)
        scores_list.append({'f': f, 'r': r, 'p': p, 'c': c, 't': lat})
        print(f"✔️ Terminé: {item['question'][:30]}... | F:{f} R:{r} P:{p} C:{c}")

    avg = lambda k: sum(x[k] for x in scores_list) / len(scores_list)
    print(f"\n{'='*20} BILAN {MODEL_TAG} + {EMB_MODEL} {'='*20}")
    print(f"Faithfulness: {avg('f'):.2f}/5 | Relevance: {avg('r'):.2f}/5")
    print(f"Precision:    {avg('p'):.2f}/5 | Consistency: {avg('c'):.2f}/5")
    print(f"Latence Moyenne: {avg('t'):.2f}s\n{'='*55}")

if __name__ == "__main__":
    run()
