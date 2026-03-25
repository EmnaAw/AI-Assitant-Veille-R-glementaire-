import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from database import hybrid_search

# --- CONFIGURATION (Winner: Combo A) ---
DB_DIR = "./db_bge_m3"  # Change this to your BGE database folder
EMB_MODEL = "BAAI/bge-m3"       # The BGE-M3 model
LLM_MODEL = "vigogne-llama-3"

def run_rag():
    # 1. Initialize Models
    print(f"--- Initializing Combo A: {LLM_MODEL} + BGE-M3 ---")
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    
    if not os.path.exists(DB_DIR):
        print(f"❌ Error: Database {DB_DIR} not found. Ensure you ingested data with BGE-M3!")
        return

    db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    print("\n✅ Assistant Prêt (Mode Performance - Combo A).")
    print("(Tapez 'exit' pour quitter)")

    while True:
        user_query = input("\n[VOUS]: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        # 2. Hybrid Retrieval
        print("🔍 Recherche (Semantic BGE + Keyword BM25)...")
        docs = hybrid_search(user_query, db, k=3)

        # 3. Process Context & Sources
        context_parts = []
        sources_found = []
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Inconnu')
            sources_found.append(f"{source} ({'✅ PDF' if source.endswith('.pdf') else '📄 TXT'})")
            
            # No prefix cleaning needed for BGE-M3
            context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")

        context_text = "\n\n".join(context_parts)

        # 4. Prompt
        prompt = f"""### Instruction:
Tu es un expert en veille réglementaire tunisienne. Réponds à la question suivante en utilisant UNIQUEMENT le contexte juridique fourni. 
Si la réponse ne se trouve pas dans le contexte, indique que tu n'as pas l'information.

CONTEXTE JURIDIQUE:
{context_text}

QUESTION:
{user_query}

### Réponse:"""

        # 5. Execution
        print("🤖 Génération de la réponse...\n")
        response = llm.invoke(prompt)

        print("-" * 60)
        print(f"RÉPONSE:\n{response}")
        print("-" * 60)
        print(f"SOURCES UTILISÉES: {', '.join(set(sources_found))}")

if __name__ == "__main__":
    run_rag()