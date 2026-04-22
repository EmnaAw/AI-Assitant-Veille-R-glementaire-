import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from database import hybrid_search

# --- CONFIGURATION ---
DB_DIR = "./db_vigogne_bge_m3" 
EMB_MODEL = "BAAI/bge-m3"
LLM_MODEL = "vigogne-llama-3"

def run_rag():
    # 1. Initialize Models
    print(f"--- Initializing: {LLM_MODEL} + BGE-M3 ---")
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    
    if not os.path.exists(DB_DIR):
        print(f" Error: Database {DB_DIR} not found.")
        return

    db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    print("\n Assistant Prêt à répondre à vos questions sur la réglementation tunisienne.")
    print("(Tapez 'exit' pour quitter)")

    while True:
        user_query = input("\n[VOUS]: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        # 2. Hybrid Retrieval
        print("🔍 Recherche en cours...")
        docs = hybrid_search(user_query, db, k=6)

        # 3. Process Context (Content only, no metadata/sources)
        context_parts = []
        for doc in docs:
            # We only extract the text, ignoring 'source' in metadata
            context_parts.append(doc.page_content)

        context_text = "\n\n".join(context_parts)

        # 4. Strict Prompt (Instructs AI to avoid mentioning sources/files)
        prompt = f"""### Instruction:
Tu es un expert en veille réglementaire tunisienne. Réponds à la question suivante en utilisant le contexte juridique fourni.
Donne une réponse directe et professionnelle. Ne mentionne jamais de noms de fichiers, de numéros de sources, ou de références aux documents fournis dans ta réponse.

CONTEXTE JURIDIQUE:
{context_text}

QUESTION:
{user_query}

### Réponse:"""

        # 5. Execution
        print(" Génération de la réponse...\n")
        response = llm.invoke(prompt)

        # Clean Output
        print("-" * 60)
        print(f"RÉPONSE:\n{response}")
        print("-" * 60)

if __name__ == "__main__":
    run_rag()