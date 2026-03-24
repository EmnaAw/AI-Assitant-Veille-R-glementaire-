import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from database import hybrid_search

# --- CONFIGURATION (Winner: Combo C) ---
DB_DIR = "./db_vigogne_multilingual_e5"
EMB_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "mistral:v0.3"

def run_rag():
    # 1. Initialize Models
    print(f"--- Initializing Combo C: {LLM_MODEL} + E5 ---")
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    
    # Load the existing Vector DB
    if not os.path.exists(DB_DIR):
        print(f"❌ Error: Database directory {DB_DIR} not found. Run ingestion first!")
        return

    db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    print("\n✅ Assistant Prêt. Posez vos questions juridiques.")
    print("(Tapez 'exit' pour quitter)")

    while True:
        user_query = input("\n[VOUS]: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        # 2. Hybrid Retrieval (Semantic E5 + Keyword BM25)
        print("🔍 Recherche dans la base de données...")
        docs = hybrid_search(user_query, db, k=3)

        # 3. Process Context & Verify PDF Source
        context_parts = []
        pdf_sources = []
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Inconnu')
            is_pdf = "✅ PDF" if source.endswith('.pdf') else "📄 TXT"
            pdf_sources.append(f"{source} ({is_pdf})")
            
            # Clean 'passage: ' prefix for the LLM if it exists
            content = doc.page_content.replace("passage: ", "", 1)
            context_parts.append(f"[Source {i+1}: {source}]\n{content}")

        context_text = "\n\n".join(context_parts)

        # 4. Professional Legal Prompt
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
        print(f"SOURCES UTILISÉES: {', '.join(set(pdf_sources))}")

if __name__ == "__main__":
    run_rag()