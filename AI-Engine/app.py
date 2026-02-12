import requests
import os
from config import MISTRAL_API_KEY
from database import get_vector_db, hybrid_search

def ask_mistral(query, context_docs):
    if not context_docs:
        return "❌ Aucun document pertinent trouvé dans la base de données."

    context_text = "\n\n".join([f"[{d.metadata.get('source')}]: {d.page_content}" for d in context_docs])
    
    prompt = f"""[INST] Tu es un Expert Juridique Tunisien. 
RÈGLES : 
1. Réponds UNIQUEMENT via le CONTEXTE. 
2. Si inconnu, dis que l'info n'est pas dans les documents.
3. Cite la source.

CONTEXTE :
{context_text}

QUESTION :
{query} [/INST]"""

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={
                "model": "open-mistral-7b", 
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.0
            },
            timeout=10 # Stop waiting after 10 seconds
        )
        response.raise_for_status() # Check for HTTP errors
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Erreur API : {str(e)}"

# --- THIS PART IS REQUIRED TO RUN IN TERMINAL ---
if __name__ == "__main__":
    print("⏳ Initialisation de la base de données...")
    db = get_vector_db()
    print("✅ Base de données prête.")

    while True:
        user_in = input("\n👉 Posez votre question (ou 'exit') : ")
        if user_in.lower() in ['exit', 'quit']:
            break
        
        print("🔍 Recherche dans les documents...")
        docs = hybrid_search(user_in, db)
        
        print("🤖 Réflexion de Mistral...")
        answer = ask_mistral(user_in, docs)
        
        print(f"\n⚖️ RÉPONSE :\n{answer}")