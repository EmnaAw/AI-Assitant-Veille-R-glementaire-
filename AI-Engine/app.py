import requests
import os
import json
from database import get_vector_db, hybrid_search
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    RAG_LLM_MODEL,
    ollama_headers,
)

def ask_mistral(query, context_docs):
    if not context_docs:
        return "Aucune information trouvée dans les documents fournis."

    # --- STEP 1: DEDUPLICATION (Inchangé) ---
    unique_contexts = {}
    for d in context_docs:
        content = d.page_content.strip()
        source = d.metadata.get('source', 'Document inconnu')
        
        if content not in unique_contexts:
            unique_contexts[content] = [source]
        else:
            if source not in unique_contexts[content]:
                unique_contexts[content].append(source)

    formatted_context = ""
    for content, sources in unique_contexts.items():
        source_label = ", ".join(sources)
        formatted_context += f"[Sources: {source_label}]\n{content}\n\n"

    # --- STEP 2: CONFIGURATION OLLAMA ---
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    
    prompt = f"""[INST] Tu es un Expert Juridique Tunisien. 
OBJECTIF : Fournir UNE SEULE réponse synthétique, précise et très courte.

RÈGLES STRICTES :
1. NE TE RÉPÈTE PAS : Fusionne les informations des différentes sources en un seul paragraphe cohérent.
2. SOIS CONCIS : Ne dépasse pas 5 phrases. Va directement à l'essentiel.
3. PAS D'HALLUCINATION : N'utilise QUE le contexte fourni. Si l'info n'y est pas, dis-le.
4. CITATION : À la fin, liste les sources entre crochets.

CONTEXTE :
{formatted_context}

QUESTION :
{query} [/INST]"""

    # --- STEP 3: APPEL API OLLAMA ---
    payload = {
        "model": RAG_LLM_MODEL,
        "prompt": prompt,
        "stream": False,  # Important pour recevoir la réponse d'un bloc
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
            headers=ollama_headers(),
        )
        response.raise_for_status()
        # Ollama renvoie la réponse dans le champ 'response'
        return response.json()['response']
    except Exception as e:
        return f" Erreur Technique : {str(e)}"

# --- MAIN INTERFACE (Inchangé) ---
if __name__ == "__main__":
    print(" Chargement du cerveau juridique...")
    db = get_vector_db()
    print(" Prêt. (Dédoublonnage activé)")

    while True:
        user_in = input("\n Votre question : ")
        if user_in.lower() in ['exit', 'quit']:
            print("Fermeture...")
            break
        
        docs = hybrid_search(user_in, db)
        answer = ask_mistral(user_in, docs)
        
        print("\n" + "─"*50)
        print(f" RÉPONSE SYNTHÉTIQUE :\n{answer}")
        print("─"*50)
