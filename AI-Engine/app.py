import requests
import os
from config import MISTRAL_API_KEY
from database import get_vector_db, hybrid_search

def ask_mistral(query, context_docs):
    if not context_docs:
        return "❌ Aucune information trouvée dans les documents fournis."

    # --- STEP 1: DEDUPLICATION ---
    # This prevents the AI from seeing the same text 3 times if it exists in 3 files
    unique_contexts = {}
    for d in context_docs:
        content = d.page_content.strip()
        source = d.metadata.get('source', 'Document inconnu')
        
        if content not in unique_contexts:
            unique_contexts[content] = [source]
        else:
            # If text is same, just add the new source name to the list
            if source not in unique_contexts[content]:
                unique_contexts[content].append(source)

    # Reconstruct a single clean context string
    formatted_context = ""
    for content, sources in unique_contexts.items():
        source_label = ", ".join(sources)
        formatted_context += f"[Sources: {source_label}]\n{content}\n\n"

    # --- STEP 2: THE "STRICT & SHORT" PROMPT ---
    prompt = f"""[INST] Tu es un Expert Juridique Tunisien. 
OBJECTIF : Fournir UNE SEULE réponse synthétique, précise et très courte.

RÈGLES STRICTES :
1. NE TE RÉPÈTE PAS : Fusionne les informations des différentes sources en un seul paragraphe cohérent.
2. SOIS CONCIS : Ne dépasse pas 5 phrases. Va directement à l'essentiel.
3. PAS D'HALLUCINATION : N'utilise QUE les documents fournis. Si l'info n'y est pas, dis-le.
4. CITATION : À la fin de ta réponse, liste les sources utilisées entre crochets.

CONTEXTE :
{formatted_context}

QUESTION :
{query} [/INST]"""

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={
                "model": "open-mistral-7b", 
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.0  # Zero creativity = High accuracy
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Erreur Technique : {str(e)}"

# --- MAIN INTERFACE ---
if __name__ == "__main__":
    print("⏳ Chargement du cerveau juridique...")
    db = get_vector_db()
    print("✅ Prêt. (Dédoublonnage activé)")

    while True:
        user_in = input("\n👉 Votre question : ")
        if user_in.lower() in ['exit', 'quit']:
            print("Fermeture...")
            break
        
        # 1. Hybrid Search (finds chunks)
        docs = hybrid_search(user_in, db)
        
        # 2. Ask AI (Merges & Cleans response)
        answer = ask_mistral(user_in, docs)
        
        print("\n" + "─"*50)
        print(f"⚖️ RÉPONSE SYNTHÉTIQUE :\n{answer}")
        print("─"*50)