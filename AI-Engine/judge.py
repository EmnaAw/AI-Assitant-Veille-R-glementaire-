import re
from langchain_ollama import OllamaLLM

# DeepSeek-R1 is the judge due to its reasoning capabilities
JUDGE_MODEL = "deepseek-r1:8b" 

def evaluate_answer(context, question, generated_answer):
    llm = OllamaLLM(model=JUDGE_MODEL, temperature=0)
    
    prompt = f"""
    ### ROLE: Expert Juridique Tunisien
    ### TÂCHE: Évaluer la qualité d'une réponse RAG.
    
    CONTEXTE RÉCUPÉRÉ:
    {context}
    
    QUESTION:
    {question}
    
    RÉPONSE GÉNÉRÉE:
    {generated_answer}
    
    ### INSTRUCTIONS DE NOTATION (Score de 1 à 5):
    1. Faithfulness (F): La réponse est-elle 100% basée sur le contexte? (Pas d'hallucination)
    2. Relevance (R): La réponse répond-elle directement à la question?
    3. Precision (P): Le modèle a-t-il extrait les bons chiffres/articles?
    4. Consistency (C): Le ton est-il formel et administratif (Droit Tunisien)?
    
    Réponds UNIQUEMENT sous ce format EXACT: F:X, R:X, P:X, C:X
    """
    
    try:
        raw_response = llm.invoke(prompt)
        # Remove reasoning tags if the judge is DeepSeek-R1
        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        scores = re.findall(r'[FRPC]:\s*(\d)', clean_response)
        if len(scores) == 4:
            return [int(s) for s in scores]
        return [0, 0, 0, 0]
    except:
        return [0, 0, 0, 0]