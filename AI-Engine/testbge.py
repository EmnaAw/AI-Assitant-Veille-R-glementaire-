import os
import time
import shutil
import re
from langchain_huggingface import HuggingFaceEmbeddings
from database import hybrid_search 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION (BGE-M3 VERSION) ---
# BAAI/bge-m3 is the identifier for the Multi-Lingual, Multi-Functionality, Multi-Granularity model
MODEL_NAME = "BAAI/bge-m3"
DB_DIR = "./db_bgem3" 
FILES = ["data/Articles.txt", "data/Lois.txt", "data/Termesjuridiques.txt"]

# Your "Gold Standard" for testing
TEST_QUERIES = [
    {"q": "Comment sont classés les établissements dangereux en trois catégories ?", "expected": "data/Articles.txt (Article 294)"},
    {"q": "Quelles sont les missions de l'Institut National de la Météorologie (INM) ?", "expected": "data/Lois.txt (Loi n° 2009-10)"},
    {"q": "Quels sont les tarifs des redevances pour l'approbation de modèles d'instruments ?", "expected": "data/Termesjuridiques.txt (Décret n° 2001-588)"},
    {"q": "Quelle est la définition de la démission dans la Convention des Teintureries ?", "expected": "data/Articles.txt (Article 58)"},
    {"q": "Quelle loi définit les unités de mesure légales en Tunisie ?", "expected": "data/Lois.txt (Loi n° 99-40)"},
    {"q": "Quelles sont les obligations de surveillance pour les détenteurs de balances automatiques ?", "expected": "data/Termesjuridiques.txt (Obligations des détenteurs)"},
    {"q": "Quels sont les domaines de protection visés par l'article 293 ?", "expected": "data/Articles.txt (Article 293)"},
    {"q": "Quelle est la validité temporelle des étiquettes de salubrité des fruits de mer ?", "expected": "data/Lois.txt (Loi n° 59-56 / Article 21)"},
    {"q": "Quelles sont les normes de carrelage pour les usines de conserves alimentaires ?", "expected": "data/Termesjuridiques.txt (Décret n° 68-228)"},
    {"q": "Quelles mentions doivent figurer exclusivement sur un certificat de travail ?", "expected": "data/Articles.txt (Article 59)"},
    {"q": "Quel organisme est chargé de la normalisation et de la propriété industrielle ?", "expected": "data/Lois.txt (Loi n° 82-66 / INNORPI)"},
    {"q": "Quelle est la marque de poinçonnage utilisée pour l'année 2017 ?", "expected": "data/Termesjuridiques.txt (Arrêté Poinçonnage 2017)"},
    {"q": "Quelle est la sanction pour l'occupation sans permission de la voie publique ?", "expected": "data/Articles.txt (Code Pénal)"},
    {"q": "Quels produits sont exemptés de contrôle vétérinaire à l'importation ?", "expected": "data/Lois.txt (Article 3 / Produits animaux)"},
    {"q": "Quel est le taux de TVA appliqué aux redevances de contrôle métrologique ?", "expected": "data/Termesjuridiques.txt (Arrêté n° 2018-1353)"},
    {"q": "L'indemnité de licenciement est-elle distincte de l'indemnité de préavis ?", "expected": "data/Articles.txt (Thème : Licenciement)"},
    {"q": "Quel texte régit le capital minimum requis pour le commerce de distribution ?", "expected": "data/Lois.txt (Loi n° 69-1)"},
    {"q": "Jusqu'à quand peut-on commercialiser les huiles d'olive étiquetées avant 2013 ?", "expected": "data/Termesjuridiques.txt (Règlement UE 357-2012)"},
    {"q": "Qui est responsable de la sécurité des salariés dans un contrat de prestation ?", "expected": "data/Articles.txt (Thème : Sous-traitance)"},
    {"q": "Quelle loi définit le système national de normalisation ?", "expected": "data/Lois.txt (Loi n° 2009-38)"},
    {"q": "Quel est le barème d'emploi pour la culture des bananes sous serres ?", "expected": "data/Termesjuridiques.txt (Barèmes agricoles)"},
    {"q": "Quelles sont les conditions de réembauche d'un travailleur démissionnaire ?", "expected": "data/Articles.txt (Article 58 / Convention Collective)"},
    {"q": "Quel texte fixe les modalités d'élaboration et de diffusion des normes ?", "expected": "data/Termesjuridiques.txt (Décret n° 83-724)"},
    {"q": "Où sont précisées les missions de l'INNORPI ?", "expected": "data/Lois.txt (Loi n° 82-66)"},
    {"q": "Quelle est la sanction pour l'usage d'eau potable pour le lavage de voitures ?", "expected": "data/Termesjuridiques.txt (Thème : Gestion de l'eau)"},
    {"q": "Quelles sont les exigences de conformité pour les équipements de sécurité ?", "expected": "data/Articles.txt (Normes de sécurité)"},
    {"q": "Quel est le tarif horaire pour une expertise ou un étalonnage métrologique ?", "expected": "data/Termesjuridiques.txt (Décret n° 2001-588 / Expertise)"},
    {"q": "Quelle autorité surveille les établissements de troisième catégorie ?", "expected": "data/Articles.txt (Article 294 / Surveillance administrative)"},
    {"q": "Quelles sont les obligations de l'exploitant d'un dépôt d'explosifs ?", "expected": "data/Termesjuridiques.txt (Sécurité / Matières explosives)"},
    {"q": "Quelle loi régit le contrôle sanitaire vétérinaire à l'exportation ?", "expected": "data/Lois.txt (Contrôle vétérinaire / Article 3)"}
]

def run_single_test():
    # 1. DELETE THE DB FOR THE NEW MODEL
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"🧹 Cleaned existing database at {DB_DIR}")

    # 2. LOAD DOCUMENTS
    all_docs = []
    for f in FILES:
        if os.path.exists(f):
            print(f"📂 Loading {f}...")
            loader = TextLoader(f, encoding='utf-8')
            all_docs.extend(loader.load())
    
    if not all_docs:
        print("❌ ERROR: No documents loaded.")
        return

    # 3. SPLIT (KEEPING SAME PARAMETERS: 1200 / 200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = [c for c in splitter.split_documents(all_docs) if c.page_content.strip()]
    print(f"✂️ Split into {len(chunks)} valid chunks.")

    # 4. INITIALIZE BGE-M3 MODEL
    print(f"🚀 Loading {MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 5. CREATE VECTOR DB
    ids = [str(i) for i in range(len(chunks))]
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR,
        ids=ids
    )
    print(f"✅ Database created successfully for {MODEL_NAME}!")

    # 6. EVALUATION WITH VISUAL AUDIT
    hits = 0
    total_mrr = 0
    total_precision = 0
    start_time = time.time()

    print(f"\n--- Detailed Audit for {MODEL_NAME} ---")
    for test in TEST_QUERIES:
        results = hybrid_search(test['q'], vector_db)
        
        found_sources = [os.path.basename(doc.metadata.get('source', '')).lower() for doc in results]
        expected_path = test['expected'].split(' (')[0].strip().lower()
        expected_filename = os.path.basename(expected_path)

        is_hit = False
        rank_score = 0
        relevant_count = 0

        for i, s in enumerate(found_sources):
            if expected_filename in s:
                if not is_hit:
                    is_hit = True
                    rank_score = 1 / (i + 1)
                relevant_count += 1
        
        status = "✅ MATCH" if is_hit else "❌ FAIL "
        print(f"{status} | Query: {test['q'][:50]}...")
        if not is_hit:
            print(f"      👉 Expected: {expected_filename}")
            print(f"      👉 AI Found: {found_sources}")

        if is_hit: hits += 1
        total_mrr += rank_score
        total_precision += (relevant_count / len(found_sources)) if found_sources else 0

    # 7. FINAL CALCULATIONS
    total_q = len(TEST_QUERIES)
    recall = (hits / total_q)
    precision = (total_precision / total_q)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mrr = total_mrr / total_q
    latency = (time.time() - start_time) / total_q

    print("\n" + "="*50)
    print(f"📊 ADVANCED METRICS FOR {MODEL_NAME}")
    print(f"✅ Recall@3:    {recall*100:.1f}%")
    print(f"🎯 Precision@3: {precision*100:.1f}%")
    print(f"🧪 F1-Score:    {f1_score*100:.1f}%")
    print(f"🔝 MRR:         {mrr:.3f}")
    print(f"⚡ Latency:     {latency:.3f}s")
    print("="*50)

if __name__ == "__main__":
    run_single_test()