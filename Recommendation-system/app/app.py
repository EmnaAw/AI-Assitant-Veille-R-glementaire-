import os

import streamlit as st

from src.pipeline_runtime import LegalRecommendationPipeline

st.set_page_config(page_title="Legal Recommendation Engine", layout="wide")
st.title("French Legal Recommendation Engine")
st.write("Semantic retrieval + deterministic truth lookup + local French generation")

DEBUG_MODE = os.getenv("APP_DEBUG", "0").lower() in {"1", "true", "yes"}


@st.cache_resource
def get_pipeline() -> LegalRecommendationPipeline:
    return LegalRecommendationPipeline()


query = st.text_area("Requête utilisateur (français)", height=120)

if st.button("Run recommendation"):
    try:
        if not query.strip():
            st.warning("Veuillez saisir une non-conformité avant de lancer la recherche.")
            st.stop()

        result = get_pipeline().run(query=query)

        if result.mode == "verified":
            st.subheader("Action recommandée")
            st.write(result.display_plan_fr or result.official_plan)
            st.subheader("Explication")
            st.write(result.explanation_fr)
        elif result.mode == "ambiguous":
            st.subheader("Plusieurs correspondances possibles")
            st.write(result.explanation_fr)
        elif result.mode == "no_match":
            st.subheader("Aucune correspondance trouvée")
            st.write(result.explanation_fr)
        else:
            st.subheader("Suggestion")
            st.write(result.explanation_fr)
    except Exception as exc:
        st.error(
            "Une erreur interne a empêché le traitement de la demande. "
            "Vérifiez la santé du pipeline et réessayez."
        )
        if DEBUG_MODE:
            st.exception(exc)
