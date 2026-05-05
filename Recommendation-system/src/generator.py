import re
import unicodedata

import requests

from .config import (
    GENERATION_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT,
)
from .prompt_builder import (
    build_french_action_prompt,
    build_french_advisory_prompt,
    build_french_ambiguity_message,
    build_french_explanation_prompt,
)
from .text_utils import clean_client_text, sentence_case


GENERIC_EXPLANATION_FRAGMENTS = (
    "cette action precise la mesure concrete",
    "cette action precise la mesure concrete a mettre en oeuvre",
    "corriger la non conformite et assurer un fonctionnement conforme",
    "fonctionnement conforme et tracable",
    "mesure concrete a mettre en oeuvre",
    "l'action corrective officielle est la suivante",
)

MEANINGLESS_EXPLANATION_WORDS = {
    "action",
    "ainsi",
    "assurer",
    "ce",
    "cela",
    "cette",
    "concrete",
    "conforme",
    "consiste",
    "corriger",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "entreprise",
    "et",
    "etre",
    "faut",
    "faire",
    "il",
    "la",
    "le",
    "les",
    "leur",
    "mesure",
    "mettre",
    "mise",
    "non",
    "oeuvre",
    "operationnelle",
    "par",
    "pour",
    "precise",
    "qu",
    "que",
    "sa",
    "simplement",
    "son",
    "suivi",
    "tracable",
    "une",
}


class Generator:
    def __init__(self, backend: str = GENERATION_BACKEND):
        self.backend = backend
        self.session = requests.Session()
        self._action_cache: dict[str, str] = {}
        self._explanation_cache: dict[tuple[str, str, str], str] = {}
        self._advisory_cache: dict[tuple[str, str, tuple[str, ...]], str] = {}

    def _clean_response(self, response: str) -> str:
        cleaned = clean_client_text(response)
        for marker in (
            "Resume court :",
            "Point de vigilance :",
            "Action recommandee :",
            "Explication :",
            "Suggestion :",
            "RÃ©sumÃ© court :",
            "Action recommandÃ©e :",
        ):
            cleaned = cleaned.replace(marker, "").strip()
        return clean_client_text(cleaned)

    def _normalize_for_analysis(self, text: str) -> str:
        value = clean_client_text(text).lower()
        return "".join(
            char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char)
        )

    def _looks_incomplete(self, text: str) -> bool:
        normalized = clean_client_text(text).strip()
        if not normalized:
            return True
        if normalized[-1] not in ".!?":
            return True
        trailing_tokens = self._normalize_for_analysis(normalized).split()[-4:]
        return any(token in {"de", "du", "des", "et", "en", "pour", "avec"} for token in trailing_tokens)

    def _is_low_quality_action(self, action: str, plan: str) -> bool:
        cleaned = clean_client_text(action)
        lowered = self._normalize_for_analysis(cleaned)
        if not cleaned:
            return True
        if len(cleaned) > 260:
            return True
        if "voici" in lowered or ":" in cleaned:
            return True
        plan_tokens = self._meaningful_tokens(plan)
        action_tokens = self._meaningful_tokens(action)
        if plan_tokens and not (plan_tokens & action_tokens):
            return True
        return self._looks_incomplete(cleaned)

    def _meaningful_tokens(self, text: str) -> set[str]:
        tokens = re.findall(r"\w+", self._normalize_for_analysis(text), flags=re.UNICODE)
        return {token for token in tokens if len(token) > 2 and token not in MEANINGLESS_EXPLANATION_WORDS}

    def _strip_legal_suffix(self, text: str) -> str:
        value = clean_client_text(text)
        patterns = (
            r"\s+conformement\b.*$",
            r"\s+selon\b.*$",
            r"\s+au titre de\b.*$",
            r"\s+prevu(?:e)?\b.*$",
            r"\s+tel que prevu\b.*$",
            r"\s+article\b.*$",
        )
        for pattern in patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip(" ,.;:")
        return sentence_case(value)

    def _contextual_explanation_from_plan(self, nc: str, plan: str, gap_type: str) -> str:
        cleaned_plan = self._strip_legal_suffix(plan)
        lowered_plan = self._normalize_for_analysis(cleaned_plan)
        lowered_nc = self._normalize_for_analysis(nc)

        if gap_type == "personnes_handicapees" or "personne handicapee" in lowered_plan or "personnes handicapees" in lowered_plan:
            return (
                "Cette action vise à corriger l'absence de personnes handicapées en assurant leur recrutement "
                "conformément à l'exigence applicable à l'entreprise."
            )

        if gap_type == "equipe_role_securite" or "equipe de securite" in lowered_plan:
            return (
                "Cette action vise à corriger l'absence ou l'insuffisance d'organisation de l'équipe de sécurité "
                "en la constituant, en définissant les personnes qui la composent et en organisant son fonctionnement."
            )

        if gap_type == "registre_securite" or ("registre" in lowered_plan and "securite" in lowered_plan):
            return (
                "Cette action vise à corriger l'absence ou le défaut de suivi du registre de sécurité "
                "en le créant, en y consignant les informations utiles et en le tenant à jour."
            )

        if gap_type == "formation" and ("gestes" in lowered_plan or "postures" in lowered_plan):
            return (
                "Cette action vise à corriger le manque de formation pratique en formant le personnel "
                "aux bons gestes et aux bonnes postures afin de réduire les risques."
            )

        if gap_type == "formation" or "formation" in lowered_plan:
            return (
                "Cette action vise à corriger le manque de formation en organisant la session requise "
                "et en assurant son suivi pour que les personnes concernées appliquent correctement l'exigence attendue."
            )

        if gap_type == "autorisation_administratif" and ("bureau detude" in lowered_plan or "dossier" in lowered_plan):
            return (
                "Cette action vise à corriger l'écart administratif constaté en faisant avancer le dossier "
                "avec l'intervenant compétent jusqu'à sa régularisation effective."
            )

        if gap_type == "autorisation_administratif" or "autorisation" in lowered_plan:
            return (
                "Cette action vise à corriger l'écart administratif constaté en obtenant ou en régularisant "
                "l'autorisation nécessaire avant de poursuivre l'activité concernée."
            )

        if gap_type == "equipement_securite" or "bloc autonome" in lowered_plan or "eclairage" in lowered_plan:
            return (
                "Cette action vise à corriger la défaillance de l'équipement de sécurité en remettant en état "
                "le dispositif concerné et en vérifiant son bon fonctionnement."
            )

        if gap_type == "dechets_sanitaires" or "conteneur" in lowered_plan or "dechet" in lowered_plan:
            return (
                "Cette action vise à corriger l'écart de gestion des déchets en mettant en place un moyen adapté "
                "pour les identifier, les séparer et les traiter correctement."
            )

        if gap_type == "analyse_controle":
            return (
                "Cette action vise à corriger l'absence ou l'insuffisance de contrôle en réalisant l'analyse, "
                "la mesure ou la vérification attendue et en assurant sa traçabilité."
            )

        if lowered_plan.startswith("mettre en place "):
            objective = cleaned_plan[17:].strip(" .")
            if objective:
                return (
                    f"Cette action vise à corriger la non-conformité constatée en mettant réellement en place "
                    f"{objective.lower()} et en le rendant opérationnel."
                )

        if lowered_plan.startswith("assurer "):
            objective = cleaned_plan[8:].strip(" .")
            if objective:
                return (
                    f"Cette action vise à garantir {objective.lower()} de manière effective "
                    "et suivie dans l'activité concernée."
                )

        if lowered_plan.startswith("tenir a jour "):
            objective = cleaned_plan[12:].strip(" .")
            if objective:
                return (
                    f"Cette action vise à maintenir {objective.lower()} à jour "
                    "et à en assurer le suivi régulier."
                )

        if lowered_plan.startswith("designer "):
            objective = cleaned_plan[9:].strip(" .")
            if objective:
                return (
                    f"Cette action vise à désigner clairement {objective.lower()} "
                    "afin que la responsabilité soit formalisée."
                )

        if "absence" in lowered_nc:
            return (
                f"Cette action vise à corriger l'absence constatée en mettant effectivement en place "
                f"{cleaned_plan.lower()}."
            )

        return f"Cette action vise à appliquer concrètement la mesure suivante : {cleaned_plan.lower()}."

    def _is_generic_explanation(self, explanation: str, nc: str, plan: str, gap_type: str) -> bool:
        lowered = self._normalize_for_analysis(explanation)
        if not lowered:
            return True
        if any(fragment in lowered for fragment in GENERIC_EXPLANATION_FRAGMENTS):
            return True
        if len(lowered) > 360:
            return True

        plan_tokens = self._meaningful_tokens(self._strip_legal_suffix(plan))
        nc_tokens = self._meaningful_tokens(nc)
        gap_tokens = self._meaningful_tokens(gap_type.replace("_", " "))
        explanation_tokens = self._meaningful_tokens(explanation)
        reference_tokens = plan_tokens | nc_tokens | gap_tokens

        if len(explanation_tokens) < 4:
            return True
        if self._looks_incomplete(explanation):
            return True
        if reference_tokens and not (reference_tokens & explanation_tokens):
            return True

        overlap_score = (
            len(plan_tokens & explanation_tokens)
            + len(nc_tokens & explanation_tokens)
            + len(gap_tokens & explanation_tokens)
        )
        return overlap_score < 2

    def _default_action(self, plan: str) -> str:
        plan = self._normalize_for_analysis(plan)

        if "personne handicapee" in plan or "personnes handicapees" in plan:
            return (
                "Assurer le recrutement d'au moins une personne en situation de handicap "
                "au sein de l'entreprise."
            )
        if "equipe de securite" in plan or "chef dequipe de securite" in plan:
            return (
                "Constituer une équipe de sécurité, désigner ses membres "
                "et assurer les formations obligatoires."
            )
        if "registre" in plan and "securite" in plan:
            return "Mettre en place et tenir à jour un registre de sécurité."
        if "formation" in plan and ("gestes" in plan or "postures" in plan):
            return "Planifier une formation sur les gestes et postures adaptées."
        if "formation" in plan:
            return "Planifier et assurer le suivi de la formation requise."
        if "bureau detude" in plan or "dossier" in plan:
            return "Faire avancer et finaliser le dossier en cours avec l'intervenant compétent."
        if "autorisation" in plan:
            return "Régulariser l'autorisation requise avant la poursuite de l'activité."
        if "bloc autonome" in plan or "eclairage" in plan:
            return "Assurer la remise en état et l'entretien de l'éclairage de sécurité."
        if "conteneur" in plan or "dechet" in plan:
            return "Mettre en place un conteneur adapté pour la gestion des déchets concernés."

        return sentence_case(plan)

    def _default_explanation(self, nc: str, plan: str, gap_type: str) -> str:
        return self._contextual_explanation_from_plan(nc=nc, plan=plan, gap_type=gap_type)

    def _default_advisory(self) -> str:
        return (
            "Cette non-conformite n'existe pas actuellement dans notre base.\n\n"
            "Suggestion : vérifiez les documents de preuve, la traçabilité et les procédures "
            "associées afin d'identifier le cas réglementaire le plus proche.\n\n"
            "Cette réponse constitue une suggestion non confirmée."
        )

    def _fallback_advisory(self) -> str:
        return (
            "Aucune correspondance officielle fiable n'a été trouvée.\n\n"
            "Suggestion : vérifiez la traçabilité documentaire, les preuves disponibles et les "
            "procédures concernées pour préciser la non-conformité.\n\n"
            "Cette réponse constitue une suggestion non confirmée."
        )

    def _candidate_aware_advisory(self, top_candidates: list[dict]) -> str:
        if not top_candidates:
            return self._fallback_advisory()

        best = top_candidates[0]
        best_nc = clean_client_text(best.get("nc", ""))
        best_plan = clean_client_text(best.get("official_plan", ""))

        if best_nc and best_plan:
            return (
                "Cette formulation n'a pas pu être rattachée avec certitude à un cas officiel unique.\n\n"
                f"Le cas le plus proche semble être : {best_nc}.\n"
                f"Action potentiellement pertinente : {best_plan}.\n\n"
                "Cette réponse reste une suggestion non confirmée et doit être validée selon le contexte réel."
            )

        return self._fallback_advisory()

    def generate_french_action(self, plan: str) -> str:
        cache_key = str(plan).strip()
        cached = self._action_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.backend == "template":
            result = self._default_action(plan)
            self._action_cache[cache_key] = result
            return result

        if self.backend == "ollama":
            prompt = build_french_action_prompt(plan=plan)
            try:
                result = self._clean_response(self._generate_with_ollama(prompt))
            except Exception:
                result = self._default_action(plan)
            if self._is_low_quality_action(result, plan=plan):
                result = self._default_action(plan)
            self._action_cache[cache_key] = clean_client_text(result) or self._default_action(plan)
            return self._action_cache[cache_key]

        raise ValueError(f"Unsupported generation backend: {self.backend}")

    def generate_french_explanation(self, nc: str, plan: str, gap_type: str) -> str:
        normalized_gap_type = clean_client_text(gap_type or "autre").replace(" ", "_").lower()
        cache_key = (str(nc).strip(), str(plan).strip(), normalized_gap_type)
        cached = self._explanation_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.backend == "template":
            result = self._default_explanation(nc=nc, plan=plan, gap_type=normalized_gap_type)
            self._explanation_cache[cache_key] = result
            return result

        if self.backend == "ollama":
            prompt = build_french_explanation_prompt(
                nc=nc,
                plan=plan,
                gap_type=normalized_gap_type,
            )
            try:
                result = self._clean_response(self._generate_with_ollama(prompt))
            except Exception:
                result = self._default_explanation(nc=nc, plan=plan, gap_type=normalized_gap_type)

            if self._is_generic_explanation(result, nc=nc, plan=plan, gap_type=normalized_gap_type):
                result = self._default_explanation(nc=nc, plan=plan, gap_type=normalized_gap_type)

            self._explanation_cache[cache_key] = sentence_case(result)
            return self._explanation_cache[cache_key]

        raise ValueError(f"Unsupported generation backend: {self.backend}")

    def generate_french_ambiguity(self, query: str, conflicting_matches: list[dict]) -> str:
        return build_french_ambiguity_message(
            query=query,
            conflicting_matches=conflicting_matches,
        )

    def generate_french_advisory(
        self,
        query: str,
        normalized_query: str,
        top_candidates: list[dict],
    ) -> str:
        cache_key = (
            str(query).strip(),
            str(normalized_query).strip(),
            tuple(str(candidate.get("nc", "")).strip() for candidate in top_candidates[:3]),
        )
        cached = self._advisory_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.backend == "template":
            result = self._candidate_aware_advisory(top_candidates)
            self._advisory_cache[cache_key] = result
            return result

        if self.backend == "ollama":
            prompt = build_french_advisory_prompt(
                query=query,
                normalized_query=normalized_query,
                top_candidates=top_candidates,
            )
            try:
                result = clean_client_text(self._generate_with_ollama(prompt))
            except Exception:
                result = self._candidate_aware_advisory(top_candidates)
            self._advisory_cache[cache_key] = result
            return result

        raise ValueError(f"Unsupported generation backend: {self.backend}")

    def _generate_with_ollama(self, prompt: str) -> str:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0.2,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        }

        try:
            response = self.session.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to connect to Ollama at {url}: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama generation failed.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}"
            )

        data = response.json()
        text = data.get("response", "").strip()
        if not text:
            raise RuntimeError(f"Ollama returned empty response: {data}")

        return text
