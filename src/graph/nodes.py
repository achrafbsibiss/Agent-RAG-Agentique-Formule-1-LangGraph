"""Noeuds du graphe agentique.

Chaque noeud est une fonction pure `AgentState -> dict` : elle lit le state et
renvoie uniquement les champs qu'elle modifie. Les reducteurs declares dans
`state.py` se chargent de la fusion.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import prompts
from src.graph.state import AgentState
from src.llm.models import get_llm
from src.tools.f1_tools import F1_TOOLS, format_documents

logger = logging.getLogger(__name__)

# Nombre de tours de conversation restitues au noeud d'analyse pour resoudre
# les references implicites. Au-dela, le cout croit sans gain de resolution.
HISTORY_TURNS = 6


# --------------------------------------------------------------------------
# Schemas de sortie structuree : contraignent le LLM a une decision exploitable
# --------------------------------------------------------------------------
class QueryAnalysis(BaseModel):
    language: str = Field(description="Langue de la question : 'fr' ou 'en'")
    standalone_question: str = Field(description="Question reecrite, comprehensible sans historique")
    complexity: str = Field(description="'simple' ou 'complexe'")


class ResearchPlan(BaseModel):
    sub_questions: list[str] = Field(description="Sous-questions autonomes a rechercher")


class RelevanceGrade(BaseModel):
    relevant: bool = Field(description="true si les passages permettent de repondre")
    reason: str = Field(description="Justification en une phrase")


class GroundingGrade(BaseModel):
    grounded: bool = Field(description="true si la reponse est soutenue par les passages")
    reason: str = Field(description="Justification en une phrase")


# Mots vides tres frequents en francais : suffisent a deviner la langue quand
# l'analyse LLM est indisponible (repli hors ligne, sans appel reseau).
_FRENCH_MARKERS = {
    "le", "la", "les", "des", "une", "un", "est", "quel", "quelle", "combien",
    "comment", "pourquoi", "qui", "que", "quoi", "ou", "avec", "pour", "dans",
    "gagne", "gagné", "titres", "ecurie", "écurie", "pilote", "course",
}


def _looks_french(text: str) -> bool:
    words = {word.strip("?.,!").lower() for word in text.split()}
    return len(words & _FRENCH_MARKERS) >= 1


def _history(messages: list[BaseMessage]) -> str:
    """Rend l'historique conversationnel sous forme lisible par le LLM."""
    turns = [
        message
        for message in messages
        if message.type in ("human", "ai") and not getattr(message, "tool_calls", None)
    ]
    if not turns:
        return "(aucun echange precedent)"
    rendered = [
        f"{'Utilisateur' if turn.type == 'human' else 'Assistant'} : {turn.content}"
        for turn in turns[-HISTORY_TURNS:]
    ]
    return "\n".join(rendered)


# --------------------------------------------------------------------------
# 1. Analyse : langue, resolution des references, routage simple/complexe
# --------------------------------------------------------------------------
def analyze_query(state: AgentState) -> dict:
    """Determine la langue, rend la question autonome et evalue sa complexite.

    C'est ici que la memoire conversationnelle devient exploitable : une
    question elliptique ("et son coequipier ?") est reecrite en question
    autonome grace a l'historique, sinon la recherche vectorielle n'aurait
    aucun terme discriminant a se mettre sous la dent.
    """
    messages = state["messages"]
    question = messages[-1].content

    grader = get_llm("grader").with_structured_output(QueryAnalysis)
    try:
        analysis: QueryAnalysis = grader.invoke(
            prompts.ANALYZE_PROMPT.format(
                history=_history(messages[:-1]),
                question=question,
            )
        )
        complexity = "complexe" if analysis.complexity.lower().startswith("complex") else "simple"
        language = "fr" if analysis.language.lower().startswith("fr") else "en"
        standalone = analysis.standalone_question
    except Exception as exc:
        # En cas d'echec (quota, sortie non conforme), on degrade proprement :
        # question inchangee, langue devinee, traitement en mode simple. Le
        # graphe poursuit au lieu de s'interrompre.
        logger.warning("Analyse indisponible (%s) : repli sur des valeurs par defaut", type(exc).__name__)
        standalone = question
        complexity = "simple"
        language = "fr" if _looks_french(question) else "en"

    logger.info("Analyse : langue=%s complexite=%s", language, complexity)
    return {
        "question": standalone,
        "language": language,
        "complexity": complexity,
        # Un nouveau tour repart d'un contexte documentaire vierge.
        "documents": [],
        "tool_iterations": 0,
        "rewrites": 0,
        "generation_retries": 0,
        "notes": [f"analyse: {language}/{complexity} | question autonome: {standalone}"],
    }


# --------------------------------------------------------------------------
# 2. Planification : decomposition des questions complexes
# --------------------------------------------------------------------------
def plan_research(state: AgentState) -> dict:
    """Decompose une question complexe en sous-questions atomiques.

    Sans plan explicite, le modele traite une question multi-hop en une seule
    recherche et ne recupere que la moitie des faits necessaires.
    """
    planner = get_llm("reasoner").with_structured_output(ResearchPlan)
    try:
        plan: ResearchPlan = planner.invoke(
            prompts.PLAN_PROMPT.format(
                question=state["question"],
                max_steps=settings.graph.max_subquestions,
            )
        )
        steps = plan.sub_questions[: settings.graph.max_subquestions]
    except Exception as exc:
        # Sans plan, l'agent traite la question d'un bloc : moins optimal mais
        # fonctionnel. Preferable a un arret du graphe.
        logger.warning("Planification indisponible (%s) : poursuite sans plan", type(exc).__name__)
        steps = []
    logger.info("Plan : %d sous-question(s)", len(steps))
    return {"plan": steps, "notes": [f"plan: {len(steps)} sous-questions"]}


# --------------------------------------------------------------------------
# 3. Agent : raisonnement et selection des outils
# --------------------------------------------------------------------------
def agent(state: AgentState) -> dict:
    """Decide de l'outil a appeler, ou signale que la recherche est terminee.

    Le budget d'iterations est verifie avant l'appel au modele : quand il est
    epuise, on n'appelle pas le LLM avec les outils. Cela evite de produire un
    `tool_call` que le graphe n'executera pas, ce qui laisserait l'historique
    dans un etat invalide pour l'appel suivant a l'API.
    """
    iterations = state.get("tool_iterations", 0)

    if iterations >= settings.graph.max_tool_iterations:
        logger.info("Budget d'outils epuise (%d tours) : passage a la redaction", iterations)
        return {"notes": [f"budget outils epuise apres {iterations} tours"]}

    plan_block = ""
    if state.get("plan"):
        steps = "\n".join(f"  {index}. {step}" for index, step in enumerate(state["plan"], 1))
        plan_block = prompts.PLAN_BLOCK.format(steps=steps)

    system = SystemMessage(
        prompts.AGENT_PROMPT.format(
            plan_block=plan_block,
            language=state.get("language", "fr"),
        )
    )
    model = get_llm("reasoner").bind_tools(F1_TOOLS)
    try:
        response = model.invoke([system] + state["messages"])
    except Exception as exc:
        # Certains modeles (ex. llama-3.3 via Groq) emettent parfois un appel
        # d'outil mal forme, que le fournisseur rejette par une erreur 400.
        # Plutot que de laisser planter le graphe, on abandonne l'usage des
        # outils pour ce tour : le routage bascule vers la redaction, qui
        # repondra avec les documents deja recuperes (ou signalera leur
        # absence). L'application reste ainsi robuste face a une question hors
        # perimetre ou a un caprice du modele.
        logger.warning("Echec de l'appel d'outil (%s) : passage a la redaction", type(exc).__name__)
        return {
            "tool_iterations": settings.graph.max_tool_iterations,
            "notes": [f"appel d'outil rejete par le fournisseur : {type(exc).__name__}"],
        }

    return {"messages": [response], "tool_iterations": iterations + 1}


# --------------------------------------------------------------------------
# 4. Evaluation de la pertinence des documents (auto-correction CRAG)
# --------------------------------------------------------------------------
def grade_documents(state: AgentState) -> dict:
    """Juge si le contexte recupere permet de repondre.

    L'evaluation porte sur le lot entier plutot que sur chaque passage : un
    seul appel LLM au lieu de k, ce qui compte sur une API gratuite a quota.
    """
    documents = state.get("documents", [])
    if not documents:
        return {"documents_relevant": False, "notes": ["evaluation: aucun document recupere"]}

    grader = get_llm("grader").with_structured_output(RelevanceGrade)
    try:
        grade: RelevanceGrade = grader.invoke(
            prompts.GRADE_PROMPT.format(
                question=state["question"],
                documents=format_documents(documents),
            )
        )
        relevant, reason = grade.relevant, grade.reason
    except Exception as exc:
        # Repli optimiste : on considere les documents exploitables et on laisse
        # la redaction trancher, plutot que de boucler en reformulation.
        logger.warning("Evaluation indisponible (%s) : documents supposes pertinents", type(exc).__name__)
        relevant, reason = True, "evaluation indisponible (repli)"
    logger.info("Pertinence : %s (%s)", relevant, reason)
    return {
        "documents_relevant": relevant,
        "notes": [f"pertinence={relevant}: {reason}"],
    }


# --------------------------------------------------------------------------
# 5. Reformulation de requete
# --------------------------------------------------------------------------
def rewrite_query(state: AgentState) -> dict:
    """Reecrit la requete apres un echec de recuperation.

    Les documents deja recuperes sont conserves : sur une question multi-hop,
    un lot juge insuffisant peut malgre tout couvrir une des sous-questions.
    """
    tried = state.get("search_queries", [])
    rewriter = get_llm("reasoner")
    new_query = rewriter.invoke(
        prompts.REWRITE_PROMPT.format(
            question=state["question"],
            tried=" | ".join(tried) if tried else "(aucune)",
        )
    ).content.strip()

    logger.info("Reformulation -> %s", new_query)
    # Le message est adresse a l'agent : il oriente le prochain appel d'outil
    # sans reecrire la question originale de l'utilisateur.
    nudge = SystemMessage(
        f"Les recherches precedentes ont echoue. Relance une recherche avec "
        f"cette requete reformulee : \"{new_query}\""
    )
    return {
        "messages": [nudge],
        "rewrites": state.get("rewrites", 0) + 1,
        "notes": [f"reformulation #{state.get('rewrites', 0) + 1}: {new_query}"],
    }


# --------------------------------------------------------------------------
# 6. Generation de la reponse finale
# --------------------------------------------------------------------------
def generate(state: AgentState) -> dict:
    """Redige la reponse ancree dans les passages recuperes."""
    documents = state.get("documents", [])
    language = state.get("language", "fr")

    if not documents:
        message = (
            "Je n'ai trouve aucun document pertinent dans la base pour repondre a cette "
            "question. Elle sort probablement du perimetre du corpus (Formule 1 : "
            "reglement, ecuries, pilotes, circuits, technique, strategie, saisons 2005-2025)."
            if language == "fr"
            else "I could not find any relevant document in the knowledge base for this "
            "question. It likely falls outside the corpus scope (Formula 1: regulations, "
            "teams, drivers, circuits, technology, strategy, 2005-2025 seasons)."
        )
        return {"answer": message, "messages": [AIMessage(message)], "grounded": True}

    writer = get_llm("reasoner")
    try:
        answer = writer.invoke(
            prompts.GENERATE_PROMPT.format(
                question=state["question"],
                documents=format_documents(documents),
                language="francais" if language == "fr" else "anglais",
            )
        ).content
    except Exception as exc:
        logger.warning("Redaction indisponible (%s)", type(exc).__name__)
        answer = (
            "Le service de generation est momentanement indisponible (quota ou reseau). "
            "Reessayez dans quelques instants."
            if language == "fr"
            else "The generation service is temporarily unavailable (quota or network). "
            "Please try again shortly."
        )
        return {"answer": answer, "messages": [AIMessage(answer)], "grounded": True}

    return {"answer": answer, "messages": [AIMessage(answer)], "notes": ["reponse redigee"]}


# --------------------------------------------------------------------------
# 7. Verification de l'ancrage (anti-hallucination)
# --------------------------------------------------------------------------
def verify_grounding(state: AgentState) -> dict:
    """Verifie que la reponse ne contient pas d'affirmation absente des sources."""
    documents = state.get("documents", [])
    if not documents:
        return {"grounded": True}

    grader = get_llm("grader").with_structured_output(GroundingGrade)
    try:
        grade: GroundingGrade = grader.invoke(
            prompts.GROUNDING_PROMPT.format(
                documents=format_documents(documents),
                answer=state["answer"],
            )
        )
        grounded, reason = grade.grounded, grade.reason
    except Exception as exc:
        # Repli : on livre la reponse telle quelle plutot que de boucler ou
        # d'echouer. Le pire cas est une reponse non verifiee, pas un plantage.
        logger.warning("Verification indisponible (%s) : livraison en l'etat", type(exc).__name__)
        grounded, reason = True, "verification indisponible (repli)"
    logger.info("Ancrage : %s (%s)", grounded, reason)

    update: dict = {
        "grounded": grounded,
        "notes": [f"ancrage={grounded}: {reason}"],
    }
    if not grounded:
        update["generation_retries"] = state.get("generation_retries", 0) + 1
    return update
