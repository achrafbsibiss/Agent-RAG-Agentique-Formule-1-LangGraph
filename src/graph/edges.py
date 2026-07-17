"""Fonctions de routage conditionnel.

Ces fonctions n'ecrivent jamais dans le state : elles lisent des champs deja
calcules par les noeuds et renvoient le nom du noeud suivant. Toute la logique
de decision de l'agent est donc concentree et testable ici.
"""

from __future__ import annotations

import logging
from typing import Literal

from src.config import settings
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def route_after_analysis(state: AgentState) -> Literal["plan_research", "agent"]:
    """Une question complexe passe par le planificateur, une question simple non.

    Planifier une question factuelle ajouterait un appel LLM et un aller-retour
    d'outil sans rien apporter : le routage protege la latence des cas simples.
    """
    if state.get("complexity") == "complexe":
        return "plan_research"
    return "agent"


def route_after_agent(state: AgentState) -> Literal["tools", "generate"]:
    """Poursuit vers les outils si l'agent en a demande, sinon vers la redaction."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None

    if last is not None and getattr(last, "tool_calls", None):
        return "tools"
    return "generate"


def route_after_grading(state: AgentState) -> Literal["agent", "rewrite_query"]:
    """Boucle d'auto-correction de la recuperation.

    Documents pertinents -> l'agent reprend la main (il peut encore chercher
    d'autres faits de son plan, ou conclure).
    Documents hors sujet -> reformulation, dans la limite du budget. Budget
    epuise, on renvoie quand meme vers l'agent : la redaction dira honnetement
    que l'information est absente, ce qui vaut mieux qu'une boucle infinie.
    """
    if state.get("documents_relevant"):
        return "agent"

    if state.get("rewrites", 0) < settings.graph.max_query_rewrites:
        return "rewrite_query"

    logger.info("Budget de reformulation epuise : redaction avec le contexte disponible")
    return "agent"


def route_after_verification(state: AgentState) -> Literal["generate", "__end__"]:
    """Regenere une reponse jugee non ancree, dans la limite du budget."""
    if state.get("grounded", True):
        return "__end__"

    if state.get("generation_retries", 0) <= settings.graph.max_generation_retries:
        logger.info("Reponse non ancree : nouvelle tentative de redaction")
        return "generate"

    logger.warning("Reponse non ancree apres %d tentative(s) : livraison en l'etat",
                   state.get("generation_retries", 0))
    return "__end__"
