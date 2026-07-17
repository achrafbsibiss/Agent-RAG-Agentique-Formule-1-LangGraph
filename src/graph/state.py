"""State partage du graphe agentique.

Le state est le seul canal de communication entre les noeuds. Chaque champ
porte un reducteur explicite qui decrit comment deux mises a jour concurrentes
fusionnent, ce qui rend le comportement du graphe deterministe et lisible.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document as LCDocument
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState


def merge_documents(
    left: list[LCDocument] | None,
    right: list[LCDocument] | None,
) -> list[LCDocument]:
    """Concatene les documents recuperes en dedupliquant par `chunk_id`.

    Plusieurs appels d'outils dans un meme tour ramenent souvent les memes
    chunks. Sans deduplication, le contexte de generation se remplit de
    doublons au detriment de sources reellement nouvelles.

    Une mise a jour a `[]` reinitialise volontairement l'accumulateur : c'est
    ainsi que le noeud de reformulation repart d'un contexte propre.
    """
    if right == []:
        return []
    left = left or []
    right = right or []

    seen = {doc.metadata.get("chunk_id") for doc in left}
    merged = list(left)
    for document in right:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id not in seen:
            seen.add(chunk_id)
            merged.append(document)
    return merged


QueryComplexity = Literal["simple", "complexe"]


class AgentState(TypedDict, total=False):
    """State circulant dans le graphe.

    `messages` porte l'historique conversationnel (et donc la memoire
    multi-tours, restauree par le checkpointer a partir du `thread_id`).
    Les autres champs sont l'espace de travail interne d'un seul tour.
    """

    # --- Conversation ---
    messages: Annotated[list, add_messages]

    # --- Analyse de la question ---
    question: str
    language: str            # 'fr' ou 'en' : impose la langue de la reponse
    complexity: QueryComplexity
    plan: list[str]          # sous-questions produites par le planificateur

    # --- Recuperation ---
    documents: Annotated[list[LCDocument], merge_documents]
    search_queries: Annotated[list[str], operator.add]  # trace des requetes emises
    documents_relevant: bool

    # --- Compteurs de garde-fou (evitent les boucles infinies) ---
    tool_iterations: int
    rewrites: int
    generation_retries: int

    # --- Sortie ---
    answer: str
    grounded: bool
    notes: Annotated[list[str], operator.add]  # journal de decision, pour la trace
