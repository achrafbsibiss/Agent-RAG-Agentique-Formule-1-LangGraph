"""Outils mis a disposition de l'agent.

Deux familles d'outils coexistent :

1. Les outils de recuperation renvoient un `Command`. En plus du texte lu par
   le LLM, ils ecrivent les chunks bruts dans `state["documents"]`. Les noeuds
   d'evaluation et de generation manipulent ainsi de vrais objets `Document`
   (metadonnees, URL de citation) plutot que de re-parser du texte.

2. Les outils deterministes (arithmetique, inventaire) renvoient une chaine.
   Le calcul de points est deporte en Python car un LLM se trompe
   regulierement sur une somme de dix termes.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from src.config import CATEGORIES, settings
from src.ingestion.retriever import get_retriever

# Nombre maximum de caracteres d'un chunk restitue au LLM dans un resultat
# d'outil. Borne la taille du contexte sans tronquer l'information utile.
SNIPPET_CHARS = 700


def format_documents(documents: list[LCDocument]) -> str:
    """Met en forme les chunks pour lecture par le LLM, avec index de citation."""
    if not documents:
        return "Aucun document trouve."

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        meta = document.metadata
        body = document.page_content[:SNIPPET_CHARS]
        blocks.append(
            f"[SOURCE {index}] titre={meta['title']} | categorie={meta['category']} "
            f"| section={meta['section']}\n{body}"
        )
    return "\n\n".join(blocks)


def _retrieval_command(
    documents: list[LCDocument],
    tool_call_id: str,
    query: str,
) -> Command:
    """Construit la mise a jour de state commune aux outils de recuperation."""
    return Command(
        update={
            "documents": documents,
            "search_queries": [query],
            "messages": [
                ToolMessage(format_documents(documents), tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def rechercher_documents(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Recherche des passages dans toute la base documentaire Formule 1.

    Outil a utiliser par defaut. La recherche est hybride (semantique + mots
    cles) et multilingue : une question en francais retrouve des documents
    rediges en anglais.

    Args:
        query: La question ou les mots cles a rechercher. Formuler une requete
            precise et autonome (ex. "reglement DRS zones d'activation") plutot
            qu'un pronom ou une question elliptique.
    """
    documents = get_retriever().search(query, k=settings.retrieval.top_k)
    return _retrieval_command(documents, tool_call_id, query)


@tool
def rechercher_par_categorie(
    query: str,
    categorie: Literal[
        "reglement", "ecurie", "pilote", "circuit", "glossaire", "saison", "technique", "histoire"
    ],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Recherche des passages dans une seule categorie de la base documentaire.

    A privilegier quand la question vise clairement un type de document : le
    filtre supprime le bruit des autres categories et ameliore la precision.

    Args:
        query: La question ou les mots cles a rechercher.
        categorie: Categorie a interroger. 'reglement' = regles sportives et
            techniques ; 'ecurie' = equipes ; 'pilote' = biographies ;
            'circuit' = tracés ; 'glossaire' = definitions (DRS, pneus, pit stop) ;
            'saison' = championnats 2023-2025 ; 'technique' = moteur, aero, ERS ;
            'histoire' = histoire de la F1 et palmares.
    """
    documents = get_retriever().search(query, k=settings.retrieval.top_k, category=categorie)
    return _retrieval_command(documents, tool_call_id, f"[{categorie}] {query}")


@tool
def comparer_entites(
    entite_a: str,
    entite_b: str,
    aspect: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Recupere en parallele des passages sur deux entites a comparer.

    Une comparaison formulee en une seule requete ("Verstappen contre Hamilton")
    ramene surtout des passages qui evoquent leur rivalite, pas les faits
    propres a chacun. Cet outil emet donc une requete par entite et fusionne
    les resultats, garantissant que les deux cotes sont documentes.

    Args:
        entite_a: Premiere entite (pilote, ecurie, circuit, saison...).
        entite_b: Seconde entite.
        aspect: Angle de comparaison (ex. "titres mondiaux", "puissance moteur",
            "longueur du circuit").
    """
    retriever = get_retriever()
    # Moitie du budget par entite : le contexte final reste comparable a celui
    # d'une recherche simple, mais equilibre entre les deux cotes.
    per_entity = max(2, settings.retrieval.top_k // 2)

    documents: list[LCDocument] = []
    for entity in (entite_a, entite_b):
        documents.extend(retriever.search(f"{entity} {aspect}", k=per_entity))

    return _retrieval_command(documents, tool_call_id, f"{entite_a} vs {entite_b} ({aspect})")


@tool
def inventaire_corpus() -> str:
    """Liste les categories et les titres de documents disponibles dans la base.

    A appeler quand la question semble hors perimetre, pour verifier ce que la
    base contient reellement avant de repondre qu'une information est absente.
    """
    retriever = get_retriever()
    lines = ["Categories disponibles (nombre de passages indexes) :"]
    for category, count in retriever.categories().items():
        lines.append(f"  - {category} : {count} passages")

    lines.append("\nDocuments indexes :")
    for category in CATEGORIES:
        titles = retriever.titles(category)
        if titles:
            lines.append(f"  [{category}] " + " ; ".join(titles))
    return "\n".join(lines)


# Bareme officiel de la Formule 1 : points attribues selon la position.
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


@tool
def calculer_points_championnat(
    positions_course: list[int],
    positions_sprint: list[int] | None = None,
    meilleurs_tours: int = 0,
) -> str:
    """Calcule un total de points de championnat a partir de positions d'arrivee.

    Applique le bareme officiel. Une position hors des points (au-dela de la
    10e en course, de la 8e en sprint) rapporte zero.

    Args:
        positions_course: Positions d'arrivee en Grand Prix, ex. [1, 3, 2].
        positions_sprint: Positions d'arrivee en course sprint, ex. [2, 1].
        meilleurs_tours: Nombre de meilleurs tours en course marques dans le
            top 10 (+1 point chacun ; ce bonus a ete supprime a partir de 2025).
    """
    positions_sprint = positions_sprint or []
    detail: list[str] = []
    total = 0

    for position in positions_course:
        points = RACE_POINTS.get(position, 0)
        total += points
        detail.append(f"  course P{position} -> {points} pts")

    for position in positions_sprint:
        points = SPRINT_POINTS.get(position, 0)
        total += points
        detail.append(f"  sprint P{position} -> {points} pts")

    if meilleurs_tours:
        total += meilleurs_tours
        detail.append(f"  meilleurs tours x{meilleurs_tours} -> {meilleurs_tours} pts")

    return "Detail du calcul :\n" + "\n".join(detail) + f"\n  TOTAL = {total} points"


# Outils exposes au modele de raisonnement, dans l'ordre de priorite d'usage.
F1_TOOLS = [
    rechercher_documents,
    rechercher_par_categorie,
    comparer_entites,
    inventaire_corpus,
    calculer_points_championnat,
]
