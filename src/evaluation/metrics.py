"""Metriques d'evaluation du systeme.

Trois dimensions sont mesurees, conformement a la consigne :
- la pertinence des documents recuperes (hit@k, MRR) ;
- le temps de reponse (latence bout-en-bout) ;
- la qualite de la reponse (jugee par un LLM juge, sur des criteres explicites).
"""

from __future__ import annotations

from langchain_core.documents import Document as LCDocument


def _match(expected_title: str, retrieved_titles: list[str]) -> bool:
    """Correspondance souple entre un titre attendu et les titres recuperes.

    Une correspondance partielle suffit : le titre resolu par Wikipedia peut
    differer legerement du titre attendu (redirections, sous-titres).
    """
    expected = expected_title.lower()
    return any(expected in title.lower() or title.lower() in expected
               for title in retrieved_titles)


def hit_at_k(expected_docs: list[str], retrieved: list[LCDocument]) -> float:
    """Fraction des documents attendus effectivement recuperes (rappel@k).

    Renvoie 1.0 si aucune attente n'est declaree : la question ne cible alors
    aucun document precis et ne doit pas penaliser le score de recuperation.
    """
    if not expected_docs:
        return 1.0
    titles = [doc.metadata.get("title", "") for doc in retrieved]
    hits = sum(1 for expected in expected_docs if _match(expected, titles))
    return hits / len(expected_docs)


def reciprocal_rank(expected_docs: list[str], retrieved: list[LCDocument]) -> float:
    """Inverse du rang du premier document attendu (MRR par question).

    Recompense le fait de placer une source pertinente en tete de liste, pas
    seulement de la faire figurer quelque part dans le top-k.
    """
    if not expected_docs:
        return 1.0
    for rank, doc in enumerate(retrieved, start=1):
        title = doc.metadata.get("title", "")
        if any(_match(expected, [title]) for expected in expected_docs):
            return 1.0 / rank
    return 0.0
