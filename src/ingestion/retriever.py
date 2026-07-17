"""Recherche hybride : dense (semantique) + BM25 (lexical).

La recherche dense seule echoue sur les requetes tres litterales, typiques du
domaine F1 : noms propres, sigles (DRS, ERS), millesimes ("2024"). La recherche
BM25 seule echoue des que la question est reformulee ou traduite. Les deux
listes de resultats sont donc fusionnees par Reciprocal Rank Fusion, qui
combine des rangs plutot que des scores et evite d'avoir a normaliser des
echelles heterogenes.
"""

from __future__ import annotations

import functools
import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument

from src.config import settings
from src.corpus.build_corpus import load_corpus
from src.ingestion.chunker import chunk_documents
from src.ingestion.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

# Constante d'amortissement du RRF. 60 est la valeur de la publication
# d'origine (Cormack et al., 2009) et reste un defaut robuste.
RRF_K = 60


class HybridRetriever:
    """Recherche hybride avec filtre optionnel par categorie."""

    def __init__(self) -> None:
        self._store = load_vectorstore()
        # BM25 travaille en memoire : il lui faut le texte des chunks, que
        # Chroma ne rend pas commodement. On rejoue donc le meme decoupage
        # deterministe que celui de l'indexation.
        self._chunks = chunk_documents(load_corpus())
        logger.info("Retriever pret : %d chunks en memoire pour BM25", len(self._chunks))

    # -- Recherche lexicale -------------------------------------------------
    @functools.lru_cache(maxsize=16)
    def _bm25(self, category: str | None) -> BM25Retriever | None:
        pool = [
            chunk
            for chunk in self._chunks
            if category is None or chunk.metadata["category"] == category
        ]
        if not pool:
            return None
        retriever = BM25Retriever.from_documents(pool)
        retriever.k = settings.retrieval.fetch_k
        return retriever

    # -- Recherche dense ----------------------------------------------------
    def _dense(self, query: str, category: str | None, k: int) -> list[LCDocument]:
        # MMR plutot que similarite pure : sur un corpus encyclopedique, les
        # k meilleurs chunks proviennent souvent du meme paragraphe. MMR
        # diversifie les sources, ce qui compte pour les questions comparatives.
        return self._store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=settings.retrieval.fetch_k,
            lambda_mult=0.5,
            filter={"category": category} if category else None,
        )

    # -- Fusion -------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int | None = None,
        category: str | None = None,
    ) -> list[LCDocument]:
        """Renvoie les k chunks les plus pertinents, fusion dense + BM25."""
        k = k or settings.retrieval.top_k
        dense_hits = self._dense(query, category, k=settings.retrieval.fetch_k // 2)

        bm25 = self._bm25(category)
        lexical_hits = bm25.invoke(query)[: settings.retrieval.fetch_k // 2] if bm25 else []

        scores: dict[str, float] = {}
        by_id: dict[str, LCDocument] = {}

        for weight, hits in (
            (settings.retrieval.dense_weight, dense_hits),
            (settings.retrieval.bm25_weight, lexical_hits),
        ):
            for rank, document in enumerate(hits):
                chunk_id = document.metadata["chunk_id"]
                by_id[chunk_id] = document
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank + 1)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return self._diversify(by_id, ranked, k)

    def _diversify(
        self,
        by_id: dict[str, LCDocument],
        ranked: list[tuple[str, float]],
        k: int,
    ) -> list[LCDocument]:
        """Selectionne les k meilleurs chunks en plafonnant leur source.

        Parcours glouton par score decroissant : un chunk est retenu tant que
        son document source n'a pas atteint `max_per_document`. Si le plafond
        empeche de reunir k chunks, on complete avec les meilleurs restants
        plutot que de renvoyer une liste incomplete.
        """
        cap = settings.retrieval.max_per_document
        selected: list[LCDocument] = []
        overflow: list[LCDocument] = []
        per_document: dict[str, int] = {}

        for chunk_id, _score in ranked:
            document = by_id[chunk_id]
            source = document.metadata["doc_id"]
            if per_document.get(source, 0) < cap:
                per_document[source] = per_document.get(source, 0) + 1
                selected.append(document)
                if len(selected) == k:
                    return selected
            else:
                overflow.append(document)

        return (selected + overflow)[:k]

    def categories(self) -> dict[str, int]:
        """Compte les chunks par categorie : sert a l'outil d'inventaire."""
        counts: dict[str, int] = {}
        for chunk in self._chunks:
            category = chunk.metadata["category"]
            counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items()))

    def titles(self, category: str | None = None) -> list[str]:
        """Liste les titres de documents disponibles."""
        return sorted({
            chunk.metadata["title"]
            for chunk in self._chunks
            if category is None or chunk.metadata["category"] == category
        })


@functools.lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Instance partagee : evite de recharger le modele d'embeddings a chaque appel."""
    return HybridRetriever()
