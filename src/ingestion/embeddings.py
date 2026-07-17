"""Modele d'embeddings local et multilingue.

Le corpus est majoritairement anglophone alors que les questions peuvent etre
posees en francais. Un modele multilingue projette les deux langues dans le
meme espace vectoriel : une question FR retrouve un passage EN sans traduction
prealable.

L'inference passe par FastEmbed (ONNX Runtime) plutot que par
sentence-transformers/PyTorch. PyTorch ne publie plus de roue macOS x86_64
depuis la 2.2, et aucune pour Python 3.13 : ONNX rend le projet installable sur
ce poste, tout en etant plus rapide sur CPU et bien plus leger a telecharger.
"""

from __future__ import annotations

import functools

import numpy as np
from langchain_core.embeddings import Embeddings

from src.config import MODEL_CACHE_DIR, settings

# La famille E5 est entrainee avec des prefixes d'instruction : omettre
# `query: ` / `passage: ` degrade sensiblement le rappel. Les modeles
# `paraphrase-*` sont entraines sans prefixe et n'en attendent aucun.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


def _normalize(vector: np.ndarray) -> list[float]:
    """Ramene un vecteur sur la sphere unite.

    Indispensable : FastEmbed applique un mean pooling SANS normalisation
    finale pour les modeles `paraphrase-*`. Sans cette etape, un produit
    scalaire ne mesure plus un angle mais reste domine par la norme des
    vecteurs -- or la norme croit avec la longueur et la banalite du texte.
    Les passages generalistes ressortaient alors devant les passages
    specialises, quelle que soit la question posee.
    """
    norm = np.linalg.norm(vector)
    # Un vecteur nul ne peut pas etre normalise ; on le renvoie tel quel.
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


class FastEmbedEmbeddings(Embeddings):
    """Adaptateur LangChain minimal au-dessus de FastEmbed."""

    def __init__(self, model_name: str) -> None:
        import warnings

        from fastembed import TextEmbedding

        # FastEmbed signale que le mean pooling remplace le CLS pooling ; sans
        # incidence ici, puisqu'on normalise nous-memes les vecteurs en aval.
        # On masque cet avertissement pour garder une sortie CLI lisible.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*mean pooling.*", category=UserWarning)
            self._model = TextEmbedding(model_name=model_name, cache_dir=str(MODEL_CACHE_DIR))
        self._uses_prefix = "e5" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._uses_prefix:
            texts = [E5_PASSAGE_PREFIX + text for text in texts]
        return [_normalize(vector) for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        if self._uses_prefix:
            text = E5_QUERY_PREFIX + text
        return _normalize(next(iter(self._model.embed([text]))))


@functools.lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Instance partagee du modele d'embeddings configure.

    Le modele est charge en memoire une seule fois : l'indexation et la
    recherche reutilisent le meme objet.
    """
    return FastEmbedEmbeddings(settings.retrieval.embedding_model)
