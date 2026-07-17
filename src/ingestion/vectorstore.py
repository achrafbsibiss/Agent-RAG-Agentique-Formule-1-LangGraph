"""Construction et chargement de la base vectorielle Chroma."""

from __future__ import annotations

import logging
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument

from src.config import CHROMA_DIR, COLLECTION_NAME
from src.ingestion.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Chroma limite la taille des lots envoyes au modele d'embeddings.
BATCH_SIZE = 128


def build_index(chunks: list[LCDocument], reset: bool = True) -> Chroma:
    """Indexe les chunks dans Chroma et persiste la collection sur disque."""
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        store.add_documents(
            documents=batch,
            ids=[chunk.metadata["chunk_id"] for chunk in batch],
        )
        logger.info("Indexe %d / %d chunks", min(start + BATCH_SIZE, len(chunks)), len(chunks))

    return store


def load_vectorstore() -> Chroma:
    """Recharge la collection persistee."""
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    if store._collection.count() == 0:
        raise RuntimeError(
            "Base vectorielle vide. Lancer `python scripts/02_index.py` avant d'interroger l'agent."
        )
    return store
