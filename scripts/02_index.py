"""Etape 2 : decoupe le corpus en chunks et construit la base vectorielle."""

import _bootstrap  # noqa: F401

import logging

from src.config import CHROMA_DIR, settings
from src.corpus.build_corpus import load_corpus
from src.ingestion.chunker import chunk_documents
from src.ingestion.vectorstore import build_index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    documents = load_corpus()
    print(f"{len(documents)} documents charges")

    chunks = chunk_documents(documents)
    print(f"{len(chunks)} chunks (taille={settings.retrieval.chunk_size}, "
          f"recouvrement={settings.retrieval.chunk_overlap})")

    lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"longueur moyenne : {sum(lengths) // len(lengths)} caracteres")

    print(f"\nIndexation avec {settings.retrieval.embedding_model} "
          f"(premier lancement : telechargement du modele)...")
    store = build_index(chunks)

    print(f"\nOK : {store._collection.count()} vecteurs dans {CHROMA_DIR}")


if __name__ == "__main__":
    main()
