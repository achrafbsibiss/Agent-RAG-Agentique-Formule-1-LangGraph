"""Decoupage des documents en chunks indexables.

Le decoupage respecte la hierarchie du texte nettoye (titres `##`, paragraphes,
phrases) pour eviter de couper une idee en plein milieu. Chaque chunk est
prefixe de son titre et de sa section : un chunk isole reste ainsi
interpretable par le LLM, qui ne voit pas le document complet.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.corpus.build_corpus import Document


def _current_section(text: str, position: int) -> str:
    """Retrouve le titre de section actif a une position donnee du document."""
    headings = list(re.finditer(r"^## (.+)$", text[:position], flags=re.MULTILINE))
    return headings[-1].group(1).strip() if headings else "Introduction"


def chunk_documents(documents: list[Document]) -> list[LCDocument]:
    """Transforme les documents bruts en chunks LangChain enrichis."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[LCDocument] = []

    for document in documents:
        pieces = splitter.split_text(document.text)
        cursor = 0

        for index, piece in enumerate(pieces):
            # `str.find` depuis le curseur : evite de retomber sur une
            # occurrence anterieure quand un passage se repete.
            found = document.text.find(piece[:80], cursor)
            position = found if found != -1 else cursor
            cursor = position + 1
            section = _current_section(document.text, position)

            # L'en-tete rend le chunk auto-suffisant une fois sorti de son
            # document d'origine.
            header = f"[{document.title} — {section}]"
            chunks.append(
                LCDocument(
                    page_content=f"{header}\n{piece.strip()}",
                    metadata={
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "category": document.category,
                        "language": document.language,
                        "url": document.url,
                        "section": section,
                        "chunk_index": index,
                        "chunk_id": f"{document.doc_id}#{index}",
                    },
                )
            )

    return chunks
