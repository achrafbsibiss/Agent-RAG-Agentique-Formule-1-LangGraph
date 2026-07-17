"""Construction de la base documentaire Formule 1 depuis Wikipedia.

Le script telecharge chaque page listee dans `pages.PAGES`, nettoie le texte
brut (suppression des sections non informatives) et ecrit un fichier JSONL,
une ligne par document. Aucune cle d'API n'est requise.

Usage :
    python -m src.corpus.build_corpus
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass

import requests # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import CORPUS_FILE
from src.corpus.pages import PAGES

logger = logging.getLogger(__name__)

API_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "F1-AgenticRAG/1.0 (projet academique; contact: etudiant@example.com)"

# Sections de fin de page sans valeur informative pour le RAG.
DROP_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "sources", "footnotes", "citations",
    "voir aussi", "notes et références", "références", "liens externes",
    "bibliographie", "annexes",
}

# Longueur minimale d'un document conserve, en caracteres.
MIN_DOC_CHARS = 400


@dataclass
class Document:
    """Un document du corpus, avant decoupage en chunks."""

    doc_id: str
    title: str
    category: str
    language: str
    url: str
    text: str


def _clean_text(raw: str) -> str:
    """Nettoie l'extrait Wikipedia en texte brut.

    Retire les sections terminales non informatives, les titres de section
    residuels et normalise les espaces.
    """
    lines = raw.split("\n")
    kept: list[str] = []
    skipping = False

    for line in lines:
        heading = re.match(r"^(={2,6})\s*(.+?)\s*\1$", line.strip())
        if heading:
            name = heading.group(2).strip().lower()
            # Une section a supprimer coupe tout jusqu'au prochain titre.
            skipping = name in DROP_SECTIONS
            if not skipping:
                # Le titre est conserve en clair : il porte du contexte utile
                # au moment du decoupage semantique.
                kept.append(f"\n## {heading.group(2).strip()}\n")
            continue

        if not skipping:
            kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_page(title: str, lang: str) -> tuple[str, str] | None:
    """Recupere l'extrait texte d'une page. Renvoie (titre_resolu, texte)."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "formatversion": 2,
    }
    response = requests.get(
        API_TEMPLATE.format(lang=lang),
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])

    if not pages or pages[0].get("missing"):
        return None

    page = pages[0]
    extract = page.get("extract") or ""
    return page.get("title", title), extract


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return slug.strip("-")


def build_corpus() -> list[Document]:
    """Telecharge et nettoie l'ensemble des pages du plan de corpus."""
    documents: list[Document] = []
    missing: list[str] = []

    for title, category, lang in PAGES:
        try:
            result = _fetch_page(title, lang)
        except Exception as exc:  # réseau, quota, page malformée
            logger.warning("Echec du telechargement de '%s' : %s", title, exc)
            missing.append(title)
            continue

        if result is None:
            logger.warning("Page introuvable : '%s' (%s)", title, lang)
            missing.append(title)
            continue

        resolved_title, raw_text = result
        text = _clean_text(raw_text)

        if len(text) < MIN_DOC_CHARS:
            logger.warning("Page trop courte, ignoree : '%s'", resolved_title)
            missing.append(title)
            continue

        documents.append(
            Document(
                doc_id=f"{lang}-{_slugify(resolved_title)}",
                title=resolved_title,
                category=category,
                language=lang,
                url=f"https://{lang}.wikipedia.org/wiki/{resolved_title.replace(' ', '_')}",
                text=text,
            )
        )
        logger.info("OK  %-55s %7d car.  [%s]", resolved_title[:55], len(text), category)
        # Courtoisie envers l'API publique de Wikipedia.
        time.sleep(0.2)

    if missing:
        logger.warning("%d page(s) non recuperee(s) : %s", len(missing), ", ".join(missing))

    return documents


def save_corpus(documents: list[Document]) -> None:
    with CORPUS_FILE.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def load_corpus() -> list[Document]:
    """Relit le corpus depuis le disque."""
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Corpus absent : {CORPUS_FILE}. Lancer d'abord `python scripts/01_build_corpus.py`."
        )
    with CORPUS_FILE.open(encoding="utf-8") as handle:
        return [Document(**json.loads(line)) for line in handle if line.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    documents = build_corpus()
    save_corpus(documents)

    total_chars = sum(len(doc.text) for doc in documents)
    print(f"\n{len(documents)} documents -> {CORPUS_FILE}")
    print(f"{total_chars:,} caracteres au total".replace(",", " "))

    by_category: dict[str, int] = {}
    for doc in documents:
        by_category[doc.category] = by_category.get(doc.category, 0) + 1
    print("\nRepartition par categorie :")
    for category, count in sorted(by_category.items(), key=lambda item: -item[1]):
        print(f"  {category:12s} {count:3d}")


if __name__ == "__main__":
    main()
