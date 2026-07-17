"""Configuration centrale du projet.

Toutes les constantes et chemins du systeme sont definis ici afin qu'aucun
module metier n'ait a manipuler de valeur codee en dur.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Chemins du projet
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _directory in (RAW_DIR, CHROMA_DIR, REPORTS_DIR, FIGURES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

CORPUS_FILE = RAW_DIR / "corpus_f1.jsonl"
COLLECTION_NAME = "formula1"

# Cache des modeles d'embeddings ONNX. Fixe explicitement pour eviter le
# repertoire temporaire par defaut de FastEmbed (/var/folders...), efface au
# redemarrage : sans cela, le modele (jusqu'a 2 Go) serait re-telecharge.
MODEL_CACHE_DIR = DATA_DIR / ".model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LLMConfig:
    """Parametres du modele de langage principal et du modele juge."""

    provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
    temperature: float = _env_float("LLM_TEMPERATURE", 0.0)
    judge_temperature: float = _env_float("JUDGE_TEMPERATURE", 0.0)

    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass(frozen=True)
class RetrievalConfig:
    """Parametres de decoupage, d'indexation et de recherche."""

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    # Nombre de candidats examines par MMR avant selection finale.
    fetch_k: int = 20
    # Ponderation dense/lexical de la recherche hybride.
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    # Nombre maximum de chunks retenus pour un meme document source.
    # Le corpus est tres desequilibre (la page « Formule 1 » pese a elle seule
    # 390 chunks) : sans plafond, un document generaliste monopolise le top-k
    # et evince les pages specialisees qui repondent reellement a la question.
    max_per_document: int = 2


@dataclass(frozen=True)
class GraphConfig:
    """Garde-fous de la boucle agentique.

    Ces limites empechent le graphe de boucler indefiniment lorsque le corpus
    ne contient pas la reponse : chaque compteur est porte par le state.
    """

    # Tours maximum agent -> outils -> agent.
    max_tool_iterations: int = 4
    # Reformulations maximum de la requete quand les documents sont juges hors sujet.
    max_query_rewrites: int = 2
    # Regenerations maximum quand la reponse est jugee non ancree dans les sources.
    max_generation_retries: int = 1
    # Sous-questions maximum produites par le planificateur.
    max_subquestions: int = 4


# --------------------------------------------------------------------------
# Categories du corpus : servent de filtre de metadonnees pour les outils
# --------------------------------------------------------------------------
CATEGORIES: tuple[str, ...] = (
    "reglement",
    "ecurie",
    "pilote",
    "circuit",
    "glossaire",
    "saison",
    "technique",
    "histoire",
)


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


settings = Settings()
