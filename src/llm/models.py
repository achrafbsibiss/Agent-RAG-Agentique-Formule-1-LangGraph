"""Fabrique de modeles de langage.

Le graphe n'instancie jamais un fournisseur directement : il demande un role
(`reasoner`, `grader`, `judge`) et recoit un `BaseChatModel`. Changer de
fournisseur se limite alors a editer `.env`, sans toucher au graphe.

Roles :
- `reasoner` : planification, appels d'outils, redaction de la reponse.
- `grader`   : decisions binaires structurees (pertinence, ancrage). Temperature
               nulle et sorties contraintes par un schema Pydantic.
- `judge`    : evaluation hors ligne des reponses, utilise par le harnais de test.
"""

from __future__ import annotations

import functools
import os

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings


class LLMProviderError(RuntimeError):
    """Configuration de fournisseur absente ou invalide."""


def _build(model_name: str, temperature: float) -> BaseChatModel:
    provider = settings.llm.provider

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise LLMProviderError(
                "GROQ_API_KEY absente. Creer une cle gratuite sur "
                "https://console.groq.com/keys puis la renseigner dans .env"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, temperature=temperature, max_retries=3)

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise LLMProviderError(
                "GOOGLE_API_KEY absente. Creer une cle gratuite sur "
                "https://aistudio.google.com/apikey puis la renseigner dans .env"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_retries=3)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            base_url=settings.llm.ollama_base_url,
            temperature=temperature,
        )

    raise LLMProviderError(
        f"LLM_PROVIDER inconnu : '{provider}'. Valeurs acceptees : groq, google, ollama."
    )


def _full_model() -> str:
    return {
        "groq": settings.llm.groq_model,
        "google": settings.llm.google_model,
        "ollama": settings.llm.ollama_model,
    }[settings.llm.provider]


def _fast_model() -> str:
    """Modele leger pour les roles a fort volume (Ollama n'en a qu'un seul)."""
    return {
        "groq": settings.llm.groq_fast_model,
        "google": settings.llm.google_fast_model,
        "ollama": settings.llm.ollama_model,
    }[settings.llm.provider]


# Chaque role -> (selecteur de modele, temperature). Le raisonneur utilise le
# gros modele ; l'evaluateur et le juge, a fort volume et faible enjeu, sont
# routes vers le modele leger pour menager le quota de tokens.
_ROLES = {
    "reasoner": (_full_model, lambda: settings.llm.temperature),
    "grader": (_fast_model, lambda: 0.0),
    "judge": (_fast_model, lambda: settings.llm.judge_temperature),
}


@functools.lru_cache(maxsize=4)
def get_llm(role: str = "reasoner") -> BaseChatModel:
    """Renvoie le modele associe a un role (instances mises en cache)."""
    if role not in _ROLES:
        raise ValueError(f"Role inconnu : '{role}'. Attendu : {list(_ROLES)}")
    model_selector, temperature_selector = _ROLES[role]
    return _build(model_selector(), temperature_selector())


def describe_llm() -> str:
    """Description lisible des modeles actifs, affichee par la CLI et les rapports."""
    provider = settings.llm.provider
    return f"{provider}:{_full_model()} (roles legers: {_fast_model()})"
