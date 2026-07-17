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


def _build(temperature: float) -> BaseChatModel:
    provider = settings.llm.provider

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise LLMProviderError(
                "GROQ_API_KEY absente. Creer une cle gratuite sur "
                "https://console.groq.com/keys puis la renseigner dans .env"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.llm.groq_model,
            temperature=temperature,
            max_retries=3,
        )

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise LLMProviderError(
                "GOOGLE_API_KEY absente. Creer une cle gratuite sur "
                "https://aistudio.google.com/apikey puis la renseigner dans .env"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm.google_model,
            temperature=temperature,
            max_retries=3,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm.ollama_model,
            base_url=settings.llm.ollama_base_url,
            temperature=temperature,
        )

    raise LLMProviderError(
        f"LLM_PROVIDER inconnu : '{provider}'. Valeurs acceptees : groq, google, ollama."
    )


@functools.lru_cache(maxsize=4)
def get_llm(role: str = "reasoner") -> BaseChatModel:
    """Renvoie le modele associe a un role (instances mises en cache)."""
    temperatures = {
        "reasoner": settings.llm.temperature,
        "grader": 0.0,
        "judge": settings.llm.judge_temperature,
    }
    if role not in temperatures:
        raise ValueError(f"Role inconnu : '{role}'. Attendu : {list(temperatures)}")
    return _build(temperatures[role])


def describe_llm() -> str:
    """Description lisible du modele actif, affichee par la CLI et les rapports."""
    provider = settings.llm.provider
    model = {
        "groq": settings.llm.groq_model,
        "google": settings.llm.google_model,
        "ollama": settings.llm.ollama_model,
    }.get(provider, "?")
    return f"{provider}:{model}"
