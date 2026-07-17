"""Juge LLM : note la qualite d'une reponse sur des criteres explicites.

Le juge lit la question, les passages reellement fournis a l'agent et la
reponse produite. Il n'a pas acces a une reponse de reference : il evalue la
fidelite aux sources et la completude, pas la conformite a un corrige. Cela
convient a un corpus encyclopedique ou plusieurs formulations sont correctes.
"""

from __future__ import annotations

from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel, Field

from src.llm.models import get_llm


class QualityScore(BaseModel):
    fidelite: int = Field(description="1-5 : la reponse est-elle fidele aux passages, sans invention ?")
    completude: int = Field(description="1-5 : la reponse couvre-t-elle tout ce que la question demande ?")
    clarte: int = Field(description="1-5 : la reponse est-elle claire, structuree et bien redigee ?")
    commentaire: str = Field(description="Justification en une phrase")


JUDGE_PROMPT = """Tu es un evaluateur rigoureux de reponses d'un assistant \
documentaire Formule 1. Tu notes une reponse sur trois criteres, de 1 (tres \
mauvais) a 5 (excellent).

Question posee :
{question}

Passages qui etaient a disposition de l'assistant :
{documents}

Reponse produite par l'assistant :
{answer}

Criteres :
- fidelite : chaque fait de la reponse est-il soutenu par les passages ? Une \
invention ou une erreur factuelle doit fortement baisser cette note.
- completude : la reponse traite-t-elle tous les aspects de la question ? Une \
comparaison qui n'aborde qu'un cote est incomplete.
- clarte : la reponse est-elle directe, structuree et lisible ?

Si l'assistant declare honnetement manquer d'information alors que les passages \
la contenaient, baisse completude. S'il l'invente, baisse fidelite.
"""


def judge_answer(
    question: str,
    answer: str,
    documents: list[LCDocument],
) -> QualityScore:
    """Note une reponse. Renvoie des scores par defaut si le juge echoue."""
    from src.tools.f1_tools import format_documents

    judge = get_llm("judge").with_structured_output(QualityScore)
    try:
        return judge.invoke(
            JUDGE_PROMPT.format(
                question=question,
                documents=format_documents(documents),
                answer=answer,
            )
        )
    except Exception as exc:  # quota, sortie non conforme
        return QualityScore(fidelite=0, completude=0, clarte=0,
                            commentaire=f"echec du juge : {exc}")
