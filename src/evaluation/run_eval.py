"""Harnais d'evaluation du systeme RAG agentique.

Pour chaque question du jeu de test, le harnais :
1. execute le graphe complet (chaque question sur son propre thread, sans
   contamination de memoire) et chronometre la latence bout-en-bout ;
2. mesure la pertinence des documents recuperes (hit@k, MRR) ;
3. fait noter la reponse par le LLM juge.

Sorties dans reports/ : un CSV detaille par question et un rapport Markdown
agrege (moyennes par categorie), directement reutilisable dans le rendu PDF.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from src.config import REPORTS_DIR
from src.evaluation.judge import judge_answer
from src.evaluation.metrics import hit_at_k, reciprocal_rank
from src.evaluation.questions import ALL_QUESTIONS, EvalQuestion
from src.graph.build import build_graph
from src.llm.models import describe_llm

logger = logging.getLogger(__name__)


def _run_one(graph, question: EvalQuestion) -> dict:
    """Execute une question et collecte toutes les metriques la concernant."""
    # thread_id unique par question : le harnais evalue des reponses
    # independantes, pas une conversation.
    config = {"configurable": {"thread_id": f"eval-{question.id}"}}

    start = time.perf_counter()
    state = graph.invoke(
        {"messages": [("user", question.question)]},
        config=config,
    )
    latency = time.perf_counter() - start

    documents = state.get("documents", [])
    answer = state.get("answer", "")

    score = judge_answer(question.question, answer, documents)

    retrieved_titles = sorted({doc.metadata.get("title", "") for doc in documents})
    logger.info("[%s] %.1fs  hit@k=%.2f  fidelite=%d  %s",
                question.id, latency,
                hit_at_k(question.expected_docs, documents),
                score.fidelite, question.question[:45])

    return {
        "id": question.id,
        "categorie": question.category,
        "langue": question.language,
        "question": question.question,
        "latence_s": round(latency, 2),
        "nb_docs": len(documents),
        "hit_at_k": round(hit_at_k(question.expected_docs, documents), 3),
        "mrr": round(reciprocal_rank(question.expected_docs, documents), 3),
        "grounded": state.get("grounded", None),
        "complexite_detectee": state.get("complexity", ""),
        "fidelite": score.fidelite,
        "completude": score.completude,
        "clarte": score.clarte,
        "sources": " | ".join(retrieved_titles),
        "commentaire_juge": score.commentaire,
        "reponse": answer,
    }


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Moyennes par categorie de question, plus une ligne globale."""
    metrics = ["latence_s", "hit_at_k", "mrr", "fidelite", "completude", "clarte"]
    by_category = frame.groupby("categorie")[metrics].mean().round(3)
    overall = frame[metrics].mean().round(3)
    overall.name = "global"
    return pd.concat([by_category, overall.to_frame().T])


def _write_report(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Resultats de l'evaluation — Agent RAG Formule 1\n")
    lines.append(f"- Modele LLM : `{describe_llm()}`")
    lines.append(f"- Questions evaluees : {len(frame)} "
                 f"({(frame.categorie == 'simple').sum()} simples, "
                 f"{(frame.categorie == 'complexe').sum()} complexes)\n")

    lines.append("## Synthese par categorie\n")
    lines.append(summary.to_markdown())
    lines.append("\n## Detail par question\n")
    detail_cols = ["id", "categorie", "langue", "latence_s", "hit_at_k", "mrr",
                   "fidelite", "completude", "clarte"]
    lines.append(frame[detail_cols].to_markdown(index=False))

    lines.append("\n## Lecture des metriques\n")
    lines.append("- **latence_s** : temps de reponse bout-en-bout (graphe complet).")
    lines.append("- **hit_at_k** : fraction des documents attendus effectivement recuperes.")
    lines.append("- **mrr** : qualite du classement (1.0 = source pertinente en tete).")
    lines.append("- **fidelite / completude / clarte** : notes du LLM juge sur 5.")

    report_path = REPORTS_DIR / "evaluation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport Markdown -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation du systeme RAG agentique")
    parser.add_argument("--limit", type=int, default=0,
                        help="n'evaluer que les N premieres questions (test rapide)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="  %(message)s" if not args.verbose else "  [%(name)s] %(message)s",
    )
    # Le graphe est sans memoire : chaque question est isolee.
    graph = build_graph(checkpointer=None)

    questions = ALL_QUESTIONS[: args.limit] if args.limit else ALL_QUESTIONS
    print(f"Evaluation de {len(questions)} questions avec {describe_llm()}\n")

    csv_path = REPORTS_DIR / "evaluation.csv"
    rows: list[dict] = []
    for question in questions:
        try:
            rows.append(_run_one(graph, question))
        except Exception as exc:  # quota, reseau : on n'annule pas tout le run
            logger.warning("[%s] ECHEC (%s) : question ignoree", question.id, type(exc).__name__)
            rows.append({
                "id": question.id, "categorie": question.category,
                "langue": question.language, "question": question.question,
                "erreur": f"{type(exc).__name__}: {exc}",
            })
        # Sauvegarde incrementale : un plantage ulterieur ne perd pas les
        # questions deja evaluees.
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = pd.DataFrame(rows)
    # Seules les lignes reellement evaluees comptent dans les agregats.
    frame = frame[frame.get("latence_s").notna()] if "latence_s" in frame else frame
    print(f"\nCSV detaille    -> {csv_path}")

    if frame.empty:
        print("Aucune question evaluee (quota epuise ?). Rapport non genere.")
        return

    summary = _aggregate(frame)
    _write_report(frame, summary)

    print("\n=== Synthese ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
