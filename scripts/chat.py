"""Interface conversationnelle en ligne de commande.

Illustre la memoire du systeme : toutes les questions d'une session partagent
le meme `thread_id`, donc l'agent resout les references d'un tour a l'autre
("et son coequipier ?"). La commande `/reset` ouvre un nouveau thread.

Usage :
    python scripts/chat.py
    python scripts/chat.py --trace     # affiche le journal de decision du graphe
"""

import _bootstrap  # noqa: F401

import argparse
import logging
import uuid

from langchain_core.messages import HumanMessage

from src.graph.build import build_agent
from src.llm.models import describe_llm


BANNER = """
============================================================
  Agent RAG Formule 1  —  architecture LangGraph
============================================================
  Modele : {model}
  Posez vos questions en francais ou en anglais.
  Commandes : /reset (nouvelle conversation) · /quit (sortir)
============================================================
"""


# Libelle lisible affiche pendant l'execution de chaque noeud du graphe.
# L'utilisateur voit ainsi la progression au lieu d'un ecran fige pendant
# les ~30 s que dure le raisonnement complet.
NODE_LABELS = {
    "analyze_query": "analyse de la question",
    "plan_research": "planification (question complexe)",
    "agent": "raisonnement de l'agent",
    "tools": "recherche documentaire",
    "grade_documents": "evaluation des documents",
    "rewrite_query": "reformulation de la requete",
    "generate": "redaction de la reponse",
    "verify_grounding": "verification anti-hallucination",
}


def _run_with_progress(agent, question: str, config: dict) -> dict:
    """Execute le graphe en affichant chaque etape, puis renvoie le state final.

    Le flux `stream_mode="updates"` sert uniquement a l'affichage de la
    progression. Le state final est ensuite relu via `get_state`, qui applique
    les reducteurs et fournit donc la vue coherente (documents accumules,
    reponse finale) plutot que le dernier delta brut.
    """
    print("  ", end="", flush=True)

    for step in agent.stream(
        {"messages": [HumanMessage(question)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name in step:  # {nom_du_noeud: delta} a chaque etape
            print(f"· {NODE_LABELS.get(node_name, node_name)} ", end="", flush=True)

    print()  # fin de la ligne de progression
    return agent.get_state(config).values


def _print_trace(state: dict) -> None:
    print("\n  ─ trace de decision ─")
    for note in state.get("notes", []):
        print(f"    · {note}")
    documents = state.get("documents", [])
    if documents:
        sources = sorted({doc.metadata["title"] for doc in documents})
        print(f"    · sources : {', '.join(sources)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat avec l'agent RAG Formule 1")
    parser.add_argument("--trace", action="store_true", help="afficher le journal de decision")
    parser.add_argument("--verbose", action="store_true", help="logs detailles des noeuds")
    args = parser.parse_args()

    # Par defaut on garde la sortie propre pour la demo : seuls les messages
    # d'erreur remontent. `--verbose` reactive les logs detailles des noeuds.
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="  [%(name)s] %(message)s",
    )

    agent = build_agent()
    thread_id = str(uuid.uuid4())
    print(BANNER.format(model=describe_llm()))

    while True:
        try:
            question = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Au revoir.")
            break
        if question.lower() == "/reset":
            thread_id = str(uuid.uuid4())
            print("  (nouvelle conversation : la memoire est videe)\n")
            continue

        config = {"configurable": {"thread_id": thread_id}}
        state = _run_with_progress(agent, question, config)

        print(f"\nAgent > {state.get('answer', '(pas de reponse)')}\n")
        if args.trace:
            _print_trace(state)


if __name__ == "__main__":
    main()
