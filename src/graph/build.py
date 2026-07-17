"""Assemblage du graphe agentique LangGraph.

Le graphe est construit noeud par noeud plutot que via un agent preconstruit
(`create_agent`), ce qui permet d'inserer les etapes propres a l'approche
Agentic RAG : analyse, planification, evaluation de la recuperation,
reformulation et verification de l'ancrage.

Topologie :

    START
      v
    analyze_query ------(simple)------> agent <----------------+
      |                                  |  ^                  |
      +---(complexe)--> plan_research ---+  |                  |
                                           |                   |
                        +---(tool_calls)---+                   |
                        v                                      |
                      tools --> grade_documents ---(pertinent)-+
                                     |                         |
                                     +---(hors sujet)--> rewrite_query
                                     |
    generate <---(aucun tool_call)---+
      v
    verify_grounding ---(non ancre)--> generate
      v
     END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.graph.edges import (
    route_after_agent,
    route_after_analysis,
    route_after_grading,
    route_after_verification,
)
from src.graph.nodes import (
    agent,
    analyze_query,
    generate,
    grade_documents,
    plan_research,
    rewrite_query,
    verify_grounding,
)
from src.graph.state import AgentState
from src.tools.f1_tools import F1_TOOLS


def build_graph(checkpointer: MemorySaver | None = None):
    """Compile le graphe agentique.

    Args:
        checkpointer: Sauvegarde du state entre les tours. Passer `None` pour
            un graphe sans memoire (utile pour la visualisation ou pour isoler
            chaque question du harnais d'evaluation).
    """
    builder = StateGraph(AgentState)

    builder.add_node("analyze_query", analyze_query)
    builder.add_node("plan_research", plan_research)
    builder.add_node("agent", agent)
    # ToolNode sait interpreter les `Command` renvoyes par les outils de
    # recuperation, qui ecrivent directement dans `state["documents"]`.
    builder.add_node("tools", ToolNode(F1_TOOLS))
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate", generate)
    builder.add_node("verify_grounding", verify_grounding)

    builder.add_edge(START, "analyze_query")

    builder.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {"plan_research": "plan_research", "agent": "agent"},
    )
    builder.add_edge("plan_research", "agent")

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "generate": "generate"},
    )

    # Tout resultat d'outil est evalue avant de revenir a l'agent : c'est la
    # boucle d'auto-correction de la recuperation.
    builder.add_edge("tools", "grade_documents")

    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"agent": "agent", "rewrite_query": "rewrite_query"},
    )
    builder.add_edge("rewrite_query", "agent")

    builder.add_edge("generate", "verify_grounding")
    builder.add_conditional_edges(
        "verify_grounding",
        route_after_verification,
        {"generate": "generate", "__end__": END},
    )

    return builder.compile(checkpointer=checkpointer)


def build_agent():
    """Graphe pret a l'emploi, avec memoire conversationnelle en RAM.

    `MemorySaver` conserve le state par `thread_id` : deux conversations
    lancees avec des identifiants differents n'ont aucune visibilite l'une sur
    l'autre, et une conversation reprise retrouve son historique.
    """
    return build_graph(checkpointer=MemorySaver())
