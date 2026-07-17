"""Etape 3 : exporte la topologie du graphe agentique.

Produit deux fichiers dans reports/figures/ :
- graphe.mmd : source Mermaid (toujours generee, sans dependance) ;
- graphe.png : rendu PNG (via l'API mermaid.ink, si le reseau est disponible).

Le graphe est compile SANS checkpointer : la memoire ne change pas la
topologie et n'a pas a etre visualisee.
"""

import _bootstrap  # noqa: F401

from src.config import FIGURES_DIR
from src.graph.build import build_graph


def main() -> None:
    graph = build_graph(checkpointer=None)
    drawable = graph.get_graph()

    mermaid_source = drawable.draw_mermaid()
    mmd_path = FIGURES_DIR / "graphe.mmd"
    mmd_path.write_text(mermaid_source, encoding="utf-8")
    print(f"Source Mermaid -> {mmd_path}")

    try:
        png = drawable.draw_mermaid_png()
        png_path = FIGURES_DIR / "graphe.png"
        png_path.write_bytes(png)
        print(f"Rendu PNG      -> {png_path}")
    except Exception as exc:  # rendu distant indisponible (hors ligne, quota)
        print(f"Rendu PNG indisponible ({exc}).")
        print("Le fichier .mmd reste utilisable : le coller dans https://mermaid.live")

    print("\n--- Apercu Mermaid ---")
    print(mermaid_source)


if __name__ == "__main__":
    main()
