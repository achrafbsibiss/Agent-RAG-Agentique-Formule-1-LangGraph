"""Etape 1 : telecharge et nettoie la base documentaire Formule 1."""

import _bootstrap  # noqa: F401  (effet de bord : ajoute la racine au sys.path)

from src.corpus.build_corpus import main

if __name__ == "__main__":
    main()
