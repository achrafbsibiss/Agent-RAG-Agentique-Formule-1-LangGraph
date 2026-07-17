"""Etape 5 : injecte les resultats d'evaluation dans le rapport et rend le PDF.

Lit reports/evaluation.csv, construit le tableau de synthese (moyennes par
categorie + global), l'insere dans reports/rapport.html a la place du marqueur
<!-- RESULTS_TABLE -->, puis convertit le HTML en PDF via Chrome sans interface.

Usage :
    python scripts/05_build_report.py
"""

import _bootstrap  # noqa: F401

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR
from src.llm.models import describe_llm

# Le template garde le marqueur ; la version remplie est un fichier distinct,
# ce qui rend le script rejouable sans abimer le gabarit.
TEMPLATE_PATH = REPORTS_DIR / "rapport.html"
FILLED_PATH = REPORTS_DIR / "rapport_final.html"
PDF_PATH = REPORTS_DIR / "rapport.pdf"
CSV_PATH = REPORTS_DIR / "evaluation.csv"

METRICS = ["latence_s", "hit_at_k", "mrr", "fidelite", "completude", "clarte"]
LABELS = {
    "latence_s": "Latence (s)", "hit_at_k": "hit@k", "mrr": "MRR",
    "fidelite": "Fidélité /5", "completude": "Complétude /5", "clarte": "Clarté /5",
}

# Emplacements possibles de Chrome/Chromium selon la plateforme.
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]


def _build_results_html(frame: pd.DataFrame) -> str:
    """Construit le tableau HTML de synthese a partir du CSV d'evaluation."""
    frame = frame[frame["latence_s"].notna()].copy()
    n_simple = int((frame.categorie == "simple").sum())
    n_complex = int((frame.categorie == "complexe").sum())

    by_cat = frame.groupby("categorie")[METRICS].mean().round(2)
    overall = frame[METRICS].mean().round(2)

    def row(name: str, series) -> str:
        cells = "".join(f'<td class="num">{series[m]:.2f}</td>' for m in METRICS)
        return f"<tr><td><strong>{name}</strong></td>{cells}</tr>"

    header = "".join(f'<th class="num">{LABELS[m]}</th>' for m in METRICS)
    body = ""
    if "simple" in by_cat.index:
        body += row(f"Questions simples ({n_simple})", by_cat.loc["simple"])
    if "complexe" in by_cat.index:
        body += row(f"Questions complexes ({n_complex})", by_cat.loc["complexe"])
    body += row(f"Global ({len(frame)})", overall)

    return (
        f'<p class="meta">Modèle : <code>{describe_llm()}</code> · '
        f'{len(frame)} questions évaluées.</p>'
        f'<table><tr><th style="width:24%">Catégorie</th>{header}</tr>{body}</table>'
    )


def _inject_results() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} absent : lancer d'abord scripts/04_evaluate.py")

    frame = pd.read_csv(CSV_PATH)
    results_html = _build_results_html(frame)

    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "<!-- RESULTS_TABLE -->", results_html
    )
    FILLED_PATH.write_text(html, encoding="utf-8")
    print(f"Resultats injectes dans {FILLED_PATH}")


def _find_chrome() -> str | None:
    for path in CHROME_CANDIDATES:
        if path and Path(path).exists():
            return path
    return None


def _render_pdf() -> None:
    chrome = _find_chrome()
    if not chrome:
        print("Chrome/Chromium introuvable. Ouvrir reports/rapport.html et "
              "'Imprimer > Enregistrer en PDF' manuellement.")
        return

    subprocess.run(
        [
            chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_PATH}", FILLED_PATH.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    size_kb = PDF_PATH.stat().st_size // 1024
    print(f"PDF genere -> {PDF_PATH} ({size_kb} Ko)")


def main() -> None:
    _inject_results()
    _render_pdf()


if __name__ == "__main__":
    main()
