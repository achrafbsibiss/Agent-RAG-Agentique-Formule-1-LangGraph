"""Jeu d'evaluation : 10 questions simples et 10 questions complexes.

Chaque entree porte des `expected_docs` : les titres de documents que la
recuperation devrait idealement faire remonter. Ils servent a mesurer la
pertinence des sources (hit@k) independamment de la qualite de la redaction.

- Les questions simples ciblent un fait unique dans un seul document.
- Les questions complexes exigent comparaison, calcul, synthese ou croisement
  de plusieurs documents. Elles sont volontairement bilingues (FR/EN) pour
  eprouver la recuperation cross-lingue.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: str          # 'simple' ou 'complexe'
    language: str
    expected_docs: list[str] = field(default_factory=list)


SIMPLE_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion("S01", "Qu'est-ce que le DRS en Formule 1 ?", "simple", "fr",
                 ["Drag reduction system", "Système de réduction de la traînée"]),
    EvalQuestion("S02", "What does a yellow flag mean in Formula One?", "simple", "en",
                 ["Racing flags"]),
    EvalQuestion("S03", "Combien de points rapporte une victoire en Grand Prix ?", "simple", "fr",
                 ["List of Formula One points systems", "Formula One regulations"]),
    EvalQuestion("S04", "A quoi sert un arret au stand (pit stop) en course ?", "simple", "fr",
                 ["Pit stop"]),
    EvalQuestion("S05", "Ou se situe le circuit de Monza ?", "simple", "fr",
                 ["Monza Circuit"]),
    EvalQuestion("S06", "What is a safety car used for?", "simple", "en",
                 ["Safety car"]),
    EvalQuestion("S07", "Qu'est-ce que le KERS ?", "simple", "fr",
                 ["Kinetic energy recovery system"]),
    EvalQuestion("S08", "How many world championships has Lewis Hamilton won?", "simple", "en",
                 ["Lewis Hamilton"]),
    EvalQuestion("S09", "Qu'est-ce que le parc ferme en Formule 1 ?", "simple", "fr",
                 ["Parc fermé", "Formula One regulations"]),
    EvalQuestion("S10", "A quoi sert un turbocompresseur dans un moteur ?", "simple", "fr",
                 ["Turbocharger"]),
]


COMPLEX_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion("C01",
                 "Compare Max Verstappen and Lewis Hamilton in terms of world titles won.",
                 "complexe", "en", ["Max Verstappen", "Lewis Hamilton"]),
    EvalQuestion("C02",
                 "Combien de points marque un pilote qui finit 1er, 3e puis 2e sur trois Grands Prix ?",
                 "complexe", "fr", ["List of Formula One points systems"]),
    EvalQuestion("C03",
                 "Quelles differences y a-t-il entre le circuit de Monaco et celui de Monza ?",
                 "complexe", "fr", ["Circuit de Monaco", "Monza Circuit", "Grand Prix automobile de Monaco"]),
    EvalQuestion("C04",
                 "How do DRS and KERS each help a car, and how do they differ?",
                 "complexe", "en", ["Drag reduction system", "Kinetic energy recovery system"]),
    EvalQuestion("C05",
                 "Qui a remporte le championnat pilotes en 2008 et en 2010, et avec quelle ecurie ?",
                 "complexe", "fr", ["2008 Formula One World Championship", "2010 Formula One World Championship"]),
    EvalQuestion("C06",
                 "Which teams did Fernando Alonso and Sebastian Vettel both drive for during their careers?",
                 "complexe", "en", ["Fernando Alonso", "Sebastian Vettel"]),
    EvalQuestion("C07",
                 "Comment la strategie d'arret aux stands (undercut) et le depassement "
                 "influencent-ils le resultat d'une course ?",
                 "complexe", "fr", ["Pit stop", "Overtaking", "Drafting (aerodynamics)"]),
    EvalQuestion("C08",
                 "Qui a remporte le championnat 2024 et avec quelle ecurie, et qui etait deuxieme ?",
                 "complexe", "fr", ["2024 Formula One World Championship", "Championnat du monde de Formule 1 2024"]),
    EvalQuestion("C09",
                 "Compare the power units used in modern Formula One with the role of the ERS.",
                 "complexe", "en", ["Formula One engines", "Kinetic energy recovery system", "Formula One car"]),
    EvalQuestion("C10",
                 "En quoi les carrieres d'Ayrton Senna et de Michael Schumacher se distinguent-elles ?",
                 "complexe", "fr", ["Ayrton Senna", "Michael Schumacher"]),
]


ALL_QUESTIONS: list[EvalQuestion] = SIMPLE_QUESTIONS + COMPLEX_QUESTIONS
