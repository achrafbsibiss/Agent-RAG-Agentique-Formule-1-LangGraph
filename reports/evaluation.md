# Resultats de l'evaluation — Agent RAG Formule 1

- Modele LLM : `groq:llama-3.1-8b-instant (roles legers: llama-3.1-8b-instant)`
- Questions evaluees : 20 (10 simples, 10 complexes)

## Synthese par categorie

|          |   latence_s |   hit_at_k |   mrr |   fidelite |   completude |   clarte |
|:---------|------------:|-----------:|------:|-----------:|-------------:|---------:|
| complexe |     110.621 |      0.7   | 0.65  |       4.1  |         3.4  |      4.5 |
| simple   |     111.733 |      0.65  | 0.464 |       3    |         2.9  |      3.5 |
| global   |     111.177 |      0.675 | 0.557 |       3.55 |         3.15 |      4   |

## Detail par question

| id   | categorie   | langue   |   latence_s |   hit_at_k |   mrr |   fidelite |   completude |   clarte |
|:-----|:------------|:---------|------------:|-----------:|------:|-----------:|-------------:|---------:|
| S01  | simple      | fr       |       73.42 |        0.5 | 0.167 |          4 |            4 |        5 |
| S02  | simple      | en       |       82.66 |        1   | 1     |          4 |            4 |        5 |
| S03  | simple      | fr       |      201.72 |        0   | 0     |          5 |            5 |        5 |
| S04  | simple      | fr       |      159.4  |        1   | 0.143 |          4 |            4 |        5 |
| S05  | simple      | fr       |       63.88 |        1   | 1     |          5 |            4 |        5 |
| S06  | simple      | en       |       86.06 |        1   | 1     |          4 |            4 |        5 |
| S07  | simple      | fr       |       83.61 |        1   | 1     |          0 |            0 |        0 |
| S08  | simple      | en       |      204.26 |        1   | 0.333 |          4 |            4 |        5 |
| S09  | simple      | fr       |      156.2  |        0   | 0     |          0 |            0 |        0 |
| S10  | simple      | fr       |        6.12 |        0   | 0     |          0 |            0 |        0 |
| C01  | complexe    | en       |      116.96 |        1   | 1     |          4 |            3 |        5 |
| C02  | complexe    | fr       |      121.84 |        0   | 0     |          5 |            4 |        4 |
| C03  | complexe    | fr       |      123.03 |        1   | 1     |          4 |            3 |        4 |
| C04  | complexe    | en       |      103.46 |        1   | 1     |          4 |            4 |        5 |
| C05  | complexe    | fr       |       71.32 |        1   | 1     |          2 |            3 |        4 |
| C06  | complexe    | en       |       78.63 |        1   | 0.5   |          4 |            2 |        5 |
| C07  | complexe    | fr       |      126.53 |        0   | 0     |          4 |            4 |        5 |
| C08  | complexe    | fr       |       69.71 |        1   | 1     |          5 |            3 |        4 |
| C09  | complexe    | en       |      147.26 |        0   | 0     |          5 |            4 |        4 |
| C10  | complexe    | fr       |      147.47 |        1   | 1     |          4 |            4 |        5 |

## Lecture des metriques

- **latence_s** : temps de reponse bout-en-bout (graphe complet).
- **hit_at_k** : fraction des documents attendus effectivement recuperes.
- **mrr** : qualite du classement (1.0 = source pertinente en tete).
- **fidelite / completude / clarte** : notes du LLM juge sur 5.