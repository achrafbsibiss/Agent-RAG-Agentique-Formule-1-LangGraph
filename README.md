# 🏎️ Agent RAG Agentique — Formule 1 (LangGraph)

Système **Agentic RAG** qui répond à des questions simples et complexes sur la
Formule 1 à partir d'une base documentaire construite depuis Wikipédia. Le
graphe de raisonnement est **entièrement écrit à la main avec LangGraph** — sans
`create_agent` — afin de contrôler chaque étape : analyse, planification,
récupération, auto-correction et vérification anti-hallucination.

> Projet réalisé pour l'évaluation de fin de module (Master IIBDCC — SMA & IAD).

---

## 1. Ce que fait le système

- **Bilingue FR / EN** : questions et réponses en français ou en anglais, sur un
  corpus majoritairement anglophone (récupération cross-lingue).
- **Agentique** : l'agent décide quels outils appeler, juge lui-même si les
  documents récupérés sont pertinents, reformule sa requête en cas d'échec, et
  vérifie que sa réponse finale est bien ancrée dans les sources.
- **Mémoire** : chaque conversation possède un fil (`thread_id`) ; l'agent
  résout les références implicites d'un tour à l'autre.
- **Gratuit** : LLM via une clé gratuite (Groq / Google Gemini) ou en local
  (Ollama). Embeddings 100 % locaux (ONNX, aucune clé).

## 2. Architecture du graphe

```
        ┌───────────────┐
        │ analyze_query │  langue · question autonome · simple/complexe
        └───────┬───────┘
       simple   │   complexe
         ┌──────┴───────┐
         ▼              ▼
       agent  ◄──  plan_research      décomposition en sous-questions
         │  ▲
  appel  │  │  documents pertinents
  outil  ▼  │
       tools │
         │   │
         ▼   │
  grade_documents ──(hors sujet)──► rewrite_query ──┐
         │                                          │
   (pertinent → agent)                              │
         │  ◄───────────────────────────────────────┘
         ▼  (plus d'outil à appeler)
      generate ◄──(réponse non ancrée)──┐
         │                              │
         ▼                              │
   verify_grounding ───────────────────┘
         │
         ▼
        END
```

| Nœud | Rôle |
|------|------|
| `analyze_query` | détecte la langue, rend la question autonome (mémoire), route simple/complexe |
| `plan_research` | décompose une question complexe en sous-questions atomiques |
| `agent` | raisonne et choisit les outils (boucle ReAct maison, budgétée) |
| `tools` | exécute les outils et écrit les documents dans le state |
| `grade_documents` | juge la pertinence du contexte récupéré (auto-correction CRAG) |
| `rewrite_query` | reformule la requête après un échec de récupération |
| `generate` | rédige la réponse ancrée, avec citations `[SOURCE n]` |
| `verify_grounding` | vérifie l'absence d'hallucination, regénère si besoin |

Le state, la mémoire et le routage conditionnel sont détaillés dans
[`src/graph/`](src/graph/).

## 3. Outils de l'agent

| Outil | Usage |
|-------|-------|
| `rechercher_documents` | recherche hybride (dense + BM25) sur tout le corpus |
| `rechercher_par_categorie` | recherche filtrée (règlement, pilote, circuit…) |
| `comparer_entites` | récupère en parallèle les faits sur deux entités à comparer |
| `inventaire_corpus` | liste les catégories et documents disponibles |
| `calculer_points_championnat` | calcul déterministe des points (barème officiel) |

## 4. Installation

```bash
cd f1_agentic_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# éditer .env : choisir LLM_PROVIDER et renseigner la clé correspondante
```

Obtenir une clé **gratuite** :
- Groq : <https://console.groq.com/keys>
- Google Gemini : <https://aistudio.google.com/apikey>
- Ollama (local, sans clé) : <https://ollama.com>

## 5. Utilisation

```bash
# 1. Construire la base documentaire (Wikipédia, sans clé)
python scripts/01_build_corpus.py

# 2. Découper + indexer (télécharge le modèle d'embeddings au 1er lancement)
python scripts/02_index.py

# 3. Exporter la visualisation du graphe
python scripts/03_visualize_graph.py

# 4. Discuter avec l'agent
python scripts/chat.py            # ajouter --trace pour voir le raisonnement

# 5. Lancer l'évaluation (20 questions + métriques + rapports)
python scripts/04_evaluate.py
```

## 6. Évaluation

`scripts/04_evaluate.py` exécute 10 questions simples et 10 questions complexes
(bilingues), puis mesure :

- **latence** bout-en-bout par question ;
- **pertinence des documents** récupérés (`hit@k`, `MRR`) ;
- **qualité de la réponse** notée par un LLM juge (fidélité, complétude, clarté).

Sorties : `reports/evaluation.csv` (détail) et `reports/evaluation.md` (synthèse).

## 7. Structure du projet

```
f1_agentic_rag/
├── scripts/            # points d'entrée numérotés (corpus → index → chat → éval)
├── src/
│   ├── config.py       # configuration centrale
│   ├── corpus/         # construction de la base documentaire
│   ├── ingestion/      # embeddings, chunking, vectorstore, retriever hybride
│   ├── llm/            # fabrique de modèles (Groq / Gemini / Ollama)
│   ├── tools/          # outils de l'agent
│   ├── graph/          # state, nœuds, arêtes, assemblage du graphe
│   └── evaluation/     # jeu de questions, métriques, juge, harnais
├── reports/            # résultats d'évaluation + figures
└── requirements.txt
```

## 8. Choix techniques notables

- **Embeddings via FastEmbed (ONNX)** et non PyTorch : PyTorch ne publie plus de
  roue pour macOS x86_64 / Python 3.13, ce qui rendait le projet ininstallable.
  ONNX est aussi plus léger et plus rapide sur CPU.
- **Récupération hybride + RRF** : la recherche dense seule échoue sur les
  sigles (DRS, ERS) et les noms propres ; BM25 les rattrape. La fusion par
  Reciprocal Rank Fusion combine les deux sans normaliser des scores hétérogènes.
- **Plafond de chunks par document** : le corpus est déséquilibré (la page
  « Formule 1 » pèse ~390 chunks) ; sans plafond, un document généraliste
  monopolise le top-k.
