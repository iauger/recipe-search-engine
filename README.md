# Recipe Information Retrieval System
**Author:** Ian Auger  
**Course:** INFO 624 — Information Retrieval  

---

## Overview

A two-stage recipe search engine built on the Food.com corpus (~230,000 recipes, 1.1M reviews). The system combines classical BM25 lexical retrieval with rule-based query alignment and neural embedding similarity derived from a residual deep learning model trained on review-derived quality signals.

This project is the third phase of a multi-course research arc:

| Phase | Course | Contribution |
|---|---|---|
| 1 | DSCI 632 | Gold-labeled review dataset — 17-tag culinary taxonomy via Word2Vec centroid classifier |
| 2 | CS 615 | RecipeNet Residual V2 — 128D recipe embeddings + Bayesian quality predictions |
| 3 | INFO 624 | This IR system — Elasticsearch index + two-stage retrieval pipeline |

---

## System Architecture

```
User Query
    │
    ├── parse_user_intent()         # Extract structured intent (cuisine, protein, dietary, time)
    │       │
    │       └── Intent Tier         # High / Medium / Low → stratified weight profile
    │
    ├── Stage 1: Elasticsearch BM25
    │       └── Top-K candidates with hard filters + soft boosts
    │
    └── Stage 2: SemanticReranker
            ├── score_alignment()           # Rule-based structured intent matching
            ├── compute_semantic_similarity()# RecipeNet cosine similarity (128D)
            ├── get_quality_score()         # Predicted Bayesian rating
            └── combine_scores()            # Weighted sum → ranked results
```

---

## Prerequisites

- Python 3.12+
- Docker Desktop (for Elasticsearch)
- The Phase 2 model artifacts (see **Data & Model Files** below)

---

## Installation

**1. Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Start Elasticsearch:**
```bash
docker compose up -d
```
Elasticsearch will be available at `http://localhost:9200`. Allow 15–20 seconds for the container to become healthy before proceeding.

**3. Configure paths (optional):**

Copy `.env.example` to `.env` and update paths if your data files are in non-default locations. Default paths assume all processed files are in `data/processed/`.

---

## Data & Model Files

The following files are required and are **not included in this repository** due to size. They are outputs of Phase 1 and Phase 2 and should be placed in `data/processed/`:

| File | Source | Description |
|---|---|---|
| `PROCESSED_search_recipes.parquet` | Phase 2 preprocessing | Recipe documents for ES indexing, but processed as part of Phase 2 `preprocessing.py` |
| `final_residual_v2_embeddings.pt` | Phase 2 inference | RecipeNet embedding bundle |
| `best_model_residual_v2_all_features_mse.pth` | Phase 2 training | RecipeNet model weights |
| `column_mapping.json` | Phase 2 preprocessing | Feature schema for query projection |

---

## Quick Start

All system components are accessible through the CLI entrypoint:

```bash
python -m main
```

This launches an interactive menu:

```
1. Run data ingestion pipeline      # Create ES index and load recipes
2. Run search engine demo           # Query the system interactively
3. Demo intent parsing              # Inspect how a query is parsed and projected
4. Run evaluation                   # Five-mode ablation study (NDCG@5, P@1)
5. Exit
```

**Recommended first run sequence:**
```
Option 1: index the corpus (takes 1-3 minutes)
Option 2: run a search query
Option 3: inspect intent parsing
Option 4: run the full evaluation (pre-seeded with 10 example queries across each intent tier)
```

---

## Search Modes

The system supports five search modes selectable at query time:

| Mode | Description |
|---|---|
| `hybrid` | Full pipeline — BM25 + alignment + embedding + quality (default) |
| `lexical` | Elasticsearch BM25 only — no reranking |
| `semantic` | Embedding cosine similarity + quality only |
| `quality` | Predicted Bayesian rating only |
| `ablation_no_sem` | Hybrid minus embedding — isolates alignment contribution |

---

## Project Structure

```
├── main.py                     # CLI entrypoint
├── docker
│   └── docker-compose.yml      # Elasticsearch container
├── requirements.txt
├── notebooks/
│   └── project_report.ipynb    # Project report and evaluation
├── src/
│   ├── config.py               # Settings and path resolution
│   ├── indexer.py              # ES index creation and bulk ingestion
│   ├── search.py               # Stage 1 lexical retrieval + intent parsing
│   ├── query_encoding.py       # Query projection into feature space
│   ├── reranker.py             # Stage 2 semantic reranking
│   ├── engine.py               # Search mode orchestration
│   ├── evaluate.py             # Five-mode ablation evaluation
│   ├── models.py               # RecipeNet architecture (from Phase 2)
│   ├── layers.py               # Neural network building blocks (from Phase 2)
│   └── notebook.py             # Notebook visualization helpers
└── scripts/
    └── generate_umap.py        # One-time UMAP projection generation
```

---

## Evaluation

The evaluation framework runs all five search modes across a stratified set of 10 queries spanning high, medium, and low intent tiers. Primary metric is NDCG@5; secondary metric is Precision@1.

Run via the CLI (option 4) or directly:
```bash
python -m src.evaluate
```

Key findings are documented in `notebooks/project_report.ipynb` Section 5.

---

## Notebook

The project report notebook walks through all major deliverables:
Requires Elasticsearch to be running and the index to be populated for live code cells. Static outputs are preserved in the committed notebook.

---

## Troubleshooting

**Elasticsearch connection refused**  
Ensure Docker Desktop is running and the container is healthy:
```bash
docker compose up -d
docker ps  # confirm recipe_search_es is running
```

**Model weights not found**  
Confirm `best_model_residual_v2_all_features_mse.pth` is in `data/processed/` or set `MODEL_WEIGHTS_PATH` in your `.env`.

**Import errors on startup**  
Ensure all dependencies are installed: `pip install -r requirements.txt`
