# src/config.py

from elasticsearch import Elasticsearch
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import logging
import json

logger = logging.getLogger(__name__)


# Helper functions for path resolution
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "requirements.txt").exists(): # crude heuristic for repo root
            return p
    # fallback: current behavior
    return here.parents[1]


def resolve_path(path: str | None, default: str) -> str:
    root = repo_root()

    if not path:
        path = default

    path = path.strip()
    p = Path(path)

    if p.is_absolute():
        return str(p)

    return str((root / p).resolve())

@dataclass
class Settings:
    
    env: str
    es_client: Elasticsearch | None
    index_name: str
    
    # Directory paths
    data_dir: str
    raw_dir: str
    processed_dir: str
    src_dir: str
    
    # Processed data file paths
    processed_recipes_path: str
    processed_embedding_path: str
    column_mapping: dict[str, int]
    
def validate_settings(s: Settings) -> None:
    require_raw = os.getenv("REQUIRE_RAW_INPUTS", "0").strip() == "1"

    if s.env == "local" and require_raw:
        for p in [s.processed_recipes_path, s.processed_embedding_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing required file: {p}")

    # Cleaned up duplicates and added src_dir
    for d in [s.data_dir, s.raw_dir, s.processed_dir, s.src_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    load_dotenv(override=False)

    env = os.getenv("ENV", "local").strip().lower()
    
    es_url = os.getenv("ES_CLIENT", "http://localhost:9200").strip().strip('"').strip("'")
    es_client = Elasticsearch(es_url) if es_url else None
    if es_client is not None:
        try:
            es_client.info()
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Elasticsearch at {es_url}. "
                f"Check that Docker is running, the container is healthy, and the URL is correct."
            ) from e
            
    index_name = os.getenv("INDEX_NAME", "recipes_v1").strip().strip('"').strip("'").lower().replace(" ", "_")
    
    src_dir = resolve_path(os.getenv("SRC_DIR"), "./src")
    data_dir = resolve_path(os.getenv("DATA_DIR"), "./data")
    raw_dir = resolve_path(os.getenv("RAW_DIR"), "./data/raw")
    processed_dir = resolve_path(os.getenv("PROCESSED_DIR"), "./data/processed")

    processed_recipes_path = resolve_path(os.getenv("PROCESSED_RECIPES_PATH"), "./data/processed/PROCESSED_search_recipes.parquet")
    processed_embedding_path = resolve_path(os.getenv("PROCESSED_EMBEDDING_PATH"), "./data/processed/final_residual_v2_embeddings.pt")
    column_mapping_path = resolve_path(os.getenv("COLUMN_MAPPING"), "./data/processed/column_mapping.json")
    
    with open(column_mapping_path, 'r') as f:
        column_mapping = json.load(f)
    
    s = Settings(
        env=env,
        es_client=es_client,
        index_name=index_name,
        src_dir=src_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        processed_recipes_path=processed_recipes_path,
        processed_embedding_path=processed_embedding_path,
        column_mapping=column_mapping
    )
    validate_settings(s)
    return s