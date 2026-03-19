# src/indexer.py
import pandas as pd
import numpy as np
import torch
from elasticsearch import helpers
from src.config import load_settings

# Adjust for Elasticsearch 8.12 np.float_ requirement
if not hasattr(np, 'float_'):
    np.float_ = np.float64 # type: ignore

def get_tags(df: pd.DataFrame) -> list[str]:
    """Extract base tag names from pred_* columns."""
    return [col.removeprefix("pred_") for col in df.columns if col.startswith("pred_")]

def create_index(s):
    """Deletes the old index if it exists and creates a new one with strict mappings."""
    
    print("Testing connection to Elasticsearch...")
    try:
        # GET request to verify the client can actually talk to the server
        info = s.es_client.info()
        print(f"Successfully connected to Elasticsearch v{info['version']['number']}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Check if a VPN or proxy is blocking the connection.")
        return

    # ignore_unavailable=True safely deletes the index if it exists, and ignores if it doesn't.
    print(f"Resetting index: {s.index_name}")
    s.es_client.indices.delete(index=s.index_name, ignore_unavailable=True)

    # The blueprint for the search engine
    properties = {
        "name": {"type": "text", "analyzer": "english"},
        "description_clean": {"type": "text", "analyzer": "english"},
        "steps_clean": {"type": "text", "analyzer": "english"},
        "ingredients_clean": {"type": "text", "analyzer": "standard"},
        "tags_clean": {"type": "keyword"},
        "minutes": {"type": "float"},
        "n_steps": {"type": "integer"},
        "n_ingredients": {"type": "integer"},
        "bayesian_rating": {"type": "float"},
        "review_count": {"type": "integer"}
    }
    
    # Dynamically add the 17 qualitative tags
    tags = get_tags(pd.read_parquet(s.processed_recipes_path))

    for tag in tags:
        properties[f"pred_{tag}"] = {"type": "boolean"}
        properties[f"intensity_{tag}"] = {"type": "float"}

    # Pass 'mappings' directly instead of a 'body' dictionary
    s.es_client.indices.create(index=s.index_name, mappings={"properties": properties})
    print(f"Created index '{s.index_name}' with hybrid search mappings.")

def generate_documents(df, bundle, s):
    """Yields documents in the format required by Elasticsearch bulk API."""
    
    for idx, row in df.iterrows():
        r_id = str(row['recipe_id'])
            
        doc = row.to_dict()
                
        yield {
            "_index": s.index_name,
            "_id": r_id,
            "_source": doc
        }

# Main orchestration function to run the ingestion process
def run_ingestion(s):
    print("Loading search data into memory.")
    df = pd.read_parquet(s.processed_recipes_path)
    bundle = torch.load(s.processed_embedding_path, weights_only=False) 
    
    create_index(s)
    
    print(f"Starting bulk ingestion of {len(df)} recipes.")
    success, failed = helpers.bulk(
        s.es_client, 
        generate_documents(df, bundle, s),
        chunk_size=500,
        stats_only=True
    )
    print(f"Successfully indexed {success} documents.")
    if failed:
        print(f"Warning: {failed} documents failed to index.")

if __name__ == "__main__":
    s = load_settings()
    run_ingestion(s)