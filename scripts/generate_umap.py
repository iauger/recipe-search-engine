# scripts/generate_umap.py

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import umap

from src.config import load_settings

"""
Legacy script from Phase 2. I'm going to use this embedding and projection as a visualization element in a Streamlit frontend.
"""


def generate_umap(
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> None:
    s = load_settings()

    print(f"Loading embedding bundle from:\n  {s.processed_embedding_path}")
    bundle = torch.load(s.processed_embedding_path, map_location="cpu", weights_only=False)

    raw_embeddings = bundle["embeddings"]          # (N, 128) tensor
    recipe_ids     = bundle["recipe_ids"]
    targets        = np.asarray(bundle["targets"], dtype=np.float32).reshape(-1)

    print(f"Loaded {len(recipe_ids):,} recipes  |  embedding shape: {raw_embeddings.shape}")

    embeddings = F.normalize(raw_embeddings.float(), p=2, dim=1).numpy()

    print(f"\nFitting UMAP  (n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"metric={metric}, random_state={random_state})")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=False,
    )

    projection = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)  # (N, 2)
    print(f"Projection shape: {projection.shape}")

    # save for persistence for downstream use in the Streamlit app
    proj_path = os.path.join(s.processed_dir, "final_residual_v2_umap_projection.npy")
    np.save(proj_path, projection)
    print(f"\nSaved projection to:\n  {proj_path}")

    reducer_path = os.path.join(s.processed_dir, "final_residual_v2_umap_reducer.pkl")
    with open(reducer_path, "wb") as f:
        pickle.dump(reducer, f)
    print(f"Saved fitted reducer to:\n  {reducer_path}")

    meta_path = os.path.join(s.processed_dir, "final_residual_v2_umap_meta.npz")
    np.savez(
        meta_path,
        recipe_ids=np.array([str(r) for r in recipe_ids]),
        targets=targets,
        projection=projection,
    )
    print(f"Saved companion metadata to:\n  {meta_path}")

    # debugging info
    print(f"\nProjection range:")
    print(f"  x: [{projection[:, 0].min():.3f}, {projection[:, 0].max():.3f}]")
    print(f"  y: [{projection[:, 1].min():.3f}, {projection[:, 1].max():.3f}]")
    print(f"\nTarget (rating) range:")
    print(f"  [{targets.min():.3f}, {targets.max():.3f}]  mean={targets.mean():.3f}")


if __name__ == "__main__":
    generate_umap()
