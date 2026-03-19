# src/engine.py

"""
Search engine orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import copy
import sys
import io
if(isinstance(sys.stdout, io.TextIOWrapper)):
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import load_settings
from src.query_encoding import QueryFeatureProjector
from src.reranker import RerankedResult, SemanticReranker
from src.search import parse_user_intent, retrieve_candidates


# Search mode definitions
class SearchMode(Enum):
    LEXICAL         = "lexical"
    SEMANTIC        = "semantic"
    QUALITY         = "quality"
    ABLATION_NO_SEM = "ablation_no_sem"
    HYBRID          = "hybrid"
    
"""Search Modes
------------
LEXICAL
    Elasticsearch BM25 only.  No reranking signal applied — raw Stage 1
    ranking is returned as-is.  Serves as the baseline for all comparisons.

SEMANTIC
    Embedding cosine similarity + quality score only.  Alignment is zeroed
    out, so structured rule matching plays no role.  Isolates the pure
    latent-space signal from the Phase 2 RecipeNet embeddings.

QUALITY
    Quality score (predicted Bayesian rating) only.  No lexical, alignment,
    or semantic signal.  Surfaces what the Phase 2 model considers the best
    recipes regardless of query relevance.  
    
ABLATION_NO_SEM
    Full hybrid pipeline minus the embedding similarity term.  Uses
    stratified weights based on query intent richness.  Direct comparison
    with HYBRID isolates the contribution of the embedding layer.

HYBRID
    Full pipeline: lexical + alignment + semantic + quality, with stratified
    weights determined by query intent richness.
"""

# Fixed weight profiles for non-stratified modes.
_FIXED_WEIGHTS: Dict[SearchMode, Dict[str, Any]] = {
    SearchMode.LEXICAL: {
        "tier": "lexical",
        "lex": 1.0,
        "alignment": 0.0,
        "semantic": 0.0,
        "quality": 0.0,
    },
    SearchMode.SEMANTIC: {
        "tier": "semantic",
        "lex": 0.0,
        "alignment": 0.0,
        "semantic": 1.0,
        "quality": 0.5,
    },
    SearchMode.QUALITY: {
        "tier": "quality",
        "lex": 0.0,
        "alignment": 0.0,
        "semantic": 0.0,
        "quality": 1.0,
    },
}

# Modes that use stratified intent-based weights rather than fixed profiles
_STRATIFIED_MODES = {SearchMode.HYBRID, SearchMode.ABLATION_NO_SEM}

# For ABLATION_NO_SEM the stratified weights are used but semantic is zeroed.
_ABLATION_SEMANTIC_ZERO = {SearchMode.ABLATION_NO_SEM}

# Results

@dataclass
class SearchResult:
    """
    Complete result bundle for a single query/mode execution.
    """
    query: str
    mode: SearchMode
    tier: str                           # intent tier label (or mode name for fixed modes)
    weights: Dict[str, Any]             # weights actually applied
    intent: Dict[str, Any]              # parsed query intent
    candidates: List[Dict[str, Any]]    # raw Stage 1 ES hits
    results: List[RerankedResult]       # reranked / ordered results
    query_embedding: Optional[Any] = field(default=None, repr=False)

# Engine
class SearchEngine:
    """
    Unified search interface over all five search modes.
    """

    def __init__(self, s: Any):
        self.s = s
        self.projector = QueryFeatureProjector(s)
        self.reranker  = SemanticReranker(s)

    # Public interface
    def run(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        return_query_embedding: bool = False,
    ) -> SearchResult:
        """
        Execute a search query in the specified mode.

        Parameters
        ----------
        query : str
            Raw user query string.
        mode : SearchMode
            Which search pipeline to use.
        top_k : int
            Number of candidates to retrieve from Elasticsearch.
        return_query_embedding : bool
            If True, the 128-D query embedding is attached to the result for
            downstream latent space visualisation.  Adds one forward pass
            overhead; leave False for evaluation loops.

        Returns
        -------
        SearchResult
        """
        # Stage 1 — lexical retrieval (always runs regardless of mode)
        candidates, intent = retrieve_candidates(self.s, query, top_k=top_k)

        # Project query into feature space
        projected_query = self.projector.project(query, intent)

        # Resolve weights for this mode
        weights, tier = self._resolve_weights(projected_query, mode)

        # Stage 2 — rerank (or pass through for LEXICAL)
        if mode == SearchMode.LEXICAL:
            # Wrap raw ES hits as RerankedResult objects for a uniform interface
            results = self._wrap_lexical_results(candidates)
        else:
            results = self.reranker.rerank(
                projected_query=projected_query,
                candidates=candidates,
                mode_weights=weights,
            )

        # Optionally attach query embedding for visualisation
        query_embedding = None
        if return_query_embedding and mode != SearchMode.LEXICAL:
            query_embedding = self.reranker.encode_query(projected_query)

        return SearchResult(
            query=query,
            mode=mode,
            tier=tier,
            weights=weights,
            intent=intent,
            candidates=candidates,
            results=results,
            query_embedding=query_embedding,
        )

    def run_all_modes(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict[SearchMode, SearchResult]:
        """
        Run a query through all five modes in a single call.

        Retrieves Stage 1 candidates once and shares them across all modes to ensure a fair comparison.
        """
        # Retrieve once, share across all modes. Candidates are copied per-mode to prevent any in-place mutation
        candidates, intent = retrieve_candidates(self.s, query, top_k=top_k)
        projected_query = self.projector.project(query, intent)

        results: Dict[SearchMode, SearchResult] = {}

        for mode in SearchMode:
            weights, tier = self._resolve_weights(projected_query, mode)
            mode_candidates = copy.deepcopy(candidates)

            if mode == SearchMode.LEXICAL:
                mode_results = self._wrap_lexical_results(mode_candidates)
            else:
                mode_results = self.reranker.rerank(
                    projected_query=projected_query,
                    candidates=mode_candidates,
                    mode_weights=weights,
                )

            results[mode] = SearchResult(
                query=query,
                mode=mode,
                tier=tier,
                weights=weights,
                intent=intent,
                candidates=candidates,
                results=mode_results,
            )

        return results

    def _resolve_weights(
        self,
        projected_query: Any,
        mode: SearchMode,
    ) -> tuple[Dict[str, Any], str]:
        """
        Return (weights_dict, tier_label) for the given mode.
        """
        if mode in _STRATIFIED_MODES:
            weights = dict(self.reranker.get_weight_profile(projected_query))
            tier = weights["tier"]

            if mode in _ABLATION_SEMANTIC_ZERO:
                weights["semantic"] = 0.0
                weights["tier"] = f"{tier}_no_sem"

            return weights, weights["tier"]

        weights = dict(_FIXED_WEIGHTS[mode])
        return weights, weights["tier"]

    @staticmethod
    def _wrap_lexical_results(
        candidates: List[Dict[str, Any]],
    ) -> List[RerankedResult]:
        """
        Wrap raw ES hits as RerankedResult objects.
        """
        return [
            RerankedResult(
                recipe_id=str(hit["_id"]),
                base_score=float(hit.get("_score", 0.0)),
                alignment_score=0.0,
                semantic_sim=0.0,
                quality_score=0.0,
                final_score=float(hit.get("_score", 0.0)),
                source=hit["_source"],
            )
            for hit in candidates
        ]

# Testing and demonstration with test queries
if __name__ == "__main__":
    s = load_settings()
    engine = SearchEngine(s)

    smoke_queries = [
        "low carb italian beef skillet",   # high intent
        "quick chicken dish",              # medium intent
        "something cozy and warming",      # low intent
    ]

    for query in smoke_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        all_results = engine.run_all_modes(query, top_k=5)

        for mode, result in all_results.items():
            w = result.weights
            print(f"\n  [{mode.value.upper()}]  tier={result.tier}  "
                  f"lex={w['lex']}  align={w['alignment']}  "
                  f"sem={w['semantic']}  quality={w['quality']}")

            for i, r in enumerate(result.results, start=1):
                name = r.source.get("name", "").title()
                mins = r.source.get("minutes")
                print(
                    f"    {i}. [{r.final_score:.3f}] {name} ({mins}m)"
                    f"  align={r.alignment_score:.2f}"
                    f"  sim={r.semantic_sim:.3f}"
                    f"  quality={r.quality_score:.2f}",
                    flush=True,
                )