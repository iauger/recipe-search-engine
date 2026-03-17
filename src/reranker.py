# src/reranker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import numpy as np

from src.config import load_settings
from src.query_encoding import QueryFeatureProjector
from src.search import parse_user_intent, retrieve_candidates

@dataclass
class RerankedResult:
    """
    A structured representation of a search result after reranking.
    """
    recipe_id: str
    base_score: float
    semantic_score: float
    quality_score: float
    final_score: float
    source: Dict[str, Any]

class SemanticReranker:
    """
    Reranks candidate recipes based on a combination of:
    - Semantic similarity to the query
    - Quality scores derived from the predictions
    - Original Elasticsearch relevance score
    """

    def __init__(self, s: Any):
        self.s = s
        self.bundle = self.load_bundle(s.processed_embedding_path)
        self.recipe_ids = self.bundle["recipe_ids"]
        self.predictions = self.bundle["predictions"]
        self.embeddings = self.bundle["embeddings"]
        self.id_to_index = self.build_id_lookup(self.recipe_ids)
    
    def load_bundle(self, bundle_path: str) -> Dict[str, Any]:
        self.bundle = torch.load(bundle_path, map_location=torch.device("cpu"), weights_only=False)
        return self.bundle
    
    def build_id_lookup(self, recipe_ids: List[Any]) -> Dict[str, Any]:
        return {str(r_id): idx for idx, r_id in enumerate(recipe_ids)}
    
    def get_quality_score(self, recipe_id: str) -> float:
        idx = self.id_to_index.get(str(recipe_id))
        if idx is None:
            return 0.0
        
        pred = self.predictions[idx]
        
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
            
        return float(pred)
    
    def score_alignment(self, projected_query: Any, candidate_source: Dict[str, Any]) -> float:
        score = 0.0

        tags = candidate_source.get("tags_clean", [])
        if tags is None:
            tags = []
        if isinstance(tags, str):
            tags = [tags]

        tags_norm = {str(tag).strip().lower() for tag in tags}

        ingredients_text = str(candidate_source.get("ingredients_clean", "")).lower()
        name_text = str(candidate_source.get("name", "")).lower()
        minutes = candidate_source.get("minutes")

        tag_groups = [
            projected_query.dietary_tags,
            projected_query.cuisines,
            projected_query.methods,
            projected_query.occasions,
            projected_query.courses,
        ]

        for group in tag_groups:
            for value in group:
                value_norm = str(value).strip().lower()
                if value_norm in tags_norm:
                    score += 1.0

        for protein in projected_query.proteins:
            protein_norm = str(protein).strip().lower()
            if protein_norm in ingredients_text:
                score += 0.75
            elif protein_norm in name_text:
                score += 0.5

        if projected_query.target_minutes is not None and minutes is not None:
            try:
                minutes_val = float(minutes)
                distance = abs(minutes_val - projected_query.target_minutes)

                if distance <= 5:
                    score += 0.75
                elif distance <= 15:
                    score += 0.4
            except (TypeError, ValueError):
                pass

        lexical_tokens = projected_query.clean_text.lower().split()

        structured_tokens = set(
            projected_query.proteins
            + projected_query.dietary_tags
            + projected_query.cuisines
            + projected_query.methods
            + projected_query.occasions
            + projected_query.courses
        )

        leftover_tokens = [
            t.strip()
            for t in lexical_tokens
            if t.strip() and t.strip() not in structured_tokens
        ]
        leftover_match = False
        
        for token in leftover_tokens:
            token = token.strip()
            
            if token in name_text:
                score += 0.75
                leftover_match = True
            elif token in ingredients_text:
                score += 0.25
                leftover_match = True
        
        if leftover_tokens and not leftover_match:
            score -= 0.75

        return score
    
    def combine_scores(self, base_score: float, semantic_score: float, quality_score: float) -> float:
        semantic_score = 0.0 if semantic_score is None else semantic_score
        quality_score = 0.0 if quality_score is None else quality_score

        lex_retrieval = 1.0
        sem_alignment = 1.0
        rec_quality = 0.25

        return (
            base_score * lex_retrieval
            + semantic_score * sem_alignment
            + quality_score * rec_quality
        )
    
    def rerank(
        self,
        projected_query: Any,
        candidates: List[Dict[str, Any]],
    ) -> List[RerankedResult]:
        """
        Main entry point.

        Inputs:
        - projected_query from QueryFeatureProjector
        - candidates from Stage 1 Elasticsearch retrieval

        Output:
        - reranked candidate list
        """
        reranked: List[RerankedResult] = []

        for hit in candidates:
            recipe_id = str(hit["_id"])
            source = hit["_source"]
            base_score = float(hit.get("_score", 0.0))

            semantic_score = self.score_alignment(
                projected_query=projected_query,
                candidate_source=source,
            )

            quality_score = self.get_quality_score(recipe_id)

            final_score = self.combine_scores(
                base_score=base_score,
                semantic_score=semantic_score,
                quality_score=quality_score,
            )

            reranked.append(
                RerankedResult(
                    recipe_id=recipe_id,
                    base_score=base_score,
                    semantic_score=semantic_score,
                    quality_score=quality_score,
                    final_score=final_score,
                    source=source,
                )
            )

        reranked.sort(key=lambda x: x.final_score, reverse=True)
        return reranked

if __name__ == "__main__":
    s = load_settings()
    raw_query = "low carb italian beef skillet"
    intent = parse_user_intent(raw_query)
    candidates, intent = retrieve_candidates(s, raw_query)
    
    for i, hit in enumerate(candidates, start=1):
            source = hit["_source"]
            name = source.get("name", "").title()
            minutes = source.get("minutes")
            score = hit.get("_score")
            print(f"{i}. [{score:.2f}] {name} ({minutes}m)")
            
    projected_query = QueryFeatureProjector(s).project(raw_query, intent)
    results = SemanticReranker(s).rerank(projected_query, candidates)
    
    print("\nReranked Results:")
    for i, result in enumerate(results, start=1):
        name = result.source.get("name", "").title()
        minutes = result.source.get("minutes")
        score = result.final_score
        print(f"{i}. [{score:.2f}] {name} ({minutes}m)")