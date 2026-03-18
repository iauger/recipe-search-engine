# src/reranker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import numpy as np

from src.config import load_settings
from src.models import RecipeNet, HeadType, AblationType
from src.query_encoding import QueryFeatureProjector
from src.search import parse_user_intent, retrieve_candidates


# ---------------------------------------------------------------------------
# Constants: derived from RecipeDataset (Phase 2) and confirmed by the saved
# model weights shape (two_tower_meta_encoder.0.linear.weight: [64, 210]).
#
# column_mapping.json has 248 entries, but 4 are text columns (indices 0-3:
# ingredients_clean, steps_clean, tags_clean, description_clean) that
# RecipeDataset never includes in its numeric tensors.  The remaining 244
# columns split as:
#
#   meta_in  (210): 10 numeric + 200 OHE (cat_* / ing_*)
#   tag_in    (34): 17 pred_* + 17 intensity_*
#
# Within meta_in:
#   num_meta  (10): minutes, n_steps, n_ingredients, calories, fat, sugar,
#                   sodium, protein, saturated_fat, carbs
#   cat_meta (200): cat_* (100) + ing_* (100)
# ---------------------------------------------------------------------------

_TEXT_COL_INDICES = {0, 1, 2, 3}   # columns skipped when building input tensors

META_DIM   = 210  # confirmed by saved weights: two_tower_meta_encoder shape [64, 210]
TAG_DIM    =  34  # 17 pred_* + 17 intensity_*
NUM_META   =  10  # continuous numeric features
CAT_META   = 200  # OHE categorical features (cat_* + ing_*)
HIDDEN_DIM = 128  # must match hidden_dim used during Phase 2 training


@dataclass
class RerankedResult:
    """Structured representation of a search result after reranking."""
    recipe_id: str
    base_score: float
    alignment_score: float
    semantic_sim: float
    quality_score: float
    final_score: float
    source: Dict[str, Any]


class SemanticReranker:
    """
    Two-stage reranker that combines:

    1. Elasticsearch base score     – lexical relevance from Stage 1 retrieval
    2. Rule-based alignment score   – deterministic structured intent matching
    3. Embedding cosine similarity  – query projected into the RecipeNet latent
                                      space and compared against stored recipe
                                      embeddings from the Phase 2 bundle
    4. Quality score                – scalar rating prediction from RecipeNet

    The embedding similarity is the key addition over the prior rule-based
    reranker: rather than relying solely on keyword matching, the query is
    encoded into the same 128-D manifold learned during Phase 2 training,
    allowing the geometry of that space to surface semantically related recipes
    that may not share exact tag or ingredient tokens with the query.
    """

    def __init__(self, s: Any):
        self.s = s
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the Phase 2 embedding bundle (recipe_ids, predictions, embeddings)
        self.bundle = self._load_bundle(s.processed_embedding_path)
        self.recipe_ids   = self.bundle["recipe_ids"]
        self.predictions  = self.bundle["predictions"]

        # Pre-normalise stored embeddings once at init for fast cosine sim later
        raw_embeddings = self.bundle["embeddings"]           # (N, 128) tensor
        self.embeddings = F.normalize(raw_embeddings.float(), p=2, dim=1)  # unit vectors

        self.id_to_index = {str(r_id): idx for idx, r_id in enumerate(self.recipe_ids)}

        # Build the feature-index lookup used when constructing query tensors.
        # Mirrors the column_mapping.json layout but skips the 4 text columns
        # that are never part of the numeric model input.
        self.col_map: Dict[str, int] = s.column_mapping
        self._meta_cols, self._tag_cols = self._split_feature_columns()

        # Load the frozen RecipeNet for query encoding
        self.model = self._load_model(s.model_weights_path)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_bundle(self, path: str) -> Dict[str, Any]:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)

    def _load_model(self, weights_path: str) -> RecipeNet:
        """
        Reconstruct the RecipeNet (Residual V2) architecture and load the
        saved Phase 2 weights.  The model is frozen and set to eval mode –
        it is used solely as a feature encoder for query projection.
        """
        model = RecipeNet(
            meta_in=META_DIM,
            tag_in=TAG_DIM,
            hidden_dim=HIDDEN_DIM,
            head_type=HeadType.RESIDUAL_V2,
            num_meta=NUM_META,
            cat_meta=CAT_META,
        ).to(self.device)

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

        # The weights were saved when the meta encoder attribute was named
        # `legacy_meta_encoder`. It was later renamed to `default_meta_encoder`
        # in models.py. Remap the keys so load_state_dict doesn't reject them.
        remapped_state_dict = {
            k.replace('legacy_meta_encoder', 'default_meta_encoder'): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(remapped_state_dict)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        print(f"RecipeNet loaded from {weights_path} (frozen, eval mode).")
        return model

    def _split_feature_columns(self):
        """
        Partition column_mapping into meta and tag index lists, preserving
        the positional order expected by RecipeNet's dual-encoder inputs.

        Text columns (indices 0-3) are skipped – they have no numeric
        representation in the model input tensors.

        Returns
        -------
        meta_cols : list[(col_name, tensor_position)]
            214 meta features in ascending tensor-position order.
        tag_cols  : list[(col_name, tensor_position)]
            34 tag features in ascending tensor-position order.
        """
        meta_entries = []
        tag_entries  = []

        for col, raw_idx in self.col_map.items():
            if raw_idx in _TEXT_COL_INDICES:
                continue  # text column – no numeric tensor slot

            if col.startswith("pred_") or col.startswith("intensity_"):
                tag_entries.append((col, raw_idx))
            else:
                meta_entries.append((col, raw_idx))

        # Sort by raw_idx so tensor positions are consistent with training
        meta_entries.sort(key=lambda x: x[1])
        tag_entries.sort(key=lambda x: x[1])

        # Re-index to contiguous 0-based positions within each sub-tensor
        meta_cols = [(col, i) for i, (col, _) in enumerate(meta_entries)]
        tag_cols  = [(col, i) for i, (col, _) in enumerate(tag_entries)]

        return meta_cols, tag_cols

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _build_query_tensors(
        self,
        projected_query: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a ProjectedQuery into (meta_tensor, tag_tensor) matching the
        exact shape expected by RecipeNet's dual encoders.

        The ProjectedQuery already carries meta_vector and tag_vector aligned
        to the column_mapping schema via QueryFeatureProjector.  We simply
        re-map those sparse activation vectors into the contiguous sub-tensors,
        skipping the 4 text-column slots.

        Numeric continuous features (minutes, calories, etc.) remain zero
        because they cannot be derived from free text.  This is an accepted
        limitation: the query will land near the origin of the numeric subspace
        but the categorical / tag activations still guide it toward the correct
        neighbourhood of the manifold.
        """
        meta_tensor = torch.zeros(1, META_DIM, dtype=torch.float32)
        tag_tensor  = torch.zeros(1, TAG_DIM,  dtype=torch.float32)

        # meta_vector from the projector is already indexed by col_map position
        for col, tensor_pos in self._meta_cols:
            raw_idx = self.col_map.get(col)
            if raw_idx is None:
                continue
            # Shift raw_idx to account for the 4 skipped text columns
            # The projector's meta_vector excludes text cols, so use tensor_pos
            val = projected_query.meta_vector[tensor_pos] if tensor_pos < len(projected_query.meta_vector) else 0.0
            meta_tensor[0, tensor_pos] = float(val)

        for col, tensor_pos in self._tag_cols:
            raw_idx = self.col_map.get(col)
            if raw_idx is None:
                continue
            val = projected_query.tag_vector[tensor_pos] if tensor_pos < len(projected_query.tag_vector) else 0.0
            tag_tensor[0, tensor_pos] = float(val)

        return meta_tensor.to(self.device), tag_tensor.to(self.device)

    def encode_query(self, projected_query: Any) -> torch.Tensor:
        """
        Project a structured query into the 128-D RecipeNet latent space.

        Passes the sparse query feature vectors through the frozen RecipeNet
        encoder and returns a unit-normalised 128-D embedding that can be
        directly compared to stored recipe embeddings via dot product.

        Returns
        -------
        query_embedding : torch.Tensor, shape (128,)
            Unit-normalised query embedding on CPU.
        """
        meta_tensor, tag_tensor = self._build_query_tensors(projected_query)

        with torch.no_grad():
            _, embedding = self.model(
                meta_tensor,
                tag_tensor,
                return_embeddings=True,
                ablation=AblationType.ALL_FEATURES,
            )

        # Normalise to unit vector so cosine sim reduces to dot product
        query_embedding = F.normalize(embedding.cpu().float(), p=2, dim=1).squeeze(0)  # (128,)
        return query_embedding

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------

    def compute_semantic_similarity(
        self,
        query_embedding: torch.Tensor,
        recipe_id: str,
    ) -> float:
        """
        Cosine similarity between the query embedding and the stored recipe
        embedding from the Phase 2 bundle.

        Both vectors are pre-normalised, so this reduces to a dot product.
        Returns 0.0 if the recipe_id is not found in the bundle.

        Returns a value in [-1, 1]; in practice nearly always in [0, 1]
        because ReLU activations keep embeddings non-negative.
        """
        idx = self.id_to_index.get(str(recipe_id))
        if idx is None:
            return 0.0

        recipe_embedding = self.embeddings[idx]  # (128,) unit vector
        sim = torch.dot(query_embedding, recipe_embedding).item()
        return float(sim)

    def get_quality_score(self, recipe_id: str) -> float:
        """Scalar rating prediction from the Phase 2 bundle (range ~1–5)."""
        idx = self.id_to_index.get(str(recipe_id))
        if idx is None:
            return 0.0

        pred = self.predictions[idx]
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        return float(pred)

    def score_alignment(
        self,
        projected_query: Any,
        candidate_source: Dict[str, Any],
    ) -> float:
        """
        Rule-based structured intent alignment score.

        Rewards candidates that match the structured intent extracted from the
        query (cuisine, method, course, dietary tags, proteins, timing) against
        the recipe's indexed fields.  Provides a reliable signal for exact
        constraint satisfaction that pure embedding similarity cannot guarantee.
        """
        score = 0.0

        tags = candidate_source.get("tags_clean", [])
        if tags is None:
            tags = []
        if isinstance(tags, str):
            tags = [tags]
        tags_norm = {str(t).strip().lower() for t in tags}

        ingredients_text = str(candidate_source.get("ingredients_clean", "")).lower()
        name_text = str(candidate_source.get("name", "")).lower()
        minutes = candidate_source.get("minutes")

        # High-value structured matches: cuisine, method, course, occasion,
        # dish type, and dietary constraints
        high_value_groups = [
            projected_query.cuisines,
            projected_query.methods,
            projected_query.occasions,
            projected_query.courses,
            projected_query.dish_type,
            projected_query.dietary_tags,
        ]
        for group in high_value_groups:
            for value in group:
                value_norm = str(value).strip().lower()
                if value_norm in tags_norm:
                    score += 1.0
                elif value_norm in name_text:
                    score += 0.75

        # Taste signal (softer – taste words are noisy in tags)
        for value in projected_query.taste:
            value_norm = str(value).strip().lower()
            if value_norm in tags_norm:
                score += 0.25
            elif value_norm in name_text:
                score += 0.25
            elif value_norm in ingredients_text:
                score += 0.10

        # Protein matching (ingredients and name)
        for protein in projected_query.proteins:
            protein_norm = str(protein).strip().lower()
            if protein_norm in ingredients_text:
                score += 0.75
            elif protein_norm in name_text:
                score += 0.50

        # Soft time-proximity reward
        if projected_query.target_minutes is not None and minutes is not None:
            try:
                distance = abs(float(minutes) - projected_query.target_minutes)
                if distance <= 5:
                    score += 0.75
                elif distance <= 15:
                    score += 0.4
            except (TypeError, ValueError):
                pass

        # Leftover lexical tokens not covered by structured intent
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
            for t in projected_query.clean_text.lower().split()
            if t.strip() and t.strip() not in structured_tokens
        ]
        leftover_match = False
        for token in leftover_tokens:
            if token in name_text:
                score += 0.75
                leftover_match = True
            elif token in ingredients_text:
                score += 0.25
                leftover_match = True
        if leftover_tokens and not leftover_match:
            score -= 0.75

        return score

    @staticmethod
    def get_weight_profile(projected_query: Any) -> Dict[str, Any]:
        """
        Stratified weight profiles based on structured intent richness.

        Intent richness is measured by counting the number of high-signal
        structured intent groups that fired during query parsing.  Taste and
        dish_type are intentionally excluded — they are softer signals that
        don't strongly constrain the result space and would inflate the count
        for queries that are still essentially exploratory.

        Three tiers:

        HIGH intent (3+ signals)
            Alignment dominates.  The parser has captured explicit constraints
            (cuisine, protein, dietary restriction, etc.) and rule-based
            matching is the most reliable signal.  Semantic similarity is
            compressed in this regime — all retrieved candidates look similar
            in the embedding space — so its weight is minimised.

        MEDIUM intent (1–2 signals)
            Balanced weighting.  Alignment provides partial structure; the
            embedding catches semantic outliers the rules miss (e.g. a recipe
            that is thematically right but lacks the exact tag token).

        LOW intent (0 signals)
            Semantic similarity carries the load.  Alignment fires weakly and
            uniformly across all candidates, so it is down-weighted to avoid
            rewarding noise.  Quality gets a larger boost because "give me
            something good" is a reasonable fallback for vague queries.
        """
        intent_signals = (
            len(projected_query.dietary_tags)
            + len(projected_query.cuisines)
            + len(projected_query.courses)
            + len(projected_query.methods)
            + len(projected_query.occasions)
            + len(projected_query.proteins)
        )

        if intent_signals >= 3:
            return {"tier": "high",   "lex": 1.0, "alignment": 1.5, "semantic": 0.1,  "quality": 0.25}
        elif intent_signals >= 1:
            return {"tier": "medium", "lex": 1.0, "alignment": 1.0, "semantic": 0.5,  "quality": 0.25}
        else:
            return {"tier": "low",    "lex": 0.75, "alignment": 0.25, "semantic": 1.5, "quality": 0.5}

    def combine_scores(
        self,
        base_score: float,
        alignment_score: float,
        semantic_sim: float,
        quality_score: float,
        weights: Dict[str, float],
    ) -> float:
        """
        Weighted combination of the four scoring signals using the
        stratified weight profile selected by get_weight_profile().

        semantic_sim arrives in [-1, 1]; clamped to [0, 1] to avoid
        penalising recipes that land in the opposite embedding hemisphere —
        an artifact of sparse query projection rather than true negative signal.
        """
        semantic_sim_clamped = max(0.0, semantic_sim)

        return (
            base_score             * weights["lex"]
            + alignment_score      * weights["alignment"]
            + semantic_sim_clamped * weights["semantic"]
            + quality_score        * weights["quality"]
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def rerank(
        self,
        projected_query: Any,
        candidates: List[Dict[str, Any]],
        mode_weights: Dict[str, Any] | None = None,
    ) -> List[RerankedResult]:
        """
        Rerank a list of Stage 1 Elasticsearch candidates.
 
        Pipeline per candidate:
          1. score_alignment   – rule-based structured intent check
          2. encode_query      – project query into 128-D latent space (once)
          3. compute_semantic_similarity – cosine sim vs stored embedding
          4. get_quality_score – scalar predicted rating
          5. combine_scores    – weighted sum → final_score
          6. sort descending by final_score
 
        Parameters
        ----------
        projected_query : ProjectedQuery
            Output of QueryFeatureProjector.project().
        candidates : list of ES hit dicts
            Raw hits from retrieve_candidates().
        mode_weights : dict | None
            When provided by the engine (e.g. for LEXICAL, SEMANTIC, QUALITY
            modes), bypasses get_weight_profile() entirely and uses the
            supplied fixed weights instead.  When None, stratified weights
            are resolved from the query's intent richness as normal — this
            is the default behaviour for HYBRID and ABLATION_NO_SEM.
 
        Returns
        -------
        List[RerankedResult] sorted by final_score descending.
        """
        # Use caller-supplied fixed weights (engine modes) or resolve
        # stratified weights from query intent richness (hybrid/ablation).
        weights = mode_weights if mode_weights is not None else self.get_weight_profile(projected_query)
 
        # Encode the query once – shared across all candidates
        query_embedding = self.encode_query(projected_query)
 
        reranked: List[RerankedResult] = []
 
        for hit in candidates:
            recipe_id  = str(hit["_id"])
            source     = hit["_source"]
            base_score = float(hit.get("_score", 0.0))
 
            alignment_score = self.score_alignment(projected_query, source)
            semantic_sim    = self.compute_semantic_similarity(query_embedding, recipe_id)
            quality_score   = self.get_quality_score(recipe_id)
 
            final_score = self.combine_scores(
                base_score=base_score,
                alignment_score=alignment_score,
                semantic_sim=semantic_sim,
                quality_score=quality_score,
                weights=weights,
            )
 
            reranked.append(
                RerankedResult(
                    recipe_id=recipe_id,
                    base_score=base_score,
                    alignment_score=alignment_score,
                    semantic_sim=semantic_sim,
                    quality_score=quality_score,
                    final_score=final_score,
                    source=source,
                )
            )
 
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        return reranked


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    s = load_settings()

    smoke_queries = [
        "low carb italian beef skillet",   # high intent
        "quick chicken dish",              # medium intent
        "something cozy and warming",      # low intent
    ]

    reranker = SemanticReranker(s)
    projector = QueryFeatureProjector(s)

    for raw_query in smoke_queries:
        candidates, intent = retrieve_candidates(s, raw_query)
        projected_query = projector.project(raw_query, intent)
        weights = reranker.get_weight_profile(projected_query)
        results = reranker.rerank(projected_query, candidates)

        print(f"\n{'='*65}")
        print(f"QUERY : {raw_query}")
        print(f"TIER  : {weights['tier']}  "
              f"(lex={weights['lex']}  align={weights['alignment']}  "
              f"sem={weights['semantic']}  quality={weights['quality']})")
        print(f"{'='*65}")

        print("  Stage 1 – Elasticsearch baseline:")
        for i, hit in enumerate(candidates, start=1):
            source = hit["_source"]
            name   = source.get("name", "").title()
            mins   = source.get("minutes")
            score  = hit.get("_score", 0.0)
            print(f"    {i}. [{score:.2f}] {name} ({mins}m)")

        print("  Stage 2 – Reranked:")
        for i, r in enumerate(results, start=1):
            name = r.source.get("name", "").title()
            mins = r.source.get("minutes")
            print(
                f"    {i}. [{r.final_score:.3f}] {name} ({mins}m) "
                f"| align={r.alignment_score:.2f}  "
                f"sim={r.semantic_sim:.3f}  "
                f"quality={r.quality_score:.2f}"
            )