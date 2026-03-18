# src/evaluate.py
import numpy as np
from src.config import load_settings
from src.search import parse_user_intent, retrieve_candidates
from src.query_encoding import QueryFeatureProjector
from src.reranker import SemanticReranker

TEST_QUERIES = [
    "easy chicken dinner under 45 mins",
    "vegan thanksgiving sides",
    "low carb italian beef skillet",
    "quick healthy breakfast under 15 mins",
    "spicy pork tacos"
]

def score_result(result, intent) -> float:
    """
    Rule-based relevance scorer for evaluation.

    Designed to compare:
    - baseline Elasticsearch hits
    - reranked RerankedResult objects

    Score components:
    1. Hard constraints: zero out if violated
    2. Structured intent match bonuses
    3. Leftover lexical intent bonus / penalty
    """

    # Support both ES hits and RerankedResult objects
    source = result["_source"] if isinstance(result, dict) else result.source

    score = 0.0

    tags = source.get("tags_clean", [])
    if tags is None:
        tags = []
    if isinstance(tags, str):
        tags = [tags]

    tags_norm = {str(tag).strip().lower() for tag in tags}
    name_text = str(source.get("name", "")).lower()
    ingredients_text = str(source.get("ingredients_clean", "")).lower()
    minutes = source.get("minutes")

    # -------------------------------------------------
    # 1. Hard constraints
    # -------------------------------------------------
    max_minutes = intent.get("max_minutes")
    if max_minutes is not None and minutes is not None:
        try:
            if float(minutes) > float(max_minutes):
                return 0.0
        except (TypeError, ValueError):
            return 0.0

    for diet in intent.get("dietary_tags", []):
        if str(diet).strip().lower() not in tags_norm:
            return 0.0

    # -------------------------------------------------
    # 2. Structured match bonuses
    # -------------------------------------------------
    structured_groups = [
        ("cuisines", 1.0),
        ("methods", 1.0),
        ("occasions", 1.0),
        ("courses", 1.0),
    ]

    for group_name, weight in structured_groups:
        for value in intent.get(group_name, []):
            if str(value).strip().lower() in tags_norm:
                score += weight

    # proteins often live in ingredients/title rather than tags
    for protein in intent.get("proteins", []):
        protein_norm = str(protein).strip().lower()
        if protein_norm in name_text:
            score += 1.0
        elif protein_norm in ingredients_text:
            score += 0.75

    # Soft reward for target-time proximity
    target_minutes = intent.get("target_minutes")
    if target_minutes is not None and minutes is not None:
        try:
            distance = abs(float(minutes) - float(target_minutes))
            if distance <= 5:
                score += 0.75
            elif distance <= 15:
                score += 0.4
        except (TypeError, ValueError):
            pass

    # -------------------------------------------------
    # 3. Leftover lexical intent
    # -------------------------------------------------
    clean_text = str(intent.get("clean_text", "")).lower()
    lexical_tokens = clean_text.split()

    structured_tokens = set(
        [str(x).strip().lower() for x in intent.get("proteins", [])]
        + [str(x).strip().lower() for x in intent.get("dietary_tags", [])]
        + [str(x).strip().lower() for x in intent.get("cuisines", [])]
        + [str(x).strip().lower() for x in intent.get("methods", [])]
        + [str(x).strip().lower() for x in intent.get("occasions", [])]
        + [str(x).strip().lower() for x in intent.get("courses", [])]
    )

    leftover_tokens = [
        t.strip()
        for t in lexical_tokens
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

    return max(score, 0.0)

def evaluate_query(query: str, s, projector, reranker) -> dict:
    intent = parse_user_intent(query)
    baseline_candidates, intent = retrieve_candidates(s, query)
    projected_query = projector.project(query, intent)
    reranked_results = reranker.rerank(projected_query, baseline_candidates)

    baseline_top5 = baseline_candidates[:5]
    reranked_top5 = reranked_results[:5]

    baseline_scores = [score_result(hit, intent) for hit in baseline_top5]
    reranked_scores = [score_result(result, intent) for result in reranked_top5]

    baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.0
    reranked_mean = float(np.mean(reranked_scores)) if reranked_scores else 0.0

    return {
        "query": query,
        "intent": intent,
        "baseline_results": baseline_top5,
        "reranked_results": reranked_top5,
        "baseline_scores": baseline_scores,
        "reranked_scores": reranked_scores,
        "baseline_mean": baseline_mean,
        "reranked_mean": reranked_mean,
        "delta": reranked_mean - baseline_mean,
    }
    
def evaluate_engine(s):
    projector = QueryFeatureProjector(s)
    reranker = SemanticReranker(s)

    query_reports = [
        evaluate_query(query, s, projector, reranker)
        for query in TEST_QUERIES
    ]

    baseline_means = [r["baseline_mean"] for r in query_reports]
    reranked_means = [r["reranked_mean"] for r in query_reports]
    intents = [r["intent"] for r in query_reports]

    summary = {
        "intent": intents,
        "mean_baseline_score_at_5": float(np.mean(baseline_means)) if baseline_means else 0.0,
        "mean_reranked_score_at_5": float(np.mean(reranked_means)) if reranked_means else 0.0,
        "mean_delta": float(np.mean([r["delta"] for r in query_reports])) if query_reports else 0.0,
        "query_reports": query_reports,
    }

    return summary

if __name__ == "__main__":
    s = load_settings()
    summary = evaluate_engine(s)

    for report in summary["query_reports"]:
        print(f"\nQuery: {report['query']}")
        print(f"Intent: {report['intent']}")
        print(f"Baseline mean@5: {report['baseline_mean']:.2f}")
        print(f"Reranked mean@5: {report['reranked_mean']:.2f}")
        print(f"Delta: {report['delta']:.2f}")

        print("Baseline Top 5:")
        for i, (hit, score) in enumerate(zip(report["baseline_results"], report["baseline_scores"]), start=1):
            source = hit["_source"]
            name = source.get("name", "").title()
            minutes = source.get("minutes")
            base_score = hit.get("_score", 0.0)
            print(f"  {i}. [{base_score:.2f}] {name} ({minutes}m) - Eval: {score:.2f}")

        print("Reranked Top 5:")
        for i, (result, score) in enumerate(zip(report["reranked_results"], report["reranked_scores"]), start=1):
            name = result.source.get("name", "").title()
            minutes = result.source.get("minutes")
            final_score = result.final_score
            print(f"  {i}. [{final_score:.2f}] {name} ({minutes}m) - Eval: {score:.2f}")

    print("\nOverall Summary")
    print(f"Mean baseline score@5 : {summary['mean_baseline_score_at_5']:.2f}")
    print(f"Mean reranked score@5 : {summary['mean_reranked_score_at_5']:.2f}")
    print(f"Mean delta            : {summary['mean_delta']:.2f}")