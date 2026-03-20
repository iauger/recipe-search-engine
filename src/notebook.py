import base64
from IPython.display import Image, display
from src.search import parse_user_intent
from src.reranker import SemanticReranker
from src.query_encoding import QueryFeatureProjector
from src.config import load_settings
import json
from src.engine import SearchEngine, SearchMode

s = load_settings()
projector = QueryFeatureProjector(s)
reranker = SemanticReranker(s)
engine = SearchEngine(s)

# Full pipeline demonstration for one query per intent tier
def demo_scoring_functionality(queries: list[str], mode: SearchMode) -> None:
    for query in queries:
        result = engine.run(query, mode=mode, top_k=5)
        w = result.weights

        print(f"\n{'='*65}")
        print(f"QUERY : {query}")
        print(f"TIER  : {result.tier}  |  "
            f"lex={w['lex']}  align={w['alignment']}  "
            f"sem={w['semantic']}  quality={w['quality']}")
        print(f"{'-'*65}")

        for i, r in enumerate(result.results, start=1):
            name = r.source.get('name', '').title()
            mins = r.source.get('minutes')
            print(f"  {i}. [{r.final_score:.3f}] {name} ({mins}m)"
                f"  align={r.alignment_score:.2f}"
                f"  sim={r.semantic_sim:.3f}"
                f"  quality={r.quality_score:.2f}")
        

# Demonstrate the intent parser across all three use cases
def demo_intent_parsing(use_cases: list[tuple[str, str]]) -> None:
    for label, query in use_cases:
        intent = parse_user_intent(query)
        projected = projector.project(query, intent)
        weights = reranker.get_weight_profile(projected)

        print(f"\n{'='*60}")
        print(f"USE CASE: {label}")
        print(f"Query:    '{query}'")
        print(f"Tier:     {weights['tier']}")
        print(f"Weights:  lex={weights['lex']}  align={weights['alignment']}  "
            f"sem={weights['semantic']}  quality={weights['quality']}")
        print(f"Parsed intent:")
        for key in ['proteins','dietary_tags','cuisines','courses',
                    'methods','occasions','max_minutes','taste']:
            val = intent.get(key)
            if val:
                print(f"  {key:<20} {val}")

architecture_graph = """graph TD
    A[User Query] --> B[parse_user_intent]
    B --> C{Intent Tier}
    C -->|High| D[w_align=1.5, w_sem=0.1]
    C -->|Medium| E[w_align=1.0, w_sem=0.5]
    C -->|Low| F[w_align=0.25, w_sem=1.5]

    A --> G[Stage 1: Elasticsearch BM25]
    G --> H[Top-K Candidates]

    H --> I[Stage 2: SemanticReranker]
    D --> I
    E --> I
    F --> I

    I --> J[score_alignment\nRule-based intent match]
    I --> K[compute_semantic_similarity\nRecipeNet cosine sim]
    I --> L[get_quality_score\nBayesian rating prediction]
    I --> M[base_score\nBM25 lexical score]

    J --> N[combine_scores\nWeighted sum]
    K --> N
    L --> N
    M --> N

    N --> O[Ranked Results]
"""

def render_mermaid(graph_code: str = architecture_graph) -> None:
    """
    Renders a Mermaid diagram string into a Jupyter notebook cell
    using the mermaid.ink API. No external dependencies required.
    """
    graph_bytes = graph_code.encode("ascii")
    base64_bytes = base64.b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + base64_string))
