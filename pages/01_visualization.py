# pages/01_visualization.py

import sys
import io
import pickle
import os
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import streamlit as st

from src.config import load_settings
from src.engine import SearchEngine, SearchMode


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Embedding Space — Recipe Search",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p { color: #eee !important; }
    [data-testid="stSidebar"] .stMarkdown { color: #eee; }
</style>
""", unsafe_allow_html=True)

# Cached loaders

@st.cache_resource(show_spinner="Loading search engine...")
def get_engine():
    s = load_settings()
    return SearchEngine(s)


@st.cache_data(show_spinner="Loading embedding manifold...")
def load_manifold_data(processed_dir: str):
    """
    Load the precomputed UMAP projection and companion metadata.
    Returns projection (N,2), recipe_ids (N,), targets (N,).
    """
    meta_path = os.path.join(processed_dir, "final_residual_v2_umap_meta.npz")
    if not os.path.exists(meta_path):
        return None, None, None

    data = np.load(meta_path, allow_pickle=True)
    return (
        data["projection"].astype(np.float32),   # (N, 2)
        data["recipe_ids"],                        # (N,) str
        data["targets"].astype(np.float32),        # (N,) float
    )


@st.cache_resource(show_spinner="Loading UMAP reducer...")
def load_reducer(processed_dir: str):
    """Load the fitted UMAP reducer for projecting new query points."""
    reducer_path = os.path.join(processed_dir, "final_residual_v2_umap_reducer.pkl")
    if not os.path.exists(reducer_path):
        return None
    with open(reducer_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RANK_COLORS = [
    "#e63946",  # #1 — vivid red
    "#f4a261",  # #2 — orange
    "#2a9d8f",  # #3 — teal
    "#457b9d",  # #4 — steel blue
    "#6a4c93",  # #5 — purple
    "#264653",  # #6
    "#e9c46a",  # #7
    "#52b788",  # #8
    "#a8dadc",  # #9
    "#c77dff",  # #10
]


def project_query_embedding(
    query_embedding: torch.Tensor,
    reducer,
) -> np.ndarray | None:
    """
    Project a 128D query embedding into the 2D UMAP space using the
    fitted reducer's transform() method.

    Returns a (2,) array, or None if projection fails.
    Note: UMAP transform() on a single out-of-distribution sparse vector
    may produce an approximate position — this is expected and is
    documented in the UI as an approximation.
    """
    try:
        vec = query_embedding.detach().cpu().numpy().reshape(1, -1)
        point = reducer.transform(vec).astype(np.float32)
        return point[0]  # (2,)
    except Exception as e:
        st.warning(f"Query projection failed: {e}")
        return None


def subsample_corpus(
    projection: np.ndarray,
    targets: np.ndarray,
    max_points: int = 8000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample the corpus for rendering performance.
    Stratified by rating quintile to preserve the manifold's color gradient.
    """
    n = len(projection)
    if n <= max_points:
        return projection, targets

    rng = np.random.default_rng(seed)

    # Stratified sample across 5 rating quintiles
    quintiles = np.percentile(targets, [20, 40, 60, 80])
    bins = np.digitize(targets, quintiles)
    per_bin = max_points // 5
    indices = []

    for b in range(5):
        bin_idx = np.where(bins == b)[0]
        k = min(per_bin, len(bin_idx))
        indices.append(rng.choice(bin_idx, size=k, replace=False))

    sampled = np.concatenate(indices)
    return projection[sampled], targets[sampled]


def build_figure(
    projection: np.ndarray,
    targets: np.ndarray,
    result_points: list[dict] | None = None,
    query_point: np.ndarray | None = None,
    query_label: str = "Query",
) -> go.Figure:
    """
    Build the Plotly figure.

    Layers (bottom to top):
      1. Corpus scatter — subsampled, colored by Bayesian rating
      2. Result highlights — one trace per result, colored by rank
      3. Query star — projected query position
    """
    proj_sub, tgt_sub = subsample_corpus(projection, targets)

    fig = go.Figure()

    # -- Layer 1: corpus --------------------------------------------------
    fig.add_trace(go.Scattergl(
        x=proj_sub[:, 0],
        y=proj_sub[:, 1],
        mode="markers",
        marker=dict(
            color=tgt_sub,
            colorscale="Spectral_r",
            cmin=float(targets.min()),
            cmax=float(targets.max()),
            size=3,
            opacity=0.35,
            colorbar=dict(
                title=dict(text="Bayesian Rating", side="right"),
                thickness=14,
                len=0.6,
                x=-0.12,
                xanchor="left",
                xpad=0,
            ),
            line=dict(width=0),
        ),
        name="Recipe corpus",
        hoverinfo="skip",
        showlegend=True,
    ))

    # -- Layer 2: search results ------------------------------------------
    if result_points:
        for i, rp in enumerate(result_points):
            colour = RANK_COLORS[i % len(RANK_COLORS)]
            fig.add_trace(go.Scatter(
                x=[rp["x"]],
                y=[rp["y"]],
                mode="markers+text",
                marker=dict(
                    color=colour,
                    size=14,
                    symbol="circle",
                    line=dict(color="white", width=1.5),
                ),
                text=[f"#{rp['rank']}"],
                textposition="top center",
                textfont=dict(size=10, color=colour),
                name=f"#{rp['rank']} {rp['name'][:30]}",
                hovertemplate=(
                    f"<b>#{rp['rank']} {rp['name']}</b><br>"
                    f"Rating: {rp['rating']:.2f}<br>"
                    f"Score: {rp['score']:.3f}<br>"
                    f"Sim: {rp['sim']:.3f}"
                    "<extra></extra>"
                ),
            ))

    # -- Layer 3: query star ----------------------------------------------
    if query_point is not None:
        fig.add_trace(go.Scatter(
            x=[query_point[0]],
            y=[query_point[1]],
            mode="markers+text",
            marker=dict(
                color="#ffffff",
                size=18,
                symbol="star",
                line=dict(color="#1a1a2e", width=2),
            ),
            text=["Query"],
            textposition="bottom center",
            textfont=dict(size=11, color="#ffffff"),
            name=f'Query: "{query_label}"',
            hovertemplate=(
                f"<b>Query</b>: {query_label}<br>"
                "(Approximate position — sparse projection)"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        plot_bgcolor="#0e0e1a",
        paper_bgcolor="#0e0e1a",
        font=dict(color="#ccc", family="DM Sans"),
        title=dict(
            text="RecipeNet Latent Space — UMAP Projection",
            font=dict(size=18, color="#eee"),
            x=0.02,
        ),
        xaxis=dict(
            title="UMAP Dimension 1",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title_font=dict(color="#888"),
        ),
        yaxis=dict(
            title="UMAP Dimension 2",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title_font=dict(color="#888"),
        ),
        legend=dict(
            bgcolor="rgba(30,30,50,0.85)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=11, color="#ccc"),
            itemsizing="constant",
        ),
        margin=dict(l=100, r=20, t=60, b=20),
        height=640,
    )

    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🗺️ Embedding Space")
    st.markdown("---")

    # Pre-populate from main search page if user navigated here after searching
    _default_query = st.session_state.get("shared_query", 
                     st.session_state.get("viz_query", ""))
    query = st.text_input(
        "Search query",
        value=_default_query,
        placeholder="e.g. something cozy and warming",
        label_visibility="collapsed",
    )

    top_k = st.slider("Results to show", min_value=3, max_value=10, value=5)

    mode_options = {
        SearchMode.HYBRID:  "Hybrid",
        SearchMode.LEXICAL: "Lexical",
        SearchMode.SEMANTIC:"Semantic",
    }
    selected_mode = st.radio(
        "Search mode",
        options=list(mode_options.keys()),
        format_func=lambda m: mode_options[m],
    )

    run_viz = st.button("🔍 Search & Visualise", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <small style='color:#888;line-height:1.6'>
    <b>How to read this chart</b><br>
    Each dot is a recipe in the 128-D latent space learned by RecipeNet,
    projected to 2D via UMAP. Colour encodes Bayesian rating
    (dark red = high, blue = low).<br><br>
    Numbered circles are your search results. The ★ star marks where
    the query embedding lands — its position is <em>approximate</em>
    since queries project into a space built for dense recipe vectors.
    </small>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.markdown("""
<div style='padding:8px 0 20px 0;border-bottom:2px solid #333;margin-bottom:24px;'>
    <p style='font-family:"Playfair Display",serif;font-size:30px;font-weight:700;
              color:#eee;margin:0;line-height:1.1;'>
        Latent Recipe Space
    </p>
    <p style='font-size:13px;color:#888;margin-top:6px;'>
        RecipeNet Residual V2 &nbsp;·&nbsp; 128-D embeddings &nbsp;·&nbsp; UMAP projection
    </p>
</div>
""", unsafe_allow_html=True)

# -- Load manifold data ---------------------------------------------------
s = load_settings()
projection, recipe_ids, targets = load_manifold_data(s.processed_dir)
reducer = load_reducer(s.processed_dir)

if projection is None or recipe_ids is None:
    st.error(
        "Manifold data not found. Run `python -m scripts.generate_umap` first "
        "to generate the UMAP projection and reducer files."
    )
    st.stop()

if reducer is None:
    st.warning(
        "Fitted UMAP reducer not found — query projection will be unavailable. "
        "Re-run `python -m scripts.generate_umap` to regenerate."
    )

# Build recipe_id → projection index lookup
id_to_proj_idx = {str(rid): i for i, rid in enumerate(recipe_ids)}

# -- Run search and build viz ----------------------------------------------
if run_viz and query and query.strip():
    st.session_state["viz_query"]    = query.strip()
    st.session_state["shared_query"] = query.strip()

    engine = get_engine()

    with st.spinner("Searching..."):
        result = engine.run(
            query.strip(),
            mode=selected_mode,
            top_k=top_k,
            return_query_embedding=True,
        )

    # Collect result projection points
    result_points = []
    missing_ids = []

    for rank, r in enumerate(result.results, start=1):
        recipe_id = str(r.recipe_id)
        proj_idx  = id_to_proj_idx.get(recipe_id)

        if proj_idx is None:
            missing_ids.append(recipe_id)
            continue

        rating = r.source.get("bayesian_rating", 0.0) or 0.0
        result_points.append({
            "rank":   rank,
            "name":   r.source.get("name", "Unknown").title(),
            "x":      float(projection[proj_idx, 0]),
            "y":      float(projection[proj_idx, 1]),
            "rating": float(rating),
            "score":  float(r.final_score),
            "sim":    float(r.semantic_sim),
        })

    # Project query embedding
    query_point = None
    if reducer is not None and result.query_embedding is not None:
        with st.spinner("Projecting query into manifold..."):
            query_point = project_query_embedding(result.query_embedding, reducer)

    # Build and render figure
    if targets is not None:
        fig = build_figure(
            projection=projection,
            targets=targets,
            result_points=result_points,
            query_point=query_point,
            query_label=query.strip(),
        )
    else:
        st.error("Targets data is missing from manifold.")
        st.stop()

    # Derive intent tier from the query's structured signal count,
    # independent of the mode's fixed weight label (result.tier reflects
    # the mode name for LEXICAL/SEMANTIC/QUALITY, not the intent richness).
    from src.reranker import SemanticReranker
    _intent_tier = engine.reranker.get_weight_profile(
        engine.projector.project(query.strip(), result.intent)
    )["tier"].split("_")[0]

    tier_colours = {
        "high":   ("#fce8e8", "#b03030"),
        "medium": ("#fff3e0", "#a05000"),
        "low":    ("#e8f5e9", "#2e7d32"),
    }
    bg, fg = tier_colours.get(_intent_tier, ("#f0f0f0", "#444"))
    st.markdown(
        f'<span style="display:inline-block;font-size:12px;font-weight:600;'
        f'letter-spacing:0.08em;text-transform:uppercase;padding:4px 14px;'
        f'border-radius:20px;margin-bottom:16px;background:{bg};color:{fg};">'
        f'{_intent_tier.upper()} INTENT</span>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    if missing_ids:
        st.caption(
            f"Note: {len(missing_ids)} result(s) not found in the projection "
            f"(recipe IDs may differ between the embedding bundle and the index)."
        )

    if query_point is not None:
        st.caption(
            "★ Query position is approximate — sparse query projection into a "
            "dense recipe embedding space may not land precisely near semantically "
            "related recipes. See report Section 4.3 for discussion."
        )

    # Result list below chart
    st.markdown("### Search results")
    for rank, rp in enumerate(result_points, start=1):
        colour = RANK_COLORS[(rank - 1) % len(RANK_COLORS)]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'padding:10px 16px;margin-bottom:8px;background:#1a1a2e;'
            f'border-radius:8px;border-left:4px solid {colour};">'
            f'<span style="font-size:16px;font-weight:700;color:{colour};'
            f'min-width:28px;">#{rank}</span>'
            f'<span style="color:#eee;font-size:14px;">{rp["name"]}</span>'
            f'<span style="margin-left:auto;font-size:12px;color:#888;">'
            f'★ {rp["rating"]:.2f} &nbsp;·&nbsp; score {rp["score"]:.3f} '
            f'&nbsp;·&nbsp; sim {rp["sim"]:.3f}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

elif not query or not query.strip():
    # Show the full manifold with no highlights when no query is active
    if targets is not None:
        fig = build_figure(projection=projection, targets=targets)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Targets data is missing from manifold.")
    st.markdown(
        "<p style='text-align:center;color:#666;font-size:13px;'>"
        "Enter a query in the sidebar to highlight search results on the manifold."
        "</p>",
        unsafe_allow_html=True,
    )
