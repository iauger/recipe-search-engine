# app.py
# Run with: python -m streamlit run app.py

"""
Recipe Search — Streamlit UI

Provides two interaction modes:
  1. Single mode  — search in one selected mode, full result detail per card
  2. Compare      — run all five modes, NDCG summary table + expandable lists
"""

import sys
import io
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import numpy as np

from src.config import load_settings
from src.engine import SearchEngine, SearchMode
from src.evaluate import ndcg_at_k, score_result


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Recipe Search",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — structural/layout only; card content uses inline styles
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif; }

    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p { color: #eee !important; }
    [data-testid="stSidebar"] .stMarkdown { color: #eee; }

    .app-header {
        padding: 8px 0 24px 0;
        border-bottom: 2px solid #e8e4dc;
        margin-bottom: 28px;
    }
    .app-title {
        font-family: 'Playfair Display', serif;
        font-size: 36px; font-weight: 700;
        color: #1a1a2e; margin: 0; line-height: 1.1;
    }
    .app-subtitle { font-size: 14px; color: #888; margin-top: 6px; }

    .compare-table {
        width: 100%; border-collapse: collapse;
        font-size: 13px; margin-bottom: 24px;
    }
    .compare-table th {
        background: #1a1a2e; color: #fff;
        padding: 10px 14px; text-align: left; font-weight: 500;
    }
    .compare-table td { padding: 9px 14px; border-bottom: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Engine initialisation (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading search engine...")
def get_engine():
    s = load_settings()
    return SearchEngine(s)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_LABELS = {
    SearchMode.HYBRID:          "Hybrid (full pipeline)",
    SearchMode.LEXICAL:         "Lexical (BM25 only)",
    SearchMode.SEMANTIC:        "Semantic (embeddings only)",
    SearchMode.QUALITY:         "Quality (rating signal only)",
    SearchMode.ABLATION_NO_SEM: "Ablation (no semantic)",
}

MODE_DESCRIPTIONS = {
    SearchMode.HYBRID:
        "Full pipeline: BM25 + rule-based alignment + embedding similarity + quality, "
        "with stratified weights based on query intent richness.",
    SearchMode.LEXICAL:
        "Elasticsearch BM25 only. No reranking applied. Serves as the baseline.",
    SearchMode.SEMANTIC:
        "Embedding cosine similarity + quality score only. Bypasses structured rules.",
    SearchMode.QUALITY:
        "Sorts by predicted Bayesian rating from the Phase 2 RecipeNet model. "
        "Query-agnostic — surfaces generally well-rated recipes.",
    SearchMode.ABLATION_NO_SEM:
        "Hybrid minus embedding similarity. Direct comparison with Hybrid "
        "isolates the marginal contribution of the embedding layer.",
}

# Inline style dict — bypasses Streamlit's CSS class sanitization on injected HTML
_S = {
    "card":   ("background:#fff;border:1px solid #e8e4dc;border-radius:12px;"
               "padding:18px 22px;margin-bottom:14px;"
               "box-shadow:0 2px 8px rgba(0,0,0,0.06);"),
    "rank":   ("font-size:11px;font-weight:500;letter-spacing:0.1em;"
               "color:#9e8f7a;text-transform:uppercase;margin-bottom:4px;"),
    "name":   ("font-size:18px;font-weight:700;color:#1a1a2e;"
               "margin-bottom:6px;line-height:1.3;"),
    "meta":   "font-size:13px;color:#666;margin-bottom:10px;",
    "srow":   "display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;",
    "pill":   ("font-size:11px;font-weight:500;padding:3px 10px;"
               "border-radius:20px;background:#f4f0ea;color:#5a4e3c;"),
    "pillhi": ("font-size:11px;font-weight:500;padding:3px 10px;"
               "border-radius:20px;background:#1a1a2e;color:#fff;"),
    "tagrow": "display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;",
    "chip":   ("font-size:11px;padding:2px 8px;border-radius:4px;"
               "background:#eef2ff;color:#3b3f8c;border:1px solid #d0d5f5;"),
    "ing":    "font-size:12px;color:#777;margin-top:6px;line-height:1.6;",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tier_badge(tier: str) -> str:
    colours = {
        "high":     "background:#fce8e8;color:#b03030;",
        "medium":   "background:#fff3e0;color:#a05000;",
        "low":      "background:#e8f5e9;color:#2e7d32;",
        "lexical":  "background:#f0f0f0;color:#444;",
        "semantic": "background:#f0f0f0;color:#444;",
        "quality":  "background:#f0f0f0;color:#444;",
    }
    base = ("display:inline-block;font-size:12px;font-weight:600;"
            "letter-spacing:0.08em;text-transform:uppercase;"
            "padding:4px 14px;border-radius:20px;margin-bottom:16px;")
    colour = colours.get(tier.split("_")[0], "background:#f0f0f0;color:#666;")
    label  = tier.upper() + " INTENT"
    return f'<span style="{base}{colour}">{label}</span>'


def render_intent_summary(intent: dict) -> None:
    parts = []
    if intent.get("dietary_tags"):
        parts.append(f"<b>Dietary:</b> {', '.join(intent['dietary_tags'])}")
    if intent.get("proteins"):
        parts.append(f"<b>Protein:</b> {', '.join(intent['proteins'])}")
    if intent.get("cuisines"):
        parts.append(f"<b>Cuisine:</b> {', '.join(intent['cuisines'])}")
    if intent.get("courses"):
        parts.append(f"<b>Course:</b> {', '.join(intent['courses'])}")
    if intent.get("methods"):
        parts.append(f"<b>Method:</b> {', '.join(intent['methods'])}")
    if intent.get("occasions"):
        parts.append(f"<b>Occasion:</b> {', '.join(intent['occasions'])}")
    if intent.get("max_minutes"):
        parts.append(f"<b>Max time:</b> {intent['max_minutes']} mins")
    if intent.get("taste"):
        parts.append(f"<b>Taste:</b> {', '.join(intent['taste'])}")

    if parts:
        box = ("background:#f8f6f2;border-radius:10px;padding:12px 16px;"
               "margin-bottom:20px;font-size:13px;color:#444;")
        st.markdown(
            f'<div style="{box}">Parsed intent &nbsp;|&nbsp; '
            + " &nbsp;&middot;&nbsp; ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )


def render_result_card(rank: int, result, show_scores: bool = True) -> None:
    source = result.source
    name   = source.get("name", "Unknown").title()
    minutes = source.get("minutes")
    rating  = source.get("bayesian_rating")

    tags = source.get("tags_clean", []) or []
    if isinstance(tags, str):
        tags = tags.split()

    ingredients = source.get("ingredients_clean", []) or []
    if isinstance(ingredients, str):
        ingredients = ingredients.split()

    mins_str   = f"{int(float(minutes))}m" if minutes and float(minutes) > 0 else "?"
    rating_str = f"\u2605 {float(rating):.2f}" if rating else ""

    # Score pills
    score_html = ""
    if show_scores and hasattr(result, "final_score"):
        pills = [f'<span style="{_S["pillhi"]}">Score {result.final_score:.3f}</span>']
        if result.alignment_score:
            pills.append(
                f'<span style="{_S["pill"]}">Align {result.alignment_score:.2f}</span>'
            )
        if result.semantic_sim:
            pills.append(
                f'<span style="{_S["pill"]}">Sim {result.semantic_sim:.3f}</span>'
            )
        if result.quality_score:
            pills.append(
                f'<span style="{_S["pill"]}">Quality {result.quality_score:.2f}</span>'
            )
        score_html = f'<div style="{_S["srow"]}">' + "".join(pills) + "</div>"

    # Tags (top 8)
    tag_html = ""
    if tags:
        display_tags = [t.replace("-", " ").replace("_", " ") for t in tags[:8]]
        chips = "".join(
            f'<span style="{_S["chip"]}">{t}</span>' for t in display_tags
        )
        tag_html = f'<div style="{_S["tagrow"]}">{chips}</div>'

    # Ingredients (top 10)
    ing_html = ""
    if ingredients:
        ing_clean = [i.replace("_", " ") for i in ingredients[:10]]
        ing_html = (
            f'<div style="{_S["ing"]}">'
            f'<b>Ingredients:</b> {", ".join(ing_clean)}'
            f'</div>'
        )

    # Assemble full card as one string — single st.markdown call
    card = (
        f'<div style="{_S["card"]}">'
        f'<div style="{_S["rank"]}">#{rank}</div>'
        f'<div style="{_S["name"]}">{name}</div>'
        f'<div style="{_S["meta"]}">{mins_str} &nbsp;&middot;&nbsp; {rating_str}</div>'
        f'{score_html}{tag_html}{ing_html}'
        f'</div>'
    )
    st.markdown(card, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🍳 Recipe Search")
    st.markdown("---")

    query = st.text_input(
        "Search query",
        placeholder="e.g. easy chicken dinner under 45 mins",
        label_visibility="collapsed",
    )

    st.markdown("#### Search mode")
    selected_mode = st.radio(
        "mode",
        options=list(SearchMode),
        format_func=lambda m: MODE_LABELS[m],
        label_visibility="collapsed",
    )

    top_k = st.slider("Results to retrieve", min_value=5, max_value=20, value=10, step=5)

    run_single  = st.button("🔍 Search", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("#### Compare all modes")
    run_compare = st.button("⚖️ Run comparison", use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<small style='color:#aaa'><b>Mode:</b> "
        f"{MODE_DESCRIPTIONS[selected_mode]}</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main panel header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <p class="app-title">Recipe Search</p>
    <p class="app-subtitle">
        INFO 624 &nbsp;&middot;&nbsp; Five-mode IR system &nbsp;&middot;&nbsp;
        BM25 + Rule-based Alignment + RecipeNet Embeddings
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Single mode search
# ---------------------------------------------------------------------------

if run_single and query.strip():
    engine = get_engine()

    with st.spinner(f"Searching in {MODE_LABELS[selected_mode]} mode..."):
        result = engine.run(query.strip(), mode=selected_mode, top_k=top_k)

    tier = result.tier

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(tier_badge(tier), unsafe_allow_html=True)
        st.markdown(
            f"**{len(result.results)} results** &nbsp;&middot;&nbsp; "
            f"mode: `{selected_mode.value}`"
        )
    with col_b:
        w = result.weights
        st.markdown(
            f"<small style='color:#666'>"
            f"lex={w['lex']} &nbsp;&middot;&nbsp; align={w['alignment']} "
            f"&nbsp;&middot;&nbsp; sem={w['semantic']} "
            f"&nbsp;&middot;&nbsp; quality={w['quality']}"
            f"</small>",
            unsafe_allow_html=True,
        )

    render_intent_summary(result.intent)
    st.markdown("---")

    for i, r in enumerate(result.results, start=1):
        render_result_card(i, r, show_scores=(selected_mode != SearchMode.LEXICAL))


# ---------------------------------------------------------------------------
# Compare all modes
# ---------------------------------------------------------------------------

elif run_compare and query.strip():
    engine = get_engine()

    with st.spinner("Running all five modes..."):
        all_results = engine.run_all_modes(query.strip(), top_k=top_k)

    tier   = all_results[SearchMode.HYBRID].tier
    intent = all_results[SearchMode.HYBRID].intent

    st.markdown(tier_badge(tier), unsafe_allow_html=True)
    render_intent_summary(intent)

    # -- NDCG summary table -----------------------------------------------
    st.markdown("### Mode comparison")

    mode_ndcg = {
        mode: ndcg_at_k(sr.results, intent, k=min(5, top_k))
        for mode, sr in all_results.items()
    }
    best_mode = max(mode_ndcg.keys(), key=lambda m: mode_ndcg[m])

    table_rows = ""
    for mode, sr in all_results.items():
        ndcg = mode_ndcg[mode]
        w    = sr.weights
        is_winner = mode == best_mode
        star = " &#9733;" if is_winner else ""

        # Winner row gets a warm highlight; others alternate light/white
        row_bg   = "background:#fffbf0;" if is_winner else ""
        cell_w   = "font-weight:700;color:#1a1a2e;" if is_winner else "color:#333;"
        td_base  = f"padding:9px 14px;border-bottom:1px solid #eee;{row_bg}"
        td_text  = f"{td_base}{cell_w}"

        table_rows += (
            f'<tr>'
            f'<td style="{td_text}">{MODE_LABELS[mode]}</td>'
            f'<td style="{td_base}color:#666;">{sr.tier}</td>'
            f'<td style="{td_base}color:#555;">{w["lex"]}</td>'
            f'<td style="{td_base}color:#555;">{w["alignment"]}</td>'
            f'<td style="{td_base}color:#555;">{w["semantic"]}</td>'
            f'<td style="{td_base}color:#555;">{w["quality"]}</td>'
            f'<td style="{td_text}">{ndcg:.4f}{star}</td>'
            f'</tr>'
        )

    st.markdown(
        '<table class="compare-table"><thead><tr>'
        '<th>Mode</th><th>Tier</th>'
        '<th>w_lex</th><th>w_align</th><th>w_sem</th><th>w_quality</th>'
        '<th>NDCG@5</th>'
        f'</tr></thead><tbody>{table_rows}</tbody></table>',
        unsafe_allow_html=True,
    )

    # -- Expandable per-mode result lists ---------------------------------
    st.markdown("### Results by mode")

    for mode, sr in all_results.items():
        with st.expander(
            f"{MODE_LABELS[mode]}  —  NDCG@5: {mode_ndcg[mode]:.4f}",
            expanded=(mode == SearchMode.HYBRID),
        ):
            for i, r in enumerate(sr.results, start=1):
                render_result_card(
                    i, r,
                    show_scores=(mode != SearchMode.LEXICAL),
                )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

elif not query.strip():
    st.markdown("""
    <div style="text-align:center;padding:80px 0;color:#aaa;">
        <div style="font-size:48px;margin-bottom:16px;">&#127859;</div>
        <div style="font-size:22px;color:#555;margin-bottom:8px;">
            What are you hungry for?
        </div>
        <div style="font-size:14px;">
            Try <em>easy chicken dinner under 45 mins</em>,
            <em>vegan thanksgiving sides</em>, or
            <em>something cozy and warming</em>
        </div>
    </div>
    """, unsafe_allow_html=True)
