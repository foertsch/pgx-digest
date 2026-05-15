"""Streamlit demo for pgx-digest. Two tabs: "Try it" and "Ablations".

Makes the project's thesis (deterministic core, fenced LLM, typed
Verifier) visible by attaching a verified-field badge to each card.
The UI design goal is "clinical and calm" — recruiters should be able
to understand the architecture from a screenshot in 10 seconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Load API keys from the repo's .env (override=True because VS Code may
# export a blank ANTHROPIC_API_KEY from launchd). Both paths cover the
# repo-root and worktree layouts.
try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve().parent
    load_dotenv(_here.parent / ".env", override=True)
    load_dotenv(_here.parent.parent.parent.parent / ".env", override=True)
except ImportError:
    pass

# Make repo-root imports work whether streamlit is launched from the repo
# root or from `app/`. Also lets us load `app.lib` as a sibling module.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib import (  # noqa: E402  (sys.path setup above)
    build_provider,
    find_ablation_files,
    find_fixtures,
    parse_pharmcat_json,
    run_pipeline,
    summarize_bundle,
)

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results"

# ---------------------------------------------------------------------------
# Custom CSS — tighten spacing, polish badge & card visuals
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
  /* Hide Streamlit's "Deploy" button + main menu for a portfolio-clean look */
  [data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
  .stDeployButton { display: none; }
  header[data-testid="stHeader"] { background: transparent; }
  #MainMenu { visibility: hidden; }
  /* Hide Streamlit's anchor-link icon on Markdown headers — clutter */
  [data-testid="stMarkdownContainer"] a.anchor-link { display: none !important; }
  h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a { display: none !important; }

  /* Tighter overall content */
  .block-container { padding-top: 2.5rem; max-width: 1180px; }

  /* Hero strap line — use rgba so it reads in both light and dark mode */
  .hero-tagline { font-size: 1.02rem; font-weight: 400; margin-top: -0.25rem;
                  opacity: 0.78; }
  .hero-pill { display: inline-block;
               background: rgba(15, 118, 110, 0.12);
               color: rgb(13, 148, 136);
               padding: 3px 11px; border-radius: 999px;
               font-size: 0.78rem; letter-spacing: 0.02em;
               font-weight: 500; margin-right: 0.4rem;
               border: 1px solid rgba(15, 118, 110, 0.28); }

  /* Architecture flow — step pills, wraps on narrow viewports */
  .arch-flow { display: flex; flex-wrap: wrap; align-items: center;
               gap: 0.45rem 0.55rem; padding: 0.85rem 1rem;
               border-radius: 8px;
               background: rgba(15, 118, 110, 0.06);
               border: 1px solid rgba(15, 118, 110, 0.18);
               border-left: 3px solid #0F766E; }
  .arch-step { display: inline-block; padding: 3px 10px;
               border-radius: 6px;
               background: rgba(15, 118, 110, 0.10);
               font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
               font-size: 0.82rem; }
  .arch-step-final { background: #0F766E; color: white; font-weight: 600; }
  .arch-arrow { opacity: 0.45; font-size: 0.95rem; user-select: none; }

  /* Compact KPI tiles (replacement for st.metric — the default is enormous) */
  .kpi-row { display: flex; gap: 0.75rem; margin: 0.5rem 0 1.25rem; flex-wrap: wrap; }
  .kpi-tile { flex: 1 1 0; min-width: 140px; padding: 0.7rem 0.9rem;
              border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.22);
              background: rgba(148, 163, 184, 0.06); }
  .kpi-label { font-size: 0.72rem; letter-spacing: 0.04em;
               text-transform: uppercase; opacity: 0.65; margin-bottom: 0.15rem; }
  .kpi-value { font-size: 1.05rem; font-weight: 600;
               font-variant-numeric: tabular-nums;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .kpi-value-num { font-size: 1.35rem; }  /* short numeric values get more weight */

  /* Verified badges — small pills, dark/light mode neutral */
  .verified-badge { display: inline-flex; align-items: center; gap: 6px;
                    padding: 3px 9px; border-radius: 999px;
                    font-size: 0.75rem; font-weight: 500;
                    margin: 2px 4px 2px 0; line-height: 1.2;
                    border: 1px solid transparent; }
  .verified-ok { background: rgba(16, 185, 129, 0.12);
                 color: rgb(16, 185, 129);
                 border-color: rgba(16, 185, 129, 0.32); }
  .verified-fail { background: rgba(239, 68, 68, 0.12);
                   color: rgb(239, 68, 68);
                   border-color: rgba(239, 68, 68, 0.32); }
  .verified-icon { font-weight: 700; font-size: 0.85rem; }

  /* Section title — lightweight; relies on inherited text color */
  .section-title { font-size: 1.05rem; font-weight: 600;
                   margin: 0.25rem 0 0.6rem; letter-spacing: -0.01em;
                   opacity: 0.95; }

  /* Card recommendation prose */
  .card-rec { line-height: 1.55; font-size: 0.94rem; opacity: 0.92; }

  /* Footer credit */
  .footer-note { font-size: 0.78rem; margin-top: 2.5rem;
                 padding-top: 1rem; opacity: 0.55;
                 border-top: 1px solid rgba(148, 163, 184, 0.22); }

  /* Markdown headings inside Streamlit expanders (e.g. ablation tables
     that start with "# Ablation X: …") — Streamlit renders these as
     real <h1> and they dominate the page. Scope them down. */
  [data-testid="stExpanderDetails"] h1 { font-size: 1.15rem !important;
                                          font-weight: 600 !important;
                                          margin: 0.4rem 0 0.6rem !important; }
  [data-testid="stExpanderDetails"] h2 { font-size: 1.0rem !important;
                                          font-weight: 600 !important; }
</style>
"""


def _kpi_tile(label: str, value: str, *, numeric: bool = False) -> str:
    """Render a compact dashboard tile.

    `numeric=True` bumps the value font-size — useful for short integer
    values like "1" / "2" where the tile reads better with more weight.
    """
    value_class = "kpi-value kpi-value-num" if numeric else "kpi-value"
    return (
        f"<div class='kpi-tile'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='{value_class}' title='{value}'>{value}</div>"
        f"</div>"
    )

def _arch_flow_html() -> str:
    steps = [
        ("PharmCAT JSON", False),
        ("Bundle[PGxFinding]", False),
        ("Ranker", False),
        ("Triage", False),
        ("LLM Drafter", False),
        ("Verifier", False),
        ("verified report", True),
    ]
    parts: list[str] = []
    for i, (label, final) in enumerate(steps):
        cls = "arch-step arch-step-final" if final else "arch-step"
        parts.append(f"<span class='{cls}'>{label}</span>")
        if i < len(steps) - 1:
            parts.append("<span class='arch-arrow'>→</span>")
    return "<div class='arch-flow'>" + "".join(parts) + "</div>"


ARCH_FLOW_HTML = _arch_flow_html()


# ---------------------------------------------------------------------------
# UI primitives
# ---------------------------------------------------------------------------


def _badge(field: str, value: str, ok: bool) -> str:
    """Render a verified/failed pill (inline HTML span)."""
    cls = "verified-ok" if ok else "verified-fail"
    icon = "✓" if ok else "✕"
    state = "verified" if ok else "FAILED"
    title = (
        f"verified against Bundle.{field} = {value}"
        if ok
        else f"verifier failure on Bundle.{field}"
    )
    return (
        f"<span class='verified-badge {cls}' title='{title}'>"
        f"<span class='verified-icon'>{icon}</span> "
        f"<b>{field}</b> · {state}"
        f"</span>"
    )


def _render_card(card, failures) -> None:
    """Render one verified report card."""
    failed_fields = {f.field for f in failures}
    ok = lambda field: field not in failed_fields  # noqa: E731

    badges_html = "".join(
        [
            _badge("gene", card.gene, ok("gene")),
            _badge("diplotype", card.diplotype, ok("diplotype")),
            _badge("phenotype", card.phenotype, ok("phenotype")),
            _badge("drug", card.drug, ok("drug")),
            _badge(
                "cited_pmids",
                f"{len(card.cited_pmids)} PMID",
                ok("cited_pmids"),
            ),
            _badge("recommendation", "no cross-gene leak", ok("recommendation")),
        ]
    )

    with st.container(border=True):
        # Header: bigger, clearer hierarchy.
        st.markdown(
            f"#### `{card.gene}` &nbsp;·&nbsp; {card.diplotype} &nbsp;→&nbsp; "
            f"_{card.phenotype}_"
        )
        st.markdown(f"**Drug:** {card.drug}")
        st.markdown(
            f"<div class='card-rec'>{card.recommendation}</div>",
            unsafe_allow_html=True,
        )
        if card.cited_pmids:
            links = " · ".join(
                f"[PMID {p}](https://pubmed.ncbi.nlm.nih.gov/{p})"
                for p in card.cited_pmids
            )
            st.caption(f"Citations: {links}")
        # Badges row, with a bit of breathing room.
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown(badges_html, unsafe_allow_html=True)
        if failures:
            with st.expander(f"⚠ {len(failures)} verifier failure(s)"):
                for f in failures:
                    st.write(
                        f"- `{f.field}` = {f.value!r}: {f.reason}"
                    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="pgx-digest — verified PGx reports",
    page_icon="🧬",
    layout="wide",
)

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _init_state() -> None:
    st.session_state.setdefault("api_key", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("result_fixture", None)
    st.session_state.setdefault("provider_name", "anthropic")


_init_state()


# Hero
st.title("pgx-digest")
st.markdown(
    "<div class='hero-tagline'>"
    "<span class='hero-pill'>Deterministic core</span>"
    "<span class='hero-pill'>Fenced LLM</span>"
    "<span class='hero-pill'>Typed Verifier</span>"
    "<br>Privacy-first pharmacogenomic narrative reports where every "
    "claim is verified against the source Bundle."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown(ARCH_FLOW_HTML, unsafe_allow_html=True)
st.markdown("&nbsp;", unsafe_allow_html=True)


# Sidebar — fixture picker + provider + API key
with st.sidebar:
    st.header("Configuration")
    fixtures = find_fixtures(FIXTURES_DIR)
    if not fixtures:
        st.error(f"No fixtures found in {FIXTURES_DIR}")
        st.stop()
    names = [f.name for f in fixtures]
    selected_name = st.selectbox(
        "PharmCAT fixture",
        names,
        index=0,
        help="Each fixture is a real or synthetic PharmCAT report JSON.",
    )
    selected_fixture = fixtures[names.index(selected_name)]

    provider_name = st.radio(
        "Provider",
        options=["anthropic", "gemini"],
        index=0,
        horizontal=True,
    )
    st.session_state.provider_name = provider_name

    env_var = (
        "ANTHROPIC_API_KEY" if provider_name == "anthropic" else "GEMINI_API_KEY"
    )
    api_key = st.text_input(
        f"{provider_name.capitalize()} API key",
        type="password",
        value=st.session_state.api_key,
        help=f"Optional. Falls back to ${env_var} if blank.",
        placeholder=f"paste or leave blank to use ${env_var}",
    )
    st.session_state.api_key = api_key

    st.divider()
    st.caption(
        f"`{REPO_ROOT.name}` · {len(fixtures)} fixture(s) · "
        f"[GitHub](https://github.com/foertsch/pgx-digest)"
    )


tab_try, tab_ablations = st.tabs(["▶ Try it", "📊 Ablations"])


with tab_try:
    bundle = parse_pharmcat_json(selected_fixture)
    summary = summarize_bundle(bundle)

    st.markdown(
        "<div class='section-title'>Bundle</div>", unsafe_allow_html=True
    )

    # Humanize "pharmcat_cyp2c19_minimal.json" → "CYP2C19 minimal"
    raw = selected_fixture.stem.replace("pharmcat_", "").replace(".report", "")
    fixture_label = " ".join(
        word.upper() if word.upper() in {"CYP2C19", "CYP2D6", "TPMT", "DPYD"}
        or (word.isalpha() and len(word) <= 5 and word.isupper())
        else word
        for word in raw.replace("_", " ").split()
    )
    tiles_html = (
        "<div class='kpi-row'>"
        + _kpi_tile("Fixture", fixture_label)
        + _kpi_tile("Genes called", str(summary.n_genes), numeric=True)
        + _kpi_tile("Drug recs", str(summary.n_drugs), numeric=True)
        + _kpi_tile("Privacy tier", bundle.privacy_tier.value)
        + "</div>"
    )
    st.markdown(tiles_html, unsafe_allow_html=True)

    st.dataframe(
        list(summary.rows),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    cta_col, info_col = st.columns([1, 3])
    with cta_col:
        clicked = st.button(
            "Generate verified report",
            type="primary",
            use_container_width=True,
        )
    with info_col:
        st.caption(
            f"One **{provider_name}** API call · ~\\$0.001 on "
            f"Anthropic Haiku 4.5 · no API call until you click."
        )

    if clicked:
        env_key = os.getenv(env_var)
        effective_key = st.session_state.api_key.strip() or env_key
        if not effective_key:
            st.error(
                f"No API key. Paste one in the sidebar or set ${env_var}."
            )
        else:
            with st.spinner(f"Running pipeline via {provider_name}…"):
                try:
                    provider = build_provider(
                        provider_name,
                        st.session_state.api_key.strip() or None,
                    )
                    result = run_pipeline(bundle, provider)
                    st.session_state.result = result
                    st.session_state.result_fixture = selected_fixture.name
                except Exception as e:  # noqa: BLE001 — surface anything
                    st.session_state.result = None
                    st.exception(e)

    result = st.session_state.result
    if result is not None and st.session_state.result_fixture == selected_fixture.name:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>Verified report</div>",
            unsafe_allow_html=True,
        )

        fails_by_card: dict[int, list] = {}
        for f in result.verification.failures:
            fails_by_card.setdefault(f.card_index, []).append(f)
        for i, card in enumerate(result.draft.cards):
            _render_card(card, fails_by_card.get(i, []))

        st.markdown("&nbsp;", unsafe_allow_html=True)
        if result.verification.passed:
            st.success(
                f"**Verifier: PASSED** — all {len(result.draft.cards)} "
                f"card(s) traced back to the Bundle. "
                "Every green badge = a field the Verifier checked."
            )
        else:
            st.error(
                f"**Verifier: FAILED** — "
                f"{len(result.verification.failures)} failure(s) across "
                f"{len(result.draft.cards)} card(s). Draft rejected; in "
                f"production this triggers a retry."
            )
    elif result is not None:
        st.info(
            "Fixture changed since last run. Click **Generate verified "
            "report** to draft against the new bundle."
        )


with tab_ablations:
    st.markdown(
        "<div class='section-title'>Ablation studies</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Each table answers one architectural question by toggling a "
        "single component and measuring what changes. Tables generated "
        "by `uv run examples/run_ablations.py`."
    )
    files = find_ablation_files(EVAL_RESULTS_DIR)
    if not files:
        st.info(
            "No ablation tables found at `eval_results/`. Run "
            "`uv run examples/run_ablations.py --only-verifier` "
            "for the zero-API table (Ablation A)."
        )
    else:
        for f in files:
            # Friendly label: drop the timestamp; keep ablation_X_name.md
            label = f.stem.replace("_", " ").title()
            with st.expander(label, expanded=False):
                st.markdown(f.read_text())


st.markdown(
    "<div class='footer-note'>"
    "pgx-digest is a portfolio project — not a medical device, not "
    "diagnostic, for research / educational use only. "
    "See <a href='https://github.com/foertsch/pgx-digest/blob/main/DISCLAIMER.md'>"
    "DISCLAIMER.md</a>."
    "</div>",
    unsafe_allow_html=True,
)
