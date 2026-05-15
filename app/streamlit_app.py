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
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg: #0A0A14;
    --bg-surface: #13131D;
    --bg-elevated: #1B1B28;
    --purple-100: #EDE9FE;
    --purple-300: #C4B5FD;
    --purple-400: #A78BFA;
    --purple-500: #8B5CF6;
    --purple-600: #7C3AED;
    --purple-700: #6D28D9;
    --orange-300: #FDBA74;
    --orange-400: #FB923C;
    --orange-500: #F97316;
    --green-400: #34D399;
    --red-400: #F87171;
    --text: #E5E7EB;
    --text-muted: #9CA3AF;
    --border: rgba(167, 139, 250, 0.16);
    --border-strong: rgba(167, 139, 250, 0.32);
  }

  /* Global body type — distinctive but readable */
  html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  }
  code, pre, .arch-step, .kpi-label {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
  }

  /* Hide Streamlit's "Deploy" button + main menu for a portfolio-clean look */
  [data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
  .stDeployButton { display: none; }
  header[data-testid="stHeader"] { background: transparent; }
  #MainMenu { visibility: hidden; }
  [data-testid="stMarkdownContainer"] a.anchor-link { display: none !important; }
  h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a { display: none !important; }

  /* Subtle radial purple glow behind the hero — gives the page depth */
  [data-testid="stAppViewContainer"] {
    background: radial-gradient(
      ellipse at top left,
      rgba(124, 58, 237, 0.10) 0%,
      transparent 50%
    ), radial-gradient(
      ellipse at top right,
      rgba(251, 146, 60, 0.05) 0%,
      transparent 50%
    ), var(--bg) !important;
  }

  /* Content frame */
  .block-container { padding-top: 3rem; max-width: 1180px; }

  /* HERO ============================================================== */
  /* Scoped to .hero-title only — see the explicit HTML render below.
     Avoid styling all h1s globally (the ablation expanders also have
     h1s and they'd inherit the gradient otherwise). */
  h1.hero-title {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-size: 3.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.05 !important;
    margin: 0 0 0.5rem 0 !important;
    background: linear-gradient(110deg, #FFFFFF 0%, var(--purple-300) 60%, var(--orange-300) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .hero-tagline {
    color: var(--text-muted);
    font-size: 1.04rem;
    font-weight: 400;
    line-height: 1.55;
    max-width: 720px;
    margin: 0.25rem 0 1.2rem;
  }
  .hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(167, 139, 250, 0.10);
    color: var(--purple-300);
    padding: 4px 13px; border-radius: 999px;
    font-size: 0.78rem; letter-spacing: 0.04em;
    font-weight: 500; margin-right: 0.45rem;
    border: 1px solid var(--border);
    text-transform: uppercase;
    transition: all 0.2s ease;
  }
  .hero-pill:hover {
    background: rgba(251, 146, 60, 0.10);
    color: var(--orange-300);
    border-color: rgba(251, 146, 60, 0.32);
  }

  /* ARCHITECTURE FLOW =================================================== */
  .arch-flow {
    display: flex; flex-wrap: wrap; align-items: center;
    gap: 0.5rem 0.6rem;
    padding: 1.05rem 1.15rem;
    border-radius: 12px;
    background: linear-gradient(135deg,
      rgba(124, 58, 237, 0.08) 0%,
      rgba(124, 58, 237, 0.03) 100%);
    border: 1px solid var(--border);
    backdrop-filter: blur(4px);
  }
  .arch-step {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    background: rgba(167, 139, 250, 0.07);
    border: 1px solid var(--border);
    font-size: 0.82rem;
    color: var(--purple-300);
    transition: all 0.2s ease;
  }
  .arch-step:hover {
    background: rgba(167, 139, 250, 0.14);
    border-color: var(--border-strong);
    color: #FFFFFF;
  }
  .arch-step-final {
    background: linear-gradient(135deg, var(--orange-500) 0%, var(--orange-400) 100%);
    color: #0A0A14 !important;
    border: 1px solid var(--orange-400) !important;
    font-weight: 700;
    box-shadow: 0 0 22px rgba(249, 115, 22, 0.30);
  }
  .arch-step-final:hover {
    background: linear-gradient(135deg, var(--orange-400) 0%, var(--orange-300) 100%);
    color: #0A0A14 !important;
  }
  .arch-arrow {
    color: var(--purple-400);
    opacity: 0.45;
    font-size: 1rem;
    user-select: none;
  }

  /* SECTION TITLE ====================================================== */
  .section-title {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--purple-400);
    position: relative;
    padding-left: 0.7rem;
  }
  .section-title::before {
    content: '';
    position: absolute; left: 0; top: 50%;
    transform: translateY(-50%);
    width: 3px; height: 60%;
    background: linear-gradient(180deg, var(--purple-400), var(--orange-400));
    border-radius: 2px;
  }

  /* KPI TILES ========================================================== */
  .kpi-row {
    display: flex; gap: 0.85rem;
    margin: 0.5rem 0 1.5rem; flex-wrap: wrap;
  }
  .kpi-tile {
    flex: 1 1 0; min-width: 140px;
    padding: 0.95rem 1.05rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: linear-gradient(180deg,
      rgba(167, 139, 250, 0.04) 0%,
      rgba(167, 139, 250, 0.01) 100%);
    transition: all 0.25s ease;
  }
  .kpi-tile:hover {
    border-color: var(--border-strong);
    background: linear-gradient(180deg,
      rgba(167, 139, 250, 0.07) 0%,
      rgba(167, 139, 250, 0.02) 100%);
    transform: translateY(-1px);
  }
  .kpi-label {
    font-size: 0.68rem; letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
  }
  .kpi-value {
    font-size: 1.1rem; font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--text);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .kpi-value-num {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1.65rem; font-weight: 700;
    background: linear-gradient(135deg, var(--purple-300), var(--orange-300));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  /* VERIFIED BADGES — small green pills (keep semantic green) but
     with a subtle dark-mode-friendly chrome */
  .verified-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 11px; border-radius: 999px;
    font-size: 0.74rem; font-weight: 500;
    margin: 2px 4px 2px 0; line-height: 1.2;
    transition: all 0.15s ease;
  }
  .verified-ok {
    background: rgba(52, 211, 153, 0.10);
    color: var(--green-400);
    border: 1px solid rgba(52, 211, 153, 0.32);
  }
  .verified-ok:hover {
    background: rgba(52, 211, 153, 0.16);
    border-color: rgba(52, 211, 153, 0.55);
  }
  .verified-fail {
    background: rgba(248, 113, 113, 0.10);
    color: var(--red-400);
    border: 1px solid rgba(248, 113, 113, 0.32);
  }
  .verified-icon { font-weight: 700; font-size: 0.85rem; }

  /* CARDS — Streamlit st.container(border=True) wrapper */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg,
      rgba(167, 139, 250, 0.03) 0%,
      rgba(167, 139, 250, 0.01) 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
  }
  [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--border-strong) !important;
    box-shadow:
      0 1px 3px rgba(0, 0, 0, 0.3),
      0 0 24px rgba(167, 139, 250, 0.10) !important;
  }

  .card-rec {
    line-height: 1.6; font-size: 0.94rem;
    color: var(--text);
  }

  /* PRIMARY BUTTON (Generate verified report) */
  .stButton > button[kind="primary"],
  button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--purple-600) 0%, var(--purple-500) 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--purple-400) !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.55rem 1.15rem !important;
    box-shadow: 0 0 22px rgba(124, 58, 237, 0.28) !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button[kind="primary"]:hover,
  button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, var(--orange-500) 0%, var(--orange-400) 100%) !important;
    border-color: var(--orange-400) !important;
    color: #0A0A14 !important;
    box-shadow: 0 0 30px rgba(251, 146, 60, 0.40) !important;
    transform: translateY(-1px);
  }

  /* TABS — make the active tab pop with a purple underline */
  [data-testid="stTabs"] [role="tab"] {
    font-weight: 500;
    letter-spacing: 0.01em;
    padding: 0.5rem 1rem;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--orange-300) !important;
  }
  [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, var(--purple-400), var(--orange-400)) !important;
    height: 3px !important;
  }

  /* SIDEBAR */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E0E18 0%, #0A0A14 100%) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] h2 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--purple-300);
    margin-bottom: 1rem;
  }

  /* DATAFRAME — soften the harsh table look */
  [data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
  }

  /* SUCCESS / ERROR banners */
  [data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
  }

  /* FOOTER */
  .footer-note {
    font-size: 0.78rem;
    margin-top: 3.5rem;
    padding-top: 1.25rem;
    color: var(--text-muted);
    opacity: 0.7;
    border-top: 1px solid var(--border);
  }
  .footer-note a {
    color: var(--purple-300) !important;
  }
  .footer-note a:hover {
    color: var(--orange-300) !important;
  }

  /* Markdown headings inside Streamlit expanders — Streamlit's DOM
     uses several variants of expander-detail wrappers; cover them all.
     Important: any h1 NOT marked as `.hero-title` should be small. */
  details h1,
  [data-testid="stExpanderDetails"] h1,
  [data-testid="stExpander"] h1,
  .streamlit-expander h1,
  div[data-testid="stExpanderContent"] h1 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin: 0.5rem 0 0.75rem !important;
    color: var(--purple-300) !important;
    letter-spacing: -0.01em !important;
    background: none !important;
    -webkit-text-fill-color: var(--purple-300) !important;
    line-height: 1.3 !important;
  }
  details h2,
  [data-testid="stExpanderDetails"] h2,
  [data-testid="stExpander"] h2,
  .streamlit-expander h2 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
  }
  details table { font-size: 0.85rem; }
  details th {
    color: var(--purple-300) !important;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
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


# Hero — rendered as raw HTML so .hero-title CSS scoping is unambiguous
st.markdown(
    "<h1 class='hero-title'>pgx-digest</h1>",
    unsafe_allow_html=True,
)
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
