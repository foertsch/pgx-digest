"""Streamlit demo for pgx-digest. Two tabs: "Try it" and "Ablations".

Makes the project's thesis (deterministic core, fenced LLM, typed
Verifier) visible by attaching a verified-field badge to each card.

UI ships with four named color themes spanning night→day, picked
from the sidebar. Each theme is a set of CSS variable values plugged
into the same static stylesheet.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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
# Color themes — spanning night→day with progressively muted accents.
# All themes keep semantic green for verified badges; only chrome shifts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """Color set for one named theme. Plugged into CSS variables at render time."""

    name: str
    blurb: str
    # Base surfaces (top → bottom of the page)
    bg: str
    bg_surface: str
    bg_elevated: str
    # Primary accent (purple-family in dark themes, dusty-violet in light)
    accent: str
    accent_light: str
    accent_strong: str
    # Destination highlight (orange-family — muted across all themes now)
    highlight: str
    highlight_light: str
    # Text
    text: str
    text_muted: str
    # Borders + decorative glows
    border: str
    border_strong: str
    glow_accent: str  # rgba shadow for primary glow
    glow_highlight: str  # rgba shadow for orange CTA hover
    # Visual texture
    bg_gradient: str  # CSS for radial / linear background overlays
    # Whether to invert "code" / "monospace" green for legibility on light bg
    is_light: bool = False


THEMES: tuple[Theme, ...] = (
    Theme(
        name="Night Vision",
        blurb="Vivid violet on near-black. Highest contrast.",
        bg="#0A0A14",
        bg_surface="#13131D",
        bg_elevated="#1B1B28",
        accent="#A78BFA",
        accent_light="#C4B5FD",
        accent_strong="#8B5CF6",
        highlight="#D97757",  # toned-down burnt amber (was #FB923C)
        highlight_light="#E8A57E",
        text="#E5E7EB",
        text_muted="#9CA3AF",
        border="rgba(167, 139, 250, 0.16)",
        border_strong="rgba(167, 139, 250, 0.32)",
        glow_accent="rgba(124, 58, 237, 0.22)",
        glow_highlight="rgba(217, 119, 87, 0.30)",
        bg_gradient=(
            "radial-gradient(ellipse at top left, "
            "rgba(124, 58, 237, 0.10) 0%, transparent 50%), "
            "radial-gradient(ellipse at top right, "
            "rgba(217, 119, 87, 0.04) 0%, transparent 50%)"
        ),
    ),
    Theme(
        name="Twilight",
        blurb="Deep plum, lavender accents, warm coral highlight.",
        bg="#13101E",
        bg_surface="#1D1828",
        bg_elevated="#272235",
        accent="#B084E0",
        accent_light="#D0B5F0",
        accent_strong="#9466CC",
        highlight="#C46B5A",
        highlight_light="#D89580",
        text="#E5E0EC",
        text_muted="#9590A6",
        border="rgba(176, 132, 224, 0.16)",
        border_strong="rgba(176, 132, 224, 0.32)",
        glow_accent="rgba(176, 132, 224, 0.18)",
        glow_highlight="rgba(196, 107, 90, 0.24)",
        bg_gradient=(
            "radial-gradient(ellipse at top left, "
            "rgba(148, 102, 204, 0.10) 0%, transparent 55%), "
            "radial-gradient(ellipse at bottom right, "
            "rgba(196, 107, 90, 0.04) 0%, transparent 50%)"
        ),
    ),
    Theme(
        name="Overcast",
        blurb="Slate-blue grey, muted lavender, dusty clay highlight.",
        bg="#15181F",
        bg_surface="#1F232C",
        bg_elevated="#272C37",
        accent="#9296BF",  # slate-lavender, low saturation
        accent_light="#B3B8DD",
        accent_strong="#7B82A8",
        highlight="#B58F75",  # clay
        highlight_light="#C9AB97",
        text="#DDE0E8",
        text_muted="#888EA0",
        border="rgba(146, 150, 191, 0.18)",
        border_strong="rgba(146, 150, 191, 0.32)",
        glow_accent="rgba(146, 150, 191, 0.14)",
        glow_highlight="rgba(181, 143, 117, 0.20)",
        bg_gradient=(
            "radial-gradient(ellipse at top, "
            "rgba(146, 150, 191, 0.06) 0%, transparent 60%)"
        ),
    ),
    Theme(
        name="Daybreak",
        blurb="Warm cream background, dusty violet, terracotta accents.",
        bg="#F5F1EA",
        bg_surface="#FFFFFF",
        bg_elevated="#FFFFFF",
        accent="#6B5B95",
        accent_light="#8E7DB8",
        accent_strong="#574A7E",
        highlight="#B8704A",
        highlight_light="#CC8E68",
        text="#2A2728",
        text_muted="#6B6770",
        border="rgba(107, 91, 149, 0.18)",
        border_strong="rgba(107, 91, 149, 0.32)",
        glow_accent="rgba(107, 91, 149, 0.10)",
        glow_highlight="rgba(184, 112, 74, 0.18)",
        bg_gradient=(
            "radial-gradient(ellipse at top, "
            "rgba(107, 91, 149, 0.05) 0%, transparent 60%)"
        ),
        is_light=True,
    ),
)

THEME_BY_NAME: dict[str, Theme] = {t.name: t for t in THEMES}


def _alpha(hex_color: str, alpha: float) -> str:
    """`#RRGGBB` + alpha → `rgba(r, g, b, a)`. Used so each theme can
    derive its own tint values from its accent color without hardcoding
    rgba strings everywhere.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _theme_root_css(theme: Theme) -> str:
    """Emit a `:root` block + background override for the chosen theme.

    The static stylesheet below uses `var(--*)`; this block defines the
    values. Switching themes is therefore zero-touch on the static CSS.

    Derives per-theme tint variables (soft/medium/strong + a highlight
    variant) from the accent colors via `_alpha()`, so card backgrounds
    and hover states don't need hardcoded rgba in the static CSS.
    """
    # Text that sits on top of the (saturated) highlight color — used for
    # the destination pill + primary-button hover. Dark text on dark
    # themes, white on the light theme.
    text_on_hl = "#FFFFFF" if theme.is_light else theme.bg
    # Hero gradient starting color: white on dark themes (gradient flows
    # white → accent → highlight); on light themes, white is invisible,
    # so flow accent-deep → accent → highlight instead.
    hero_grad_start = theme.accent_strong if theme.is_light else "#FFFFFF"
    # On the light theme, Streamlit's `base=dark` config still drives
    # widget chrome. We don't try to retheme widgets that draw their own
    # canvas (the dataframe stays dark — acceptable contrast on cream).
    # We just call out the light theme with a short note in the caption
    # below the selector. Nothing to inject here.
    light_overrides = ""
    return f"""<style id="theme-vars">
      :root {{
        --bg: {theme.bg};
        --bg-surface: {theme.bg_surface};
        --bg-elevated: {theme.bg_elevated};
        --accent-300: {theme.accent_light};
        --accent-400: {theme.accent};
        --accent-500: {theme.accent_strong};
        --accent-600: {theme.accent_strong};
        --accent-700: {theme.accent_strong};
        --highlight-300: {theme.highlight_light};
        --highlight-400: {theme.highlight};
        --highlight-500: {theme.highlight};
        --text: {theme.text};
        --text-muted: {theme.text_muted};
        --border: {theme.border};
        --border-strong: {theme.border_strong};
        --glow-accent: {theme.glow_accent};
        --glow-highlight: {theme.glow_highlight};
        --text-on-highlight: {text_on_hl};
        --hero-grad-start: {hero_grad_start};
        /* Accent tints, derived per-theme so each palette feels coherent */
        --tint-soft: {_alpha(theme.accent, 0.04)};
        --tint-medium: {_alpha(theme.accent, 0.08)};
        --tint-strong: {_alpha(theme.accent, 0.14)};
        --tint-highlight-soft: {_alpha(theme.highlight, 0.05)};
        --tint-highlight-medium: {_alpha(theme.highlight, 0.10)};
        --tint-highlight-strong: {_alpha(theme.highlight, 0.32)};
        /* Card/tile bottom — lighter than top, used in linear-gradients */
        --tint-faint: {_alpha(theme.accent, 0.01)};
      }}
      [data-testid="stAppViewContainer"] {{
        background: {theme.bg_gradient}, var(--bg) !important;
      }}
      [data-testid="stAppViewContainer"] *, body {{
        color: var(--text);
      }}
      {light_overrides}
    </style>"""

# ---------------------------------------------------------------------------
# Custom CSS — tighten spacing, polish badge & card visuals
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* CSS variables come from _theme_root_css(). Semantic constants stay here. */
  :root {
    --green-400: #34D399;
    --red-400: #F87171;
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

  /* Background (radial glow + bg color) is set by _theme_root_css() per theme */

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
    /* `--hero-grad-start` is theme-aware: white on dark themes, accent on light */
    background: linear-gradient(110deg, var(--hero-grad-start) 0%, var(--accent-400) 60%, var(--highlight-400) 100%);
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
    background: var(--tint-medium);
    color: var(--accent-300);
    padding: 4px 13px; border-radius: 999px;
    font-size: 0.78rem; letter-spacing: 0.04em;
    font-weight: 500; margin-right: 0.45rem;
    border: 1px solid var(--border);
    text-transform: uppercase;
    transition: all 0.2s ease;
  }
  .hero-pill:hover {
    background: var(--tint-highlight-medium);
    color: var(--highlight-300);
    border-color: var(--tint-highlight-strong);
  }

  /* ARCHITECTURE FLOW =================================================== */
  .arch-flow {
    display: flex; flex-wrap: wrap; align-items: center;
    gap: 0.5rem 0.6rem;
    padding: 1.05rem 1.15rem;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--tint-medium) 0%, var(--tint-soft) 100%);
    border: 1px solid var(--border);
    backdrop-filter: blur(4px);
  }
  .arch-step {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    background: var(--tint-medium);
    border: 1px solid var(--border);
    font-size: 0.82rem;
    color: var(--accent-300);
    transition: all 0.2s ease;
  }
  .arch-step:hover {
    background: var(--tint-strong);
    border-color: var(--border-strong);
    color: var(--text);
  }
  .arch-step-final {
    background: linear-gradient(135deg, var(--highlight-500) 0%, var(--highlight-400) 100%);
    color: var(--text-on-highlight) !important;
    border: 1px solid var(--highlight-400) !important;
    font-weight: 700;
    box-shadow: 0 0 22px var(--glow-highlight);
  }
  .arch-step-final:hover {
    background: linear-gradient(135deg, var(--highlight-400) 0%, var(--highlight-300) 100%);
    color: var(--text-on-highlight) !important;
  }
  .arch-arrow {
    color: var(--accent-400);
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
    color: var(--accent-400);
    position: relative;
    padding-left: 0.7rem;
  }
  .section-title::before {
    content: '';
    position: absolute; left: 0; top: 50%;
    transform: translateY(-50%);
    width: 3px; height: 60%;
    background: linear-gradient(180deg, var(--accent-400), var(--highlight-400));
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
    background: linear-gradient(180deg, var(--tint-soft) 0%, var(--tint-faint) 100%);
    transition: all 0.25s ease;
  }
  .kpi-tile:hover {
    border-color: var(--border-strong);
    background: linear-gradient(180deg, var(--tint-medium) 0%, var(--tint-soft) 100%);
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
    background: linear-gradient(135deg, var(--accent-300), var(--highlight-300));
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
    background: linear-gradient(180deg, var(--tint-soft) 0%, var(--tint-faint) 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
  }
  [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--border-strong) !important;
    box-shadow:
      0 1px 3px rgba(0, 0, 0, 0.3),
      0 0 24px var(--glow-accent) !important;
  }

  .card-rec {
    line-height: 1.6; font-size: 0.94rem;
    color: var(--text);
  }

  /* PRIMARY BUTTON (Generate verified report) */
  .stButton > button[kind="primary"],
  button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--accent-600) 0%, var(--accent-500) 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent-400) !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.55rem 1.15rem !important;
    box-shadow: 0 0 22px var(--glow-accent) !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button[kind="primary"]:hover,
  button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, var(--highlight-500) 0%, var(--highlight-400) 100%) !important;
    border-color: var(--highlight-400) !important;
    color: var(--text-on-highlight) !important;
    box-shadow: 0 0 30px var(--glow-highlight) !important;
    transform: translateY(-1px);
  }

  /* TABS — make the active tab pop with a purple underline */
  [data-testid="stTabs"] [role="tab"] {
    font-weight: 500;
    letter-spacing: 0.01em;
    padding: 0.5rem 1rem;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--highlight-300) !important;
  }
  [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, var(--accent-400), var(--highlight-400)) !important;
    height: 3px !important;
  }

  /* SIDEBAR */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg) 100%) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] h2 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--accent-300);
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
    color: var(--accent-300) !important;
  }
  .footer-note a:hover {
    color: var(--highlight-300) !important;
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
    color: var(--accent-300) !important;
    letter-spacing: -0.01em !important;
    background: none !important;
    -webkit-text-fill-color: var(--accent-300) !important;
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
    color: var(--accent-300) !important;
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
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    st.session_state.setdefault("api_key", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("result_fixture", None)
    st.session_state.setdefault("provider_name", "anthropic")
    st.session_state.setdefault("theme_name", THEMES[0].name)
    # `?theme=Twilight` URL override — handy for sharing a specific look
    # and for headless screenshots during development.
    qp_theme = st.query_params.get("theme")
    if qp_theme and qp_theme in THEME_BY_NAME:
        st.session_state.theme_name = qp_theme


_init_state()

# Inject theme variables FIRST so the static CSS below resolves them.
_active_theme = THEME_BY_NAME.get(st.session_state.theme_name, THEMES[0])
st.markdown(_theme_root_css(_active_theme), unsafe_allow_html=True)
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


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


# Sidebar — theme + fixture picker + provider + API key
with st.sidebar:
    st.header("Configuration")

    theme_names = [t.name for t in THEMES]
    chosen_theme_name = st.selectbox(
        "Theme",
        theme_names,
        index=theme_names.index(st.session_state.theme_name),
        help="Color palettes spanning night → day. Same UI, four moods.",
    )
    if chosen_theme_name != st.session_state.theme_name:
        st.session_state.theme_name = chosen_theme_name
        st.rerun()
    st.caption(THEME_BY_NAME[chosen_theme_name].blurb)
    st.divider()

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
