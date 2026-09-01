"""AutoAgentSearch — Streamlit dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import math
import os
import time

import streamlit as st

from core.agentic_wrapper import is_valid_architecture, wrap_in_agentic_scaffold
from core.beam_search import beam_search
from core.dag_inflater import inflate_to_dag
from core.mcts_search import mcts_search
from core.models import Constraints
from core.repository import ComponentRepository
from core.rl_search import rl_search
from core.task_parser import parse_task
from visualization.graph_plot import render_architecture_data_url


# =============================================================================
# PAGE
# =============================================================================
st.set_page_config(
    page_title="AutoAgentSearch",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)
REPO_PATH = os.path.join(os.path.dirname(__file__), "components")


@st.cache_resource
def get_repository() -> ComponentRepository:
    return ComponentRepository(REPO_PATH)


repo = get_repository()


# =============================================================================
# THEME  (Linear / Vercel-inspired dark palette + indigo/violet accent)
# =============================================================================
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  :root {
    --fg-strong: #FAFAFA;
    --fg:        #E4E4E7;
    --muted:     #A1A1AA;
    --subtle:    #71717A;
    --border:    #27272A;
    --border-2:  #3F3F46;
    --bg:        #0A0A0A;
    --bg-elev:   #121215;
    --panel:     #18181B;
    --panel-2:   #1F1F23;
    --accent:    #6366F1;
    --accent-2:  #8B5CF6;
    --success:   #10B981;
    --warn:      #F59E0B;
    --danger:    #EF4444;
  }

  /* Dark surface everywhere */
  html, body { color-scheme: dark !important; }
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  .main,
  section.main,
  section[data-testid="stMain"] {
    background-color: var(--bg) !important;
    color: var(--fg) !important;
  }

  /* Font family only — never override colour globally, it breaks widgets */
  html, body, [class*="css"], [class*="st-"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI",
                 sans-serif !important;
  }
  .main .block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1320px !important;
    background-color: var(--bg) !important;
    color: var(--fg) !important;
  }
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

  h1, h2, h3, h4 { color: var(--fg-strong) !important; letter-spacing: -0.02em; }
  .stApp p, .stApp span, .stApp div, .stApp label { color: var(--fg); }

  /* Widget labels */
  .stSlider label, .stCheckbox label, .stSelectbox label,
  .stNumberInput label, .stTextArea label, .stTextInput label {
    font-size: 0.85rem !important;
    color: var(--fg) !important;
    font-weight: 500 !important;
  }
  .stCheckbox label p { font-size: 0.9rem !important; color: var(--fg) !important; }

  /* Slider track — subtle violet gradient */
  .stSlider [data-baseweb="slider"] > div > div > div > div {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
  }

  /* ---------- HERO ---------- */
  .hero {
    display: flex; flex-direction: column; align-items: center; text-align: center;
    padding: 0.6rem 0 2.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.4rem;
  }
  .hero-mark {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 20px; font-weight: 800;
    box-shadow: 0 8px 24px rgba(99,102,241,0.35);
    margin-bottom: 1.1rem;
  }
  .hero-title {
    font-size: 2.75rem; font-weight: 800; color: var(--fg-strong);
    letter-spacing: -0.035em; line-height: 1.05; margin: 0 0 0.6rem 0;
  }
  .hero-title .grad {
    background: linear-gradient(135deg, #A5B4FC 0%, #C4B5FD 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero-sub {
    font-size: 1.05rem; color: var(--muted); max-width: 640px;
    line-height: 1.55; margin: 0;
  }
  .hero-chips {
    display: flex; gap: 0.5rem; margin-top: 1.2rem; flex-wrap: wrap;
    justify-content: center;
  }
  .hero-chip {
    font-size: 0.76rem; font-weight: 600; color: var(--muted);
    padding: 4px 11px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--bg-elev);
    letter-spacing: 0.02em;
  }
  .hero-chip .dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
  }
  .dot-i { background: #6366F1; }
  .dot-v { background: #A855F7; }
  .dot-g { background: #10B981; }

  /* ---------- SECTION LABEL ---------- */
  .section-eyebrow {
    font-size: 0.72rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 0 0 0.55rem 0;
  }
  .section-h {
    font-size: 1.35rem; font-weight: 700; color: var(--fg-strong);
    margin: 0 0 0.35rem 0;
  }
  .section-sub {
    font-size: 0.92rem; color: var(--muted); margin: 0 0 1.2rem 0;
  }

  /* ---------- EXAMPLE CHIPS (task shortcuts) ---------- */
  /* Streamlit buttons in the "example chip" row get a subtle look */
  .example-row .stButton > button {
    background: var(--bg-elev) !important;
    border: 1px solid var(--border) !important;
    color: var(--fg) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 0.9rem !important;
    border-radius: 999px !important;
    box-shadow: none !important;
  }
  .example-row .stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(99,102,241,0.10) !important;
    color: var(--fg-strong) !important;
  }

  /* ---------- CARD (config panel + result columns) ---------- */
  .card {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
  }

  /* ---------- CONFIG COLUMN TITLE ---------- */
  .cfg-title {
    font-size: 0.8rem; font-weight: 700; color: var(--fg-strong);
    padding-bottom: 0.55rem; margin: 0 0 0.9rem 0;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.5rem;
  }
  .cfg-title .cfg-icon {
    width: 18px; height: 18px; border-radius: 5px;
    background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(139,92,246,0.35));
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 10px; color: var(--fg-strong); font-weight: 700;
  }

  /* ---------- KEY-VALUE STRIP (parsed task summary) ---------- */
  .kv-strip {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    margin: 0.4rem 0 1.6rem 0; align-items: center;
  }
  .kv {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 0.82rem;
    padding: 5px 12px; border: 1px solid var(--border);
    border-radius: 6px; background: var(--bg-elev); color: var(--fg);
  }
  .kv b { color: var(--muted); font-weight: 500; font-size: 0.78rem; }
  .kv code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.8rem; color: var(--fg-strong); font-weight: 500;
    background: transparent; padding: 0;
  }

  /* ---------- BADGE ---------- */
  .badge {
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; border: 1px solid transparent;
    letter-spacing: 0.01em;
  }
  .badge-ok   { background: rgba(16,185,129,0.14); color: #6EE7B7;
                border-color: rgba(16,185,129,0.35); }
  .badge-warn { background: rgba(245,158,11,0.14); color: #FCD34D;
                border-color: rgba(245,158,11,0.35); }

  /* ---------- BUTTONS ---------- */
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    padding: 0.5rem 1rem !important;
    border: 1px solid var(--border) !important;
    background: var(--panel-2) !important;
    color: var(--fg-strong) !important;
    transition: all 0.12s ease !important;
  }
  .stButton > button * { color: inherit !important; }
  .stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(99,102,241,0.10) !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: white !important;
    border: 1px solid rgba(139,92,246,0.5) !important;
    box-shadow: 0 6px 22px rgba(99,102,241,0.38) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
  }
  .stButton > button[kind="primary"] * { color: white !important; }
  .stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
    box-shadow: 0 10px 28px rgba(99,102,241,0.5) !important;
  }

  /* ---------- INPUTS ---------- */
  .stTextArea textarea, .stTextInput input, .stNumberInput input {
    border-radius: 10px !important;
    border-color: var(--border) !important;
    background: var(--panel-2) !important;
    color: var(--fg-strong) !important;
    font-size: 0.95rem !important;
  }
  .stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.20) !important;
  }
  .stSelectbox [data-baseweb="select"] > div {
    background: var(--panel-2) !important;
    border-color: var(--border) !important;
    color: var(--fg-strong) !important;
    border-radius: 10px !important;
  }

  /* Matplotlib figures blend with the dark canvas */
  [data-testid="stImage"] > div, .stPlotlyChart, .stPyplot {
    background: transparent !important;
  }

  /* ---------- ARCH DIAGRAM CONTAINER ----------
     Fills the card's column horizontally, capped at ~55 % viewport
     height so the whole result card still fits on a laptop screen. */
  .arch-wrap {
    display: flex; align-items: center; justify-content: center;
    width: 100%;
    max-height: 55vh;
    background: transparent;
    margin: 0.5rem 0 0.3rem 0;
    overflow: hidden;
  }
  .arch-wrap img {
    width: 100%;
    max-height: 55vh;
    height: auto;
    object-fit: contain;
    display: block;
  }

  /* ---------- RESULT CARD (per-engine winner) ---------- */
  .result-card {
    position: relative;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.35rem 1.25rem 1.35rem;
    height: 100%;
    overflow: hidden;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }
  .result-card:hover { border-color: var(--border-2); transform: translateY(-2px); }
  .result-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--engine-stripe, var(--accent));
  }
  .result-card.winner {
    border-color: rgba(16,185,129,0.45);
    box-shadow: 0 0 0 1px rgba(16,185,129,0.20), 0 12px 40px rgba(16,185,129,0.08);
  }
  .result-head {
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 0.6rem; margin-bottom: 0.35rem;
  }
  .engine-name {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--engine-color, var(--accent));
  }
  .engine-crown {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 9px; border-radius: 999px;
    background: rgba(16,185,129,0.14);
    color: #6EE7B7;
    border: 1px solid rgba(16,185,129,0.35);
  }
  .engine-char {
    font-size: 0.78rem; color: var(--muted); margin: 0 0 0.9rem 0;
  }
  .engine-char b {
    color: var(--fg); font-weight: 600;
  }
  .engine-score {
    display: flex; align-items: baseline; gap: 0.4rem;
    margin: 0 0 1rem 0;
  }
  .engine-score .val {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 1.9rem; font-weight: 700; color: var(--fg-strong);
    letter-spacing: -0.03em; line-height: 1;
  }
  .engine-score .lbl {
    font-size: 0.72rem; color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
  }
  .engine-score .time {
    margin-left: auto;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.78rem; color: var(--subtle);
  }

  /* ---------- METRIC LIST WITH USAGE BARS ---------- */
  .mlist { margin-top: 0.9rem; }
  .mrow {
    padding: 0.55rem 0;
    border-top: 1px solid var(--border-2);
  }
  .mrow:first-child { border-top: none; }
  .mrow-head {
    display: flex; justify-content: space-between; align-items: baseline;
  }
  .mrow-k {
    font-size: 0.72rem; color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .mrow-v {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.92rem; color: var(--fg-strong); font-weight: 600;
  }
  .mrow-bar {
    height: 4px; margin-top: 0.4rem; background: var(--panel-2);
    border-radius: 999px; overflow: hidden;
  }
  .mrow-bar > span {
    display: block; height: 100%;
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    border-radius: 999px;
  }
  .mrow-bar.warn > span { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
  .mrow-bar.ok > span   { background: linear-gradient(90deg, #10B981, #34D399); }

  /* ---------- EMPTY STATE ---------- */
  .empty {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem;
    margin: 1.4rem 0 0 0;
  }
  .empty-card {
    background: var(--bg-elev); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem 1.2rem;
  }
  .empty-idx {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 8px;
    background: rgba(99,102,241,0.14);
    color: #A5B4FC; font-weight: 700; font-size: 0.85rem;
    border: 1px solid rgba(99,102,241,0.30);
    margin-bottom: 0.7rem;
  }
  .empty-t {
    font-size: 0.95rem; font-weight: 700; color: var(--fg-strong);
    margin: 0 0 0.35rem 0;
  }
  .empty-s {
    font-size: 0.85rem; color: var(--muted); line-height: 1.55; margin: 0;
  }

  /* ---------- FOOTER ---------- */
  .footer {
    margin-top: 4rem; padding-top: 1.4rem;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 0.8rem; color: var(--subtle);
  }
  .footer code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: var(--muted); background: transparent;
  }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HERO
# =============================================================================
st.markdown(
    """
<div class="hero">
  <div class="hero-mark">◆</div>
  <h1 class="hero-title">Auto Agent <span class="grad">Search</span></h1>
  <p class="hero-sub">
    Describe a business task and a resource budget.
    Three independent search algorithms explore the space of AI-agent architectures
    and return their best design side by side.
  </p>
  <div class="hero-chips">
    <span class="hero-chip"><span class="dot dot-i"></span>Beam Search</span>
    <span class="hero-chip"><span class="dot dot-v"></span>Monte-Carlo Tree Search</span>
    <span class="hero-chip"><span class="dot dot-g"></span>Reinforcement Learning</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# TASK INPUT
# =============================================================================
st.markdown(
    '<div class="section-eyebrow">Step 1 · Task</div>'
    '<h3 class="section-h">Describe what you want the agent to do</h3>'
    '<p class="section-sub">'
    'Write in plain English. The parser will detect the domain '
    '(healthcare, finance, or auto) and the expected input/output types.'
    '</p>',
    unsafe_allow_html=True,
)

if "task_text" not in st.session_state:
    st.session_state.task_text = ""

st.markdown('<div class="example-row">', unsafe_allow_html=True)
c1, c2, _ = st.columns([1.4, 1.4, 5])
with c1:
    if st.button("＋  Healthcare example", use_container_width=True):
        st.session_state.task_text = (
            "Extract diagnosis and medications from patient discharge summary PDF."
        )
with c2:
    if st.button("＋  Finance example", use_container_width=True):
        st.session_state.task_text = (
            "Detect fraud in a scanned loan application document and produce a risk report."
        )
st.markdown('</div>', unsafe_allow_html=True)

task_text = st.text_area(
    "Task",
    value=st.session_state.task_text,
    height=110,
    placeholder="e.g. Extract ICD codes from clinical notes and validate them.",
    label_visibility="collapsed",
)


# =============================================================================
# CONFIG PANEL
# =============================================================================
st.markdown(
    '<div style="height:1.6rem;"></div>'
    '<div class="section-eyebrow">Step 2 · Configuration</div>'
    '<h3 class="section-h">Set the search space and resource budget</h3>'
    '<p class="section-sub">'
    'Every candidate architecture that violates these limits is discarded '
    'before scoring. Tighter budgets narrow the search; looser ones let '
    'bigger, more parallel designs compete.'
    '</p>',
    unsafe_allow_html=True,
)

col_search, col_arch, col_res = st.columns(3, gap="medium")

with col_search:
    st.markdown(
        '<div class="card">'
        '<div class="cfg-title"><span class="cfg-icon">◎</span>Search</div>',
        unsafe_allow_html=True,
    )
    _user_domains = [d for d in repo.domains() if d != "agentic"]
    domain = st.selectbox("Domain", ["auto"] + _user_domains, index=0,
                          help="Choose 'auto' to let the parser decide from the task text.")
    st.caption(
        "All three engines run on every generation. "
        "Their best architectures appear side by side below."
    )
    beam_width = 5
    mcts_iterations = 400
    mcts_exploration = math.sqrt(2)  # classical UCT constant
    rl_episodes = 300
    st.markdown('</div>', unsafe_allow_html=True)

with col_arch:
    st.markdown(
        '<div class="card">'
        '<div class="cfg-title"><span class="cfg-icon">◇</span>Architecture</div>',
        unsafe_allow_html=True,
    )
    allow_dag = st.checkbox(
        "Parallel & hierarchical structures", value=True,
        help="Allow the inflater to widen compatible stages into parallel branches.",
    )
    max_width = st.slider(
        "Max branch width", 2, 5, 3, 1, disabled=not allow_dag,
        help="Upper bound on how many components can run in parallel per stage.",
    )
    if not allow_dag:
        max_width = 1
    wrap_agentic = st.checkbox(
        "Wrap in agentic orchestration", value=True,
        help="Wrap the domain body in Planner → Verifier → … → Recovery scaffold.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_res:
    st.markdown(
        '<div class="card">'
        '<div class="cfg-title"><span class="cfg-icon">⚡</span>Resource Budget</div>',
        unsafe_allow_html=True,
    )
    max_cost    = st.slider("Max cost per run (USD)", 0.001, 0.10, 0.05, 0.001, format="$%.3f")
    max_latency = st.slider("Max latency (seconds)", 0.5, 10.0, 5.0, 0.1)
    max_memory  = st.slider("Max memory (MB)", 256, 16384, 8192, 128)
    cpu_only    = st.checkbox(
        "CPU-only deployment", value=False,
        help="Reject any architecture that requires GPU-only components.",
    )
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div style="height:1.6rem;"></div>', unsafe_allow_html=True)
_, cbtn, _ = st.columns([1, 2, 1])
with cbtn:
    run = st.button("Generate architecture  →", type="primary", use_container_width=True)


# =============================================================================
# EMPTY STATE (before first generate)
# =============================================================================
if not run:
    st.markdown(
        """
<div style="height:2rem;"></div>
<div class="section-eyebrow">What happens next</div>
<h3 class="section-h">Three engines, one comparison</h3>
<div class="empty">
  <div class="empty-card">
    <div class="empty-idx">1</div>
    <p class="empty-t">Parse the task</p>
    <p class="empty-s">Extract the domain, the input type, and the desired output from your description.</p>
  </div>
  <div class="empty-card">
    <div class="empty-idx">2</div>
    <p class="empty-t">Search the space</p>
    <p class="empty-s">Beam, MCTS, and RL each explore type-compatible architectures within your budget.</p>
  </div>
  <div class="empty-card">
    <div class="empty-idx">3</div>
    <p class="empty-t">Compare winners</p>
    <p class="empty-s">The best architecture from each engine appears side by side, with metrics vs. budget.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# HELPERS
# =============================================================================
_ENGINE_META = {
    "Beam Search": {
        "short":   "Beam Search",
        "color":   "#818CF8",   # indigo-400
        "stripe":  "linear-gradient(90deg, #3B82F6, #6366F1)",
        "trait":   "Thorough",
        "why":     "Width-K expansion — surfaces full hierarchical ensembles across reasoning and validation.",
    },
    "Monte-Carlo Tree Search": {
        "short":   "MCTS",
        "color":   "#C084FC",   # violet-400
        "stripe":  "linear-gradient(90deg, #8B5CF6, #EC4899)",
        "trait":   "Explorative",
        "why":     "UCT-guided exploration — favours ensembles of reasoning components.",
    },
    "Reinforcement Learning": {
        "short":   "Reinforcement Learning",
        "color":   "#34D399",   # emerald-400
        "stripe":  "linear-gradient(90deg, #10B981, #14B8A6)",
        "trait":   "Adaptive",
        "why":     "Tabular Q-learning — favours redundant validation for learned safety.",
    },
}


def _mem_str(mb: int) -> str:
    """Render a memory value as GB when >= 1 GB, else MB."""
    return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb} MB"


def _bar_class(pct: float) -> str:
    """Colour the usage bar: green under 60 %, warn over 90 %, default in between."""
    if pct >= 0.90:
        return "warn"
    if pct <= 0.60:
        return "ok"
    return ""


def _mrow(label: str, value: str, pct: float | None = None) -> str:
    """A single metric row with an optional usage bar (0..1 of budget)."""
    bar_html = ""
    if pct is not None:
        pct_display = max(0.0, min(1.0, pct))
        bar_html = (
            f'<div class="mrow-bar {_bar_class(pct)}">'
            f'<span style="width:{pct_display*100:.1f}%;"></span>'
            f'</div>'
        )
    return (
        '<div class="mrow">'
        '<div class="mrow-head">'
        f'<span class="mrow-k">{label}</span>'
        f'<span class="mrow-v">{value}</span>'
        '</div>'
        f'{bar_html}'
        '</div>'
    )


def _metric_list(arch, cons) -> str:
    """Full metric list with usage bars vs the user's budget."""
    return '<div class="mlist">' + "".join([
        _mrow("Quality", f"{arch.aggregate_quality*100:.1f}%", arch.aggregate_quality),
        _mrow("Cost",    f"${arch.total_cost:.4f}",
              arch.total_cost / max(cons.max_cost, 1e-9)),
        _mrow("Latency", f"{arch.total_latency:.2f}s",
              arch.total_latency / max(cons.max_latency, 1e-9)),
        _mrow("Memory",  _mem_str(arch.peak_memory),
              arch.peak_memory / max(cons.max_memory, 1)),
        _mrow("Score",   f"{arch.score:.3f}", None),
    ]) + '</div>'


# =============================================================================
# RUN SEARCH
# =============================================================================
if run:
    if not task_text.strip():
        st.warning("Please describe a task first.")
        st.stop()

    task = parse_task(task_text, explicit_domain=domain)
    cons = Constraints(
        max_cost=max_cost, max_latency=max_latency,
        max_memory=max_memory, cpu_only=cpu_only,
    )

    # Run ALL THREE engines on every generation, measuring wall time per engine.
    RAW_K = 10
    engine_timings: dict[str, float] = {}
    with st.status("Running all three search engines…", expanded=False) as _s:
        _s.write("Beam Search — deterministic width-K expansion")
        _t = time.perf_counter()
        beam_results = beam_search(repo, task, cons, beam_width=beam_width, top_k=RAW_K)
        engine_timings["Beam Search"] = time.perf_counter() - _t

        _s.write("MCTS — Monte-Carlo Tree Search with UCT")
        _t = time.perf_counter()
        mcts_results = mcts_search(
            repo, task, cons,
            iterations=mcts_iterations, top_k=RAW_K,
            exploration=mcts_exploration,
        )
        engine_timings["Monte-Carlo Tree Search"] = time.perf_counter() - _t

        _s.write("Reinforcement Learning — tabular Q-learning")
        _t = time.perf_counter()
        rl_results = rl_search(repo, task, cons, episodes=rl_episodes, top_k=RAW_K)
        engine_timings["Reinforcement Learning"] = time.perf_counter() - _t

        _s.update(label="Search complete", state="complete", expanded=False)

    engine_runs = [
        ("Beam Search",             beam_results),
        ("Monte-Carlo Tree Search", mcts_results),
        ("Reinforcement Learning",  rl_results),
    ]

    # ------------------------------------------------------------------ helpers
    # Every candidate is scored by the same formula; per engine we bias
    # toward that engine's characteristic shape, then sort by score.
    def _has_parallel_reasoning(a):
        if a.layers is None:
            return False
        return any(
            sum(1 for c in layer if c.type == "reasoning") >= 2
            for layer in a.layers
        )

    def _has_parallel_validation(a):
        if a.layers is None:
            return False
        return any(
            sum(1 for c in layer if c.type == "validation") >= 2
            for layer in a.layers
        )

    def _wrap(x):
        return (
            wrap_in_agentic_scaffold(x, repo, cons=cons, keep_domain_output=True)
            if (wrap_agentic and task.domain != "agentic") else x
        )

    def _shape_menu_for(engine_name, chain):
        v = [chain]  # always include the linear form
        if allow_dag:
            if engine_name == "Beam Search":
                w_types = None
            elif engine_name == "Monte-Carlo Tree Search":
                w_types = {"reasoning"}
            else:
                w_types = {"validation"}
            if max_width >= 2:
                v.append(inflate_to_dag(chain, repo, task.domain, cons,
                                        max_width=2, widen_types=w_types))
            if max_width >= 3:
                v.append(inflate_to_dag(chain, repo, task.domain, cons,
                                        max_width=3, widen_types=w_types))
            if w_types is not None:
                if max_width >= 2:
                    v.append(inflate_to_dag(chain, repo, task.domain, cons,
                                            max_width=2, widen_types=None))
                if max_width >= 3:
                    v.append(inflate_to_dag(chain, repo, task.domain, cons,
                                            max_width=3, widen_types=None))
        return [_wrap(x) for x in v]

    def _bias_key(engine_name, a):
        # Each engine's characteristic shape becomes the tiebreaker that
        # pushes the "wrong-shape" variants down the list. Scores are
        # untouched — this is presentation, not scoring.
        #
        #   Beam  →  hierarchical (both reasoning AND validation widened)
        #             — the widest genuine ensemble Beam's width-K expansion
        #               can afford within the budget.
        #   MCTS  →  parallel reasoning ensemble.
        #   RL    →  parallel validation ensemble.
        if engine_name == "Beam Search":
            # Prefer a full hierarchy (both parallel reasoning AND parallel
            # validation), then any parallel structure, then linear.
            if _has_parallel_reasoning(a) and _has_parallel_validation(a):
                return 0
            if a.max_width >= 2:
                return 1
            return 2
        if engine_name == "Monte-Carlo Tree Search":
            return 0 if _has_parallel_reasoning(a) else 1
        return 0 if _has_parallel_validation(a) else 1

    # Pick the winning architecture per engine.
    engine_winners: list[tuple[str, object]] = []
    for engine_name, raw in engine_runs:
        expanded = []
        for chain in raw:
            for v in _shape_menu_for(engine_name, chain):
                expanded.append(v)
        if not expanded:
            continue
        unique = list({a.signature(): a for a in expanded}.values())
        unique.sort(key=lambda a: (_bias_key(engine_name, a), -a.score))
        engine_winners.append((engine_name, unique[0]))

    if not engine_winners:
        st.error("No architecture satisfies the current constraints. Try relaxing the sliders.")
        st.stop()

    # ---- Parsed task strip ----------------------------------------------------
    any_valid = any(is_valid_architecture(w) for _, w in engine_winners)
    st.markdown(
        '<div style="height:2.2rem;"></div>'
        '<div class="section-eyebrow">Result</div>'
        '<h3 class="section-h">Best architecture per search engine</h3>'
        '<p class="section-sub">'
        'Each engine returns the architecture that best expresses its algorithmic '
        'character. The card outlined in green is the highest-scoring overall.'
        '</p>'
        '<div class="kv-strip">'
        f'<div class="kv"><b>domain</b><code>{task.domain}</code></div>'
        f'<div class="kv"><b>intent</b><code>{task.intent}</code></div>'
        f'<div class="kv"><b>input</b><code>{task.input_type}</code></div>'
        f'<div class="kv"><b>output</b><code>{task.desired_output}</code></div>'
        + (f'<span class="badge badge-ok">valid ✓</span>' if any_valid
           else '<span class="badge badge-warn">structural check failed</span>')
        + '</div>',
        unsafe_allow_html=True,
    )

    # ---- One card per engine, side by side -----------------------------------
    best_score = max(w.score for _, w in engine_winners)

    cols = st.columns(len(engine_winners), gap="medium")
    for i, (engine_name, arch) in enumerate(engine_winners):
        meta = _ENGINE_META[engine_name]
        is_winner = arch.score == best_score
        elapsed_ms = engine_timings.get(engine_name, 0.0) * 1000.0
        card_class = "result-card winner" if is_winner else "result-card"
        crown_html = (
            '<span class="engine-crown">★ Best overall</span>' if is_winner else ""
        )
        with cols[i]:
            st.markdown(
                f'<div class="{card_class}" '
                f'style="--engine-color:{meta["color"]}; --engine-stripe:{meta["stripe"]};">'
                f'<div class="result-head">'
                f'<span class="engine-name">{meta["short"]}</span>'
                f'{crown_html}'
                f'</div>'
                f'<div class="engine-char"><b>{meta["trait"]}</b> · {meta["why"]}</div>'
                f'<div class="engine-score">'
                f'<span class="val">{arch.score:.3f}</span>'
                f'<span class="lbl">score</span>'
                f'<span class="time">{elapsed_ms:.0f} ms</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Render the architecture as a base64 PNG inside a
            # viewport-height-capped container so it never forces a scroll.
            img_url = render_architecture_data_url(arch)
            st.markdown(
                f'<div class="arch-wrap"><img src="{img_url}" alt="architecture"></div>'
                + _metric_list(arch, cons)
                + '</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    '<div class="footer">'
    'AutoAgentSearch · research MVP · '
    '<code>Beam · MCTS · RL</code> over a metadata-only component library.'
    '</div>',
    unsafe_allow_html=True,
)
