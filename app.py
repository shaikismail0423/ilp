"""AutoAgentSearch — Streamlit dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

from core.beam_search import beam_search
from core.mcts_search import mcts_search
from core.rl_search import rl_search
from core.benchmark import run_benchmark
from core.dag_inflater import inflate_to_dag
from core.agentic_wrapper import wrap_in_agentic_scaffold, is_valid_architecture
from core.explainer import explain_alternative, explain_best
from core.models import Constraints
from core.repository import ComponentRepository
from core.task_parser import parse_task
from visualization.graph_plot import render_architecture


# =============================================================================
# PAGE
# =============================================================================
st.set_page_config(
    page_title="AutoAgentSearch",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
REPO_PATH = os.path.join(os.path.dirname(__file__), "components")


@st.cache_resource
def get_repository() -> ComponentRepository:
    return ComponentRepository(REPO_PATH)


repo = get_repository()


# =============================================================================
# THEME  (Linear / Vercel-inspired: neutral palette + one accent)
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
    --bg:        #0A0A0A;      /* main canvas */
    --bg-elev:   #121215;      /* elevated cards */
    --panel:     #18181B;      /* sidebar */
    --panel-2:   #1F1F23;      /* subtle raised */
    --accent:    #6366F1;      /* indigo */
    --accent-2:  #8B5CF6;      /* violet */
    --success:   #10B981;
    --warn:      #F59E0B;
    --danger:    #EF4444;
  }

  /* Dark theme surface */
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

  /* Sidebar stays open on page load (initial_sidebar_state="expanded") */

  /* Font family only — never override colour globally, it breaks widgets */
  html, body, [class*="css"], [class*="st-"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI",
                 sans-serif !important;
  }
  .main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
    background-color: var(--bg) !important;
    color: var(--fg) !important;
  }
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

  h1, h2, h3, h4 { color: var(--fg-strong) !important; letter-spacing: -0.02em; }
  .stApp p, .stApp span, .stApp div, .stApp label {
    color: var(--fg);
  }

  /* Streamlit's widget labels default to a small, muted style */
  .stSlider label, .stCheckbox label, .stRadio label,
  .stSelectbox label, .stNumberInput label, .stTextArea label,
  .stTextInput label, .stButton label {
    font-size: 0.85rem !important;
    color: var(--fg) !important;
    font-weight: 500 !important;
  }
  /* Radio option labels + checkbox labels — a little larger */
  .stRadio div[role="radiogroup"] label p,
  .stCheckbox label p {
    font-size: 0.9rem !important;
    color: var(--fg) !important;
  }
  /* Slider tick / help text */
  .stSlider [data-baseweb="slider"] div,
  .stNumberInput [data-baseweb="input"] input {
    font-size: 0.88rem !important;
  }

  /* ---------- TOP BRAND BAR ---------- */
  .brand-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
  }
  .brand-name {
    font-size: 1.15rem; font-weight: 700; color: var(--fg-strong);
    letter-spacing: -0.02em;
    display: flex; align-items: center; gap: 0.6rem;
  }
  .brand-mark {
    width: 26px; height: 26px; border-radius: 7px;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 13px; font-weight: 800;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35);
  }
  .brand-tag {
    font-size: 0.82rem; color: var(--muted); font-weight: 500;
    padding: 4px 12px; border: 1px solid var(--border);
    border-radius: 999px; background: var(--panel);
  }

  /* ---------- HEADLINE ---------- */
  h1.headline, .headline {
    font-size: 2.6rem !important; font-weight: 800 !important;
    color: #09090B !important;
    letter-spacing: -0.035em !important; line-height: 1.1 !important;
    margin: 0 0 0.5rem 0 !important;
    display: block !important;
  }
  .lede {
    font-size: 1.05rem; color: var(--muted); line-height: 1.5;
    max-width: 680px; margin: 0 0 1.4rem 0;
  }

  /* ---------- SECTION LABEL ---------- */
  .section-eyebrow {
    font-size: 0.78rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 2.2rem 0 0.55rem 0;
  }
  .section-h {
    font-size: 1.5rem; font-weight: 700; color: var(--fg-strong);
    letter-spacing: -0.02em; margin: 0 0 0.5rem 0;
  }
  .section-sub {
    color: var(--muted); font-size: 0.98rem; line-height: 1.55;
    margin: 0 0 1.2rem 0; max-width: 680px;
  }
  .divider { border-top: 1px solid var(--border); margin: 2.4rem 0 0 0; }

  /* ---------- CARDS ---------- */
  .card {
    background: var(--bg-elev); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.15rem 1.3rem;
  }
  .card-panel { background: var(--panel); }

  /* ---------- METRIC ROW ---------- */
  .m-row {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 0; border: 1px solid var(--border); border-radius: 12px;
    background: var(--bg-elev); overflow: hidden;
  }
  .m-cell {
    padding: 1.05rem 1.1rem;
    border-right: 1px solid var(--border);
  }
  .m-cell:last-child { border-right: none; }
  .m-label {
    font-size: 0.78rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
  }
  .m-value {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 1.55rem; font-weight: 600; color: var(--fg-strong);
    letter-spacing: -0.02em; line-height: 1;
  }
  .m-hint {
    font-size: 0.78rem; color: var(--subtle); margin-top: 0.4rem;
  }

  /* ---------- KEY-VALUE STRIP ---------- */
  .kv-strip {
    display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0 0.2rem 0;
    align-items: center;
  }
  .kv {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 0.85rem;
    padding: 5px 12px; border: 1px solid var(--border);
    border-radius: 6px; background: var(--bg-elev);
    color: var(--fg);
  }
  .kv b { color: var(--muted); font-weight: 500; font-size: 0.82rem; }
  .kv code {
    font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.82rem;
    color: var(--fg-strong); font-weight: 500;
    background: transparent; padding: 0;
  }

  /* ---------- PIPELINE PILLS ---------- */
  .pipe {
    display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center;
    padding: 0.6rem 0;
  }
  .pipe-node {
    font-size: 0.85rem; padding: 5px 11px;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--bg-elev); color: var(--fg-strong); font-weight: 500;
  }
  .pipe-parallel {
    background: rgba(245,158,11,0.12);
    border-color: rgba(245,158,11,0.40);
    color: #FBBF24 !important;
  }
  .pipe-arrow { color: var(--subtle); font-size: 1rem; font-weight: 600; }

  /* ---------- WHY CARD ---------- */
  .why {
    background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(139,92,246,0.06) 100%);
    border: 1px solid rgba(99,102,241,0.28);
    border-left: 3px solid var(--accent);
    border-radius: 8px; padding: 1rem 1.2rem;
    font-size: 0.96rem; color: var(--fg); line-height: 1.6;
  }
  .why-label {
    font-size: 0.76rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.09em;
    margin-bottom: 0.4rem;
  }

  /* ---------- BADGE ---------- */
  .badge {
    display: inline-block;
    padding: 4px 10px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.01em;
    border: 1px solid transparent;
  }
  .badge-ok    { background: rgba(16,185,129,0.14); color: #6EE7B7; border-color: rgba(16,185,129,0.35); }
  .badge-warn  { background: rgba(245,158,11,0.14); color: #FCD34D; border-color: rgba(245,158,11,0.35); }
  .badge-muted { background: var(--panel-2); color: var(--muted); border-color: var(--border); }
  .badge-info  { background: rgba(99,102,241,0.14); color: #A5B4FC; border-color: rgba(99,102,241,0.35); }

  /* ---------- ENGINE COMPARISON CARDS ---------- */
  .eng {
    border: 1px solid var(--border); border-radius: 12px;
    padding: 1.15rem 1.25rem; background: var(--bg-elev); height: 100%;
    position: relative; overflow: hidden;
  }
  .eng::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--eng-color, var(--accent));
  }
  .eng-beam::before { background: linear-gradient(90deg, #3B82F6, #6366F1); }
  .eng-mcts::before { background: linear-gradient(90deg, #8B5CF6, #EC4899); }
  .eng-rl::before   { background: linear-gradient(90deg, #10B981, #14B8A6); }
  .eng-name {
    font-size: 0.88rem; font-weight: 700; color: var(--fg-strong);
    letter-spacing: -0.01em; margin-bottom: 0.85rem;
    display: flex; align-items: center; justify-content: space-between;
  }
  .eng-mark {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
  }
  .eng-mark-beam { background: #6366F1; box-shadow: 0 0 8px rgba(99,102,241,0.6); }
  .eng-mark-mcts { background: #A855F7; box-shadow: 0 0 8px rgba(168,85,247,0.6); }
  .eng-mark-rl   { background: #10B981; box-shadow: 0 0 8px rgba(16,185,129,0.6); }
  .eng-big {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 2.3rem; font-weight: 600; color: var(--fg-strong);
    letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.25rem;
  }
  .eng-big-label {
    font-size: 0.82rem; color: var(--muted); margin-bottom: 0.9rem;
  }
  .eng-sub {
    display: grid; grid-template-columns: repeat(3, 1fr);
    padding-top: 0.75rem; border-top: 1px solid var(--border-2);
    font-size: 0.85rem; color: var(--fg);
  }
  .eng-sub b { display: block; font-family: 'JetBrains Mono', ui-monospace, monospace;
              font-size: 0.95rem;
              color: var(--fg-strong); font-weight: 600; margin-bottom: 3px; }
  .eng-sub span { color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
                  letter-spacing: 0.06em; font-weight: 600; }
  .eng-example {
    margin-top: 0.95rem; padding-top: 0.8rem;
    border-top: 1px solid var(--border-2);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.78rem; color: var(--muted); line-height: 1.55;
  }
  .eng-example b { color: var(--fg-strong); font-weight: 700;
    font-family: 'Inter', sans-serif; text-transform: uppercase;
    font-size: 0.7rem; letter-spacing: 0.08em; display: block; margin-bottom: 4px; }

  /* ---------- TABLE ---------- */
  .tbl { width: 100%; border-collapse: collapse; }
  .tbl thead th {
    text-align: left; padding: 11px 16px;
    font-size: 0.75rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  .tbl tbody td {
    padding: 13px 16px; font-size: 0.94rem;
    border-bottom: 1px solid var(--border);
    color: var(--fg);
  }
  .tbl tbody tr:last-child td { border-bottom: none; }
  .tbl-mono { font-family: 'JetBrains Mono', ui-monospace, monospace; font-weight: 500; }

  /* ---------- BUTTONS ---------- */
  .stButton > button {
    border-radius: 8px !important;
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
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
  }
  .stButton > button[kind="primary"] * { color: white !important; }
  .stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
  }

  /* ---------- SIDEBAR ---------- */
  /* Force the sidebar to always be visible, no matter what state the
     browser has cached from a previous session. */
  section[data-testid="stSidebar"],
  [data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 260px !important;
    max-width: 260px !important;
    width: 260px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
    margin-left: 0 !important;
    left: 0 !important;
  }
  section[data-testid="stSidebar"] > div,
  section[data-testid="stSidebar"] > div:first-child {
    background-color: var(--panel) !important;
    width: 260px !important;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

  /* Push the main content over so the sidebar doesn't overlap it */
  [data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"]) {
    margin-left: 260px !important;
  }

  .side-brand {
    font-size: 1rem; font-weight: 700; color: var(--fg-strong);
    letter-spacing: -0.02em; margin-bottom: 0.35rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .side-brand-mark {
    width: 22px; height: 22px; border-radius: 6px;
    background: var(--fg-strong);
    display: inline-flex; align-items: center; justify-content: center;
    color: white; font-size: 11px; font-weight: 800;
  }
  .side-caption {
    font-size: 0.78rem; color: var(--muted); margin-bottom: 1.4rem;
  }
  .side-group {
    font-size: 0.72rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.09em;
    margin: 1.4rem 0 0.5rem 0;
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
  }

  /* ---------- EXPANDER ---------- */
  .streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: var(--panel) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    color: var(--fg-strong) !important;
    font-weight: 500 !important;
  }

  /* ---------- Set matplotlib figures with dark bg blend ---------- */
  [data-testid="stImage"] > div,
  .stPlotlyChart, .stPyplot { background: transparent !important; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        '<div class="side-brand"><span class="side-brand-mark">◆</span> AutoAgentSearch</div>'
        '<div class="side-caption">v3 · Research prototype</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-group">Task Domain</div>', unsafe_allow_html=True)
    _user_domains = [d for d in repo.domains() if d != "agentic"]
    domain = st.selectbox(
        "Domain",
        ["auto"] + _user_domains,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-group">Search Engine</div>', unsafe_allow_html=True)
    engine = st.radio(
        "Engine",
        ["Beam Search", "Monte-Carlo Tree Search", "Reinforcement Learning"],
        index=0,
        label_visibility="collapsed",
    )
    beam_width = mcts_iterations = mcts_exploration = rl_episodes = None
    if engine == "Beam Search":
        beam_width = st.slider("Beam width", 2, 10, 5, 1)
    elif engine == "Monte-Carlo Tree Search":
        mcts_iterations = st.slider("MCTS iterations", 50, 2000, 400, 50)
        mcts_exploration = st.slider("Exploration constant", 0.1, 3.0, 1.41, 0.1)
    else:
        rl_episodes = st.slider("RL training episodes", 50, 2000, 300, 50)

    st.markdown('<div class="side-group">Architecture</div>', unsafe_allow_html=True)
    allow_dag = st.checkbox("Parallel & hierarchical structures", value=True)
    max_width = st.slider("Max branch width", 2, 5, 3, 1) if allow_dag else 1
    wrap_agentic = st.checkbox("Wrap in agentic orchestration", value=True)

    st.markdown('<div class="side-group">Resource Constraints</div>', unsafe_allow_html=True)
    max_cost    = st.slider("Max cost per run (USD)", 0.001, 0.10, 0.05, 0.001, format="$%.3f")
    max_latency = st.slider("Max latency (seconds)", 0.5, 10.0, 5.0, 0.1)
    max_memory  = st.slider("Max memory (MB)", 256, 16384, 8192, 128)
    cpu_only    = st.checkbox("CPU-only deployment", value=False)


# =============================================================================
# BRAND BAR + HEADLINE
# =============================================================================
st.markdown(
    """
<div class="brand-bar">
  <div class="brand-name"><span class="brand-mark">◆</span> AutoAgentSearch</div>
  <div style="display:flex; gap:0.5rem; align-items:center;">
    <span class="brand-tag">TCS Research Prototype</span>
    <span class="brand-tag">v3</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# TASK INPUT
# =============================================================================
st.markdown(
    '<div class="section-eyebrow" style="margin-top:1.5rem;">Task</div>',
    unsafe_allow_html=True,
)

if "task_text" not in st.session_state:
    st.session_state.task_text = ""

c1, c2, _ = st.columns([1.5, 1.5, 5])
with c1:
    if st.button("Use healthcare example", use_container_width=True):
        st.session_state.task_text = (
            "Extract diagnosis and medications from patient discharge summary PDF."
        )
with c2:
    if st.button("Use finance example", use_container_width=True):
        st.session_state.task_text = (
            "Detect fraud in a scanned loan application document and produce a risk report."
        )

task_text = st.text_area(
    "Task",
    value=st.session_state.task_text,
    height=100,
    placeholder="e.g. Extract ICD codes from clinical notes and validate them.",
    label_visibility="collapsed",
)

run = st.button("Generate architecture  →", type="primary")


# =============================================================================
# HELPERS
# =============================================================================
def _mem_str(mb: int) -> str:
    return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb} MB"


def _m_cell(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="m-hint">{hint}</div>' if hint else ""
    return f'<div class="m-cell"><div class="m-label">{label}</div><div class="m-value">{value}</div>{hint_html}</div>'


def _metric_row(arch, cons):
    st.markdown(
        '<div class="m-row">'
        + _m_cell("Quality", f"{arch.aggregate_quality*100:.1f}%", "joint probability")
        + _m_cell("Cost",    f"${arch.total_cost:.4f}", f"budget ${cons.max_cost:.3f}")
        + _m_cell("Latency", f"{arch.total_latency:.2f}s", f"ceiling {cons.max_latency:.1f}s")
        + _m_cell("Memory",  _mem_str(arch.peak_memory), f"ceiling {_mem_str(cons.max_memory)}")
        + _m_cell("Score",   f"{arch.score:.3f}", "weighted overall")
        + "</div>",
        unsafe_allow_html=True,
    )


def _pipeline_pills(arch) -> str:
    parts = []
    if arch.layers is not None:
        for i, layer in enumerate(arch.layers):
            if len(layer) > 1:
                parts.append(
                    '<div class="pipe-node pipe-parallel">'
                    + " · ".join(c.name for c in layer) + "</div>"
                )
            else:
                parts.append(f'<div class="pipe-node">{layer[0].name}</div>')
            if i < len(arch.layers) - 1:
                parts.append('<span class="pipe-arrow">→</span>')
    else:
        for i, c in enumerate(arch.components):
            parts.append(f'<div class="pipe-node">{c.name}</div>')
            if i < len(arch.components) - 1:
                parts.append('<span class="pipe-arrow">→</span>')
    return f'<div class="pipe">{"".join(parts)}</div>'


def _sig_badge(p: float) -> str:
    if p != p:
        return '<span class="badge badge-muted">n/a</span>'
    if p < 0.05:
        return f'<span class="badge badge-ok">p={p:.3f}</span>'
    if p < 0.15:
        return f'<span class="badge badge-warn">p={p:.3f}</span>'
    return f'<span class="badge badge-muted">p={p:.3f}</span>'


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

    if engine == "Beam Search":
        with st.status(f"Running Beam Search — width {beam_width}", expanded=False):
            results = beam_search(repo, task, cons, beam_width=beam_width, top_k=3)
        engine_label = f"Beam Search · width {beam_width}"
    elif engine == "Monte-Carlo Tree Search":
        with st.status(f"Running MCTS — {mcts_iterations} iterations", expanded=False):
            results = mcts_search(
                repo, task, cons,
                iterations=mcts_iterations, top_k=3,
                exploration=mcts_exploration,
            )
        engine_label = f"MCTS · {mcts_iterations} iterations"
    else:
        with st.status(f"Training RL policy — {rl_episodes} episodes", expanded=False):
            results = rl_search(repo, task, cons, episodes=rl_episodes, top_k=3)
        engine_label = f"Reinforcement Learning · {rl_episodes} episodes"

    if allow_dag and max_width > 1:
        results = [inflate_to_dag(a, repo, task.domain, cons, max_width=max_width) for a in results]
    if wrap_agentic and task.domain != "agentic":
        results = [wrap_in_agentic_scaffold(a, repo, cons=cons, keep_domain_output=True) for a in results]

    if not results:
        st.error("No architecture satisfies the current constraints. Try relaxing the sliders.")
        st.stop()

    best = results[0]
    valid = is_valid_architecture(best)

    # ---- parsed task strip
    st.markdown(
        '<div style="margin-top:1.4rem;"></div>'
        '<div class="kv-strip">'
        f'<div class="kv"><b>domain</b><code>{task.domain}</code></div>'
        f'<div class="kv"><b>intent</b><code>{task.intent}</code></div>'
        f'<div class="kv"><b>input</b><code>{task.input_type}</code></div>'
        f'<div class="kv"><b>output</b><code>{task.desired_output}</code></div>'
        f'<div class="kv"><b>engine</b><code>{engine_label}</code></div>'
        + (f'<span class="badge badge-ok">valid ✓</span>' if valid
           else '<span class="badge badge-warn">structural check failed</span>')
        + '</div>',
        unsafe_allow_html=True,
    )

    # ---- best architecture
    st.markdown(
        '<div class="section-eyebrow">Result</div>'
        f'<h3 class="section-h">Best architecture &nbsp;·&nbsp; '
        f'<span style="color:#71717A;font-weight:500;">{best.n_layers} layers · width {best.max_width}</span></h3>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.pyplot(render_architecture(best), use_container_width=True)
    with right:
        _metric_row(best, cons)
        st.markdown(
            f'<div style="height:1rem;"></div>'
            f'<div class="why">'
            f'<div class="why-label">Why this architecture</div>{explain_best(best, task, cons)}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_pipeline_pills(best), unsafe_allow_html=True)

    # ---- alternatives
    if len(results) > 1:
        st.markdown(
            '<div class="section-eyebrow">Alternatives</div>'
            '<h3 class="section-h">Runner-ups</h3>',
            unsafe_allow_html=True,
        )
        for i, alt in enumerate(results[1:], start=1):
            with st.expander(
                f"#{i}   ·   Score {alt.score:.3f}   ·   {alt.n_layers} layers",
                expanded=False,
            ):
                al, ar = st.columns([1, 1])
                with al:
                    st.pyplot(render_architecture(alt), use_container_width=True)
                with ar:
                    _metric_row(alt, cons)
                    st.markdown(
                        f'<div style="height:0.8rem;"></div>'
                        f'<div class="why">'
                        f'<div class="why-label">Why it ranked lower</div>{explain_alternative(best, alt)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(_pipeline_pills(alt), unsafe_allow_html=True)

# =============================================================================
# BENCHMARK
# =============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-eyebrow">Benchmark</div>'
    '<h3 class="section-h">Beam vs MCTS vs RL</h3>',
    unsafe_allow_html=True,
)

bc1, bc2, bc3, bc4, bc5 = st.columns([1, 1, 1, 1, 1.4])
with bc1:
    bench_tasks = st.number_input("Tasks", 5, 100, 20, 5)
with bc2:
    bench_repeats = st.number_input("Repeats", 3, 50, 10, 1)
with bc3:
    bench_mcts_iter = st.number_input("MCTS iterations", 50, 2000, 400, 50)
with bc4:
    bench_rl_ep = st.number_input("RL episodes", 50, 2000, 300, 50)
with bc5:
    run_bench = st.button("Run benchmark  →", use_container_width=True, type="primary")

if run_bench:
    bench_cons = Constraints(max_cost=0.05, max_latency=5.0, max_memory=8192, cpu_only=False)
    with st.status(
        f"Running {bench_tasks} tasks × {bench_repeats} repeats across three engines...",
        expanded=True,
    ) as status:
        report = run_benchmark(
            repo, bench_cons,
            n_tasks=int(bench_tasks),
            repeats=int(bench_repeats),
            beam_width=5,
            mcts_iterations=int(bench_mcts_iter),
            rl_episodes=int(bench_rl_ep),
        )
        status.update(label="Benchmark complete", state="complete")

    # ---- engine cards row
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    mark_class = {
        "Beam Search":            "eng-mark-beam",
        "MCTS":                   "eng-mark-mcts",
        "Reinforcement Learning": "eng-mark-rl",
    }
    cols = st.columns(3)
    for col, (name, s) in zip(cols, report.scores.items()):
        mark = mark_class.get(name, "eng-mark-beam")
        example = " → ".join(s.example_pipeline[:6]) + (" …" if len(s.example_pipeline) > 6 else "")
        with col:
            st.markdown(
                '<div class="eng">'
                f'<div class="eng-name"><span>{name}</span>'
                f'<span class="eng-mark {mark}"></span></div>'
                f'<div class="eng-big">{s.success_rate*100:.1f}%</div>'
                f'<div class="eng-big-label">Task success rate</div>'
                f'<div class="eng-sub">'
                f'<div><b>{s.latency:.2f}s</b><span>latency</span></div>'
                f'<div><b>{s.complexity}</b><span>agents</span></div>'
                f'<div><b>{s.recovery_rate*100:.0f}%</b><span>recovery</span></div>'
                f'</div>'
                f'<div class="eng-example"><b>Example</b><br/>{example}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ---- pairwise significance
    st.markdown(
        '<div style="height:1.6rem;"></div>'
        '<div class="section-eyebrow">Significance</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for pair, st_ in report.pairwise_stats.items():
        d = st_["mean_delta"] * 100
        d_col = "#16A34A" if d > 0 else ("#DC2626" if d < 0 else "#71717A")
        rows.append(
            f'<tr>'
            f'<td>{pair}</td>'
            f'<td class="tbl-mono" style="color:{d_col};">{d:+.1f} pp</td>'
            f'<td>{_sig_badge(st_["t_p"])}</td>'
            f'<td>{_sig_badge(st_["wilcoxon_p"])}</td>'
            f'</tr>'
        )
    st.markdown(
        '<div class="card" style="padding:0;">'
        '<table class="tbl">'
        '<thead><tr><th>Comparison</th><th>Δ Success</th><th>Paired t-test</th><th>Wilcoxon</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    # ---- winner
    winner = max(report.scores.values(), key=lambda s: s.success_rate)
    st.markdown(
        '<div style="height:1.4rem;"></div>'
        '<div class="card" style="background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 60%,#EC4899 100%); '
        'border:none; color:white; box-shadow:0 20px 40px -12px rgba(99,102,241,0.35);">'
        '<div style="font-size:0.75rem; font-weight:700; color:rgba(255,255,255,0.85); text-transform:uppercase; letter-spacing:0.09em; margin-bottom:0.4rem;">Benchmark winner</div>'
        f'<div style="font-size:1.65rem; font-weight:700; letter-spacing:-0.02em; margin-bottom:0.4rem;">{winner.name}</div>'
        f'<div style="font-size:0.95rem; color:rgba(255,255,255,0.92); line-height:1.55;">'
        f'Task-success rate <b style="color:#fff;">{winner.success_rate*100:.1f}%</b>, '
        f'recovery rate <b style="color:#fff;">{winner.recovery_rate*100:.1f}%</b> — across '
        f'{report.n_tasks} tasks × {report.n_repeats} Monte-Carlo repeats.'
        f'</div></div>',
        unsafe_allow_html=True,
    )
