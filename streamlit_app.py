import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Quantum Annealing Portfolio", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚛️ Quantum Annealing Portfolio Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">QUBO formulation | D‑Wave Ocean SDK | Cardinality‑constrained mean‑variance | QAOA benchmark | Multi‑window selection</div>', unsafe_allow_html=True)

st.sidebar.markdown("## ⚛️ Quantum Annealing")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Risk aversion λ:** {config.RISK_AVERSION} | **Cardinality K:** {config.CARDINALITY}")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'quantum_annealing_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Quantum Portfolio Score")

with st.expander("📖 Interpretation", expanded=True):
    st.markdown("""
    - **QUBO formulation** converts the mean‑variance portfolio selection with cardinality constraint into a quadratic unconstrained binary optimisation problem.
    - The QUBO is solved using D‑Wave’s quantum annealer (or a classical simulated annealing fallback).
    - For a given window, the optimal set of `K` assets is chosen to maximise the portfolio return (or minimise risk).
    - The **score** for each ETF is the portfolio return if that ETF is selected, otherwise zero.
    - For each ETF, the rolling window that gives the **highest score** is selected.
    - The QAOA benchmark (not yet fully implemented) is included for comparison.
    """)

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">portfolio score = {etf['portfolio_score']:.6f}</div>
                <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    # Show selected assets for the best window
    win_res = uni_data.get("window_results", {})
    if win_res:
        best_win = top_etfs[0]['best_window'] if top_etfs else None
        if best_win is not None and str(best_win) in win_res:
            selected = win_res[str(best_win)].get("selected_assets", [])
            port_ret = win_res[str(best_win)].get("portfolio_return", 0.0)
            st.info(f"**Window {best_win}d:** Selected assets: {', '.join(selected)} → Portfolio return: {port_ret:.6f}")
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    score = info.get("score", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    score = info
                    win = "N/A"
                rows.append({"ETF": ticker, "Portfolio Score": score, "Best Window": win})
            df = pd.DataFrame(rows)
            df["Portfolio Score"] = pd.to_numeric(df["Portfolio Score"], errors='coerce')
            df = df.dropna(subset=["Portfolio Score"]).sort_values("Portfolio Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("The QUBO formulation uses D‑Wave’s quantum annealer (or simulated annealing) to select an optimal subset of K=5 assets. The score is the equal‑weighted return of the selected portfolio. Higher score → better portfolio → overweight those assets. For each ETF, the window that gives the highest score is selected.")
