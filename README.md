# Quantum Annealing Portfolio Engine

Uses D‑Wave’s quantum annealer (or classical simulated annealing) to solve cardinality‑constrained mean‑variance portfolio optimization as a QUBO problem. The engine selects a subset of K assets to maximise portfolio return. The score for each ETF is the portfolio return if selected, zero otherwise. Multi‑window evaluation picks the best window per ETF. A QAOA benchmark is included (placeholder).

- **QUBO formulation:** maximise μᵀx – λ xᵀΣx subject to Σxᵢ = K
- **Solver:** D‑Wave hybrid/QPU or simulated annealing fallback
- **Post‑processing:** ensures exactly K assets are selected
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by portfolio score

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
export DWAVE_TOKEN=<your_dwave_token>   # optional
python trainer.py
streamlit run streamlit_app.py
