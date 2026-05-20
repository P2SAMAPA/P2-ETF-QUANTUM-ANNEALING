import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-quantum-annealing-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Rolling windows (days)
WINDOWS = [63, 252, 504, 1008, 2016]

# Portfolio parameters
RISK_AVERSION = 1.0      # λ in QUBO: maximize μ - λ Σ
CARDINALITY = 5          # target number of assets to hold

# D‑Wave quantum annealer parameters
TOKEN = os.environ.get("DWAVE_TOKEN", "")
SAMPLER = "hybrid"       # "hybrid" or "qpu" (requires D‑Wave access)
NUM_READS = 100
ANNEALING_TIME = 20      # microseconds

# Classical benchmark: Simulated Annealing (fallback if no D‑Wave)
USE_CLASSICAL_FALLBACK = True

TOP_N = 3
