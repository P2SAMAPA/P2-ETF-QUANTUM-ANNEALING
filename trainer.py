import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from qubo_portfolio import build_qubo, solve_qubo_quantum, solve_qubo_classical, select_assets_from_sample, compute_portfolio_return
from qaoa_benchmark import qaoa_optimize

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Quantum Annealing) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            ret_win = returns.iloc[-win:]
            mu = ret_win.mean().values
            cov = ret_win.cov().values
            # Build QUBO
            qubo = build_qubo(ret_win, lambda_risk=config.RISK_AVERSION, cardinality=config.CARDINALITY)
            # Solve (quantum if token available, else classical)
            if config.TOKEN and config.SAMPLER != "classical":
                try:
                    sample = solve_qubo_quantum(qubo, sampler_type=config.SAMPLER, num_reads=config.NUM_READS, annealing_time=config.ANNEALING_TIME)
                except Exception as e:
                    print(f"    Quantum solver failed: {e}, falling back to classical")
                    sample = solve_qubo_classical(qubo, num_reads=config.NUM_READS)
            else:
                sample = solve_qubo_classical(qubo, num_reads=config.NUM_READS)
            selected_assets = select_assets_from_sample(sample, tickers, config.CARDINALITY)
            # Compute portfolio return (score)
            port_return = compute_portfolio_return(selected_assets, ret_win, equal_weight=True)
            # For each ETF, we need a per‑ETF score. We'll assign the portfolio return to all selected ETFs, and 0 to others.
            scores = {etf: port_return if etf in selected_assets else 0.0 for etf in tickers}
            window_results[win] = {
                "selected_assets": selected_assets,
                "portfolio_return": port_return,
                "scores": scores
            }
            for etf, score in scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "portfolio_score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by quantum portfolio score: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/quantum_annealing_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Quantum Annealing Engine complete ===")

if __name__ == "__main__":
    main()
