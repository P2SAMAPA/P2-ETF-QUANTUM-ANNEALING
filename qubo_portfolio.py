import numpy as np
import dimod
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dimod import SimulatedAnnealingSampler
import config

def build_qubo(returns, lambda_risk=1.0, cardinality=5):
    """
    Build QUBO matrix for mean‑variance portfolio selection with cardinality constraint.
    Variables: x_i ∈ {0,1} (select asset i).
    Objective: maximize Σ μ_i x_i - λ Σ Σ σ_ij x_i x_j.
    Constraint: (Σ x_i - K)^2 = 0 (penalty encoded).
    Returns Q matrix (upper triangular) for dimod.
    """
    n = returns.shape[1]
    mu = returns.mean().values
    cov = returns.cov().values
    # QUBO: x^T A x + x^T b
    # A_ij = -λ * σ_ij (for i≠j) + penalty term: 2 * penalty * 1 (for i≠j)
    # b_i = μ_i - 2 * penalty * K
    # Diagonal: A_ii = -λ * σ_ii + penalty
    penalty = 1000.0  # large enough to enforce cardinality
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                Q[i,i] = -mu[i] + lambda_risk * cov[i,i] + penalty
            else:
                Q[i,j] = lambda_risk * cov[i,j] + 2 * penalty
    # Penalty term: -2 * penalty * K for linear part
    linear = -2 * penalty * cardinality * np.ones(n)
    # dimod expects dict
    qubo = {}
    for i in range(n):
        qubo[(i,i)] = Q[i,i]
        qubo[(i,i)] += linear[i]   # linear terms go on diagonal
    for i in range(n):
        for j in range(i+1, n):
            qubo[(i,j)] = Q[i,j]
    return qubo

def solve_qubo_quantum(qubo, sampler_type='hybrid', num_reads=100, annealing_time=20):
    if sampler_type == 'hybrid':
        sampler = LeapHybridSampler(token=config.TOKEN)
        sampleset = sampler.sample_qubo(qubo, time_limit=1)
    elif sampler_type == 'qpu':
        sampler = EmbeddingComposite(DWaveSampler(token=config.TOKEN))
        sampleset = sampler.sample_qubo(qubo, num_reads=num_reads, annealing_time=annealing_time)
    else:
        raise ValueError("sampler_type must be 'hybrid' or 'qpu'")
    # Extract best solution
    best_sample = sampleset.first.sample
    return best_sample

def solve_qubo_classical(qubo, num_reads=100):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo, num_reads=num_reads)
    best_sample = sampleset.first.sample
    return best_sample

def select_assets_from_sample(sample, asset_names, cardinality):
    """
    Extract selected assets from binary sample; if not enough selected, add top assets by return.
    """
    selected = [asset_names[i] for i, v in sample.items() if v == 1]
    if len(selected) < cardinality:
        # Fill remaining with highest mean return among not selected
        mu = sample_mean_returns(asset_names)
        not_selected = [a for a in asset_names if a not in selected]
        mu_dict = {a: mu[a] for a in not_selected}
        sorted_extra = sorted(mu_dict.items(), key=lambda x: x[1], reverse=True)[:cardinality - len(selected)]
        selected += [a for a, _ in sorted_extra]
    elif len(selected) > cardinality:
        # Keep only the top `cardinality` by mean return
        mu = sample_mean_returns(asset_names)
        selected = sorted(selected, key=lambda a: mu[a], reverse=True)[:cardinality]
    return selected

def compute_portfolio_return(selected_assets, returns_df, equal_weight=True):
    """Return average return of selected assets (equal weight)."""
    ret = returns_df[selected_assets].mean(axis=1).mean() if equal_weight else returns_df[selected_assets].mean(axis=1).mean()  # placeholder
    return ret
