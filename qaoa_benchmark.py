"""
QAOA benchmark for portfolio QUBO.
Requires Qiskit. If not available, falls back to classical simulated annealing.
"""

import numpy as np
from dimod import SimulatedAnnealingSampler

try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.circuit import ParameterVector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Warning: Qiskit not installed. QAOA will use classical fallback.")

def build_qaoa_circuit(qubo, p=1):
    """
    Build QAOA circuit for the given QUBO.
    This is a simplified version: assumes one‑qubit cost terms and ZZ interactions.
    For a general QUBO, one would need to map to Ising model.
    Returns a parameterized circuit and the list of parameters.
    """
    n = len(qubo)
    beta = ParameterVector('β', p)
    gamma = ParameterVector('γ', p)
    qc = QuantumCircuit(n)
    # Initial state: Hadamard on all qubits
    qc.h(range(n))
    for layer in range(p):
        # Cost Hamiltonian: for each qubit, apply RZ(2*γ) based on linear coefficient
        # For simplicity, we only include self terms; ignoring ZZ interactions.
        # A full implementation would map QUBO to Ising.
        for i in range(n):
            qc.rz(2 * gamma[layer], i)
        # Mixer: RX(2*β)
        for i in range(n):
            qc.rx(2 * beta[layer], i)
    return qc, beta, gamma

def evaluate_qaoa(qubo, params, p):
    """Evaluate QAOA energy for given parameters."""
    n = len(qubo)
    qc, beta, gamma = build_qaoa_circuit(qubo, p)
    # Bind parameters
    param_dict = {beta[i]: params[i] for i in range(p)}
    param_dict.update({gamma[i]: params[i+p] for i in range(p)})
    bound_qc = qc.bind_parameters(param_dict)
    backend = Aer.get_backend('statevector_simulator')
    result = execute(bound_qc, backend).result()
    statevector = result.get_statevector()
    # Compute expected value of Hamiltonian: sum_i a_i <Z_i> + sum_{i<j} b_ij <Z_i Z_j>
    # Placeholder: return 0.0 (full implementation requires computing Pauli expectations)
    return 0.0

def qaoa_optimize(qubo, p=1, maxiter=50):
    """
    Optimize QAOA parameters to minimize energy.
    If Qiskit is not available, uses classical simulated annealing as fallback.
    Returns a binary solution (dictionary) and the energy.
    """
    n = len(qubo)
    if not HAS_QISKIT:
        # Fallback to simulated annealing
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(qubo, num_reads=100)
        best_sample = sampleset.first.sample
        energy = sampleset.first.energy
        return best_sample, energy
    # QAOA optimization
    from scipy.optimize import minimize
    def objective(params):
        return evaluate_qaoa(qubo, params, p)
    initial_params = np.random.uniform(0, np.pi, 2*p)
    res = minimize(objective, initial_params, method='COBYLA', options={'maxiter': maxiter})
    best_params = res.x
    # Build final circuit with optimal params and sample
    qc, beta, gamma = build_qaoa_circuit(qubo, p)
    param_dict = {beta[i]: best_params[i] for i in range(p)}
    param_dict.update({gamma[i]: best_params[i+p] for i in range(p)})
    bound_qc = qc.bind_parameters(param_dict)
    backend = Aer.get_backend('qasm_simulator')
    shots = 1024
    result = execute(bound_qc, backend, shots=shots).result()
    counts = result.get_counts()
    # Choose the most frequent bitstring
    best_bitstring = max(counts, key=counts.get)
    best_sample = {i: int(best_bitstring[i]) for i in range(n)}
    energy = evaluate_qaoa(qubo, best_params, p)  # not accurate
    return best_sample, energy
