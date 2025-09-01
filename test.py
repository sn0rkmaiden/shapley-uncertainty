import numpy as np
from utils.functions import *

def test_fixed_case(labels, S, likelihood, description):
    K = build_kernel(S)
    q = np.array(likelihood)
    def v_fn(S_idx): return set_function(S_idx, K, q=q, delta=1e-6)
    phi = shapley_values(len(labels), v_fn, M=500)
    w = np.maximum(phi - phi.min(), 0)
    p = w / (w.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    confgap = 1 - p.max()
    total_uncertainty = np.sum(phi) 
    print(f"\n{description}")
    print("Labels:", labels)
    print("Shapley values (phi):", phi)
    print("Total uncertainty:", total_uncertainty)
    print(f"Uncertainty (entropy): {entropy:.3f}, Confidence gap: {confgap:.3f}")

# Case 1: Beethoven (correlated)
labels1 = ["Wolfgang Amadeus Mozart", "Mozart", "Ludwig van Beethoven"]
S1 = np.array([[1.0, 1.0, 0.5],
               [1.0, 1.0, 0.5],
               [0.5, 0.5, 1.0]])
lik1 = [0.5, 0.4, 0.1]
test_fixed_case(labels1, S1, lik1, "Table 1")

# Case 2: da Vinci (uncorrelated)
labels2 = ["Wolfgang Amadeus Mozart", "Mozart", "Leonardo da Vinci"]
S2 = np.array([[1.0, 1.0, 0.0],
               [1.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]])
lik2 = [0.5, 0.4, 0.1]
test_fixed_case(labels2, S2, lik2, "Table 2")