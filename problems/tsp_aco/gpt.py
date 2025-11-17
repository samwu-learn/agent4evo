import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Combines inverse distance, node centrality, and sparsification to prioritize high-potential edges."""
    # Inverse distance as base heuristic
    inv_dist = 1 / (distance_matrix + 1e-10)  # Avoid division by zero
    
    # Node centrality (degree proxy: sum of inverse distances per node)
    centrality = np.sum(inv_dist, axis=1, keepdims=True)
    
    # Combined heuristic: inverse distance multiplied by centrality scores
    heuristic_matrix = inv_dist * centrality.T  # Outer product for edge importance
    
    # Sparsify: keep only top 20% of edges per node
    threshold = np.percentile(heuristic_matrix, 80, axis=1, keepdims=True)
    heuristic_matrix = np.where(heuristic_matrix >= threshold, heuristic_matrix, 0)
    
    return heuristic_matrix
