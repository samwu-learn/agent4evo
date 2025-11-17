import numpy as np
from scipy.sparse.csgraph import shortest_path

def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    # Compute closeness centrality (inverse of average shortest path)
    shortest_paths = shortest_path(distance_matrix, directed=False)
    closeness = 1 / (np.mean(shortest_paths, axis=1) + 1e-10)
    # Compute betweenness centrality (approximation using degree)
    degree = np.sum(distance_matrix > 0, axis=1)
    betweenness = degree / np.max(degree)
    # Combine distance with centrality penalties
    heur_matrix = distance_matrix * (closeness[:, None] + closeness[None, :]) * (betweenness[:, None] + betweenness[None, :])
    # Normalize to [0,1] range
    heur_matrix = (heur_matrix - np.min(heur_matrix)) / (np.max(heur_matrix) - np.min(heur_matrix))
    return heur_matrix
