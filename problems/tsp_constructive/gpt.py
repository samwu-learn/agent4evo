import numpy as np
from collections import defaultdict

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """Select next node using alternating momentum-density strategy with progress-aware cluster scaling."""
    if not unvisited_nodes:
        raise ValueError("No unvisited nodes remaining")

    # Dynamic phase calculation combining momentum and density characteristics
    progress = 1 - len(unvisited_nodes)/len(distance_matrix)
    density_factor = min(1.0, np.mean([distance_matrix[current_node][n] for n in unvisited_nodes]) / np.mean(distance_matrix))
    
    # Phase-adaptive weights with momentum-density balance
    immediate_weight = 0.5 - (0.1 * progress) + (0.05 * density_factor)
    cluster_weight = 0.3 + (0.1 * progress) - (0.05 * density_factor)
    destination_weight = 0.2 + (0.05 * progress * density_factor)

    # Variable cluster scaling based on solution phase
    cluster_prox = defaultdict(float)
    min_cluster = max(2, int(np.sqrt(len(unvisited_nodes))))
    max_cluster = min(5, len(unvisited_nodes))
    dynamic_cluster_size = min(max_cluster, min_cluster + int(progress * 3))
    
    for node in unvisited_nodes:
        sorted_distances = sorted(distance_matrix[node, n] for n in unvisited_nodes if n != node)
        cluster_prox[node] = np.mean(sorted_distances[:dynamic_cluster_size])

    scores = {}
    for node in unvisited_nodes:
        immediate_dist = distance_matrix[current_node][node]
        destination_dist = distance_matrix[node][destination_node]
        path_continuity = abs(distance_matrix[current_node][destination_node] - distance_matrix[node][destination_node])

        # Combined scoring with momentum-density balance
        score = (immediate_weight * immediate_dist 
                - cluster_weight * cluster_prox[node] 
                - destination_weight * destination_dist 
                - 0.05 * path_continuity)
        scores[node] = score

    return min(scores, key=scores.get)
