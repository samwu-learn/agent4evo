import numpy as np
from scipy.spatial import distance_matrix
from copy import copy
from tqdm import tqdm
from gpt import select_next_node

def eval_heuristic(node_positions: np.ndarray, select_next_node) -> float:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.
    
    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).
    
    Returns
    -------
    obj : float
        The length of the generated tour.
    '''
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=unvisited,
            distance_matrix=dist_mat,
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj

print("===================Evaluating heuristic======================")
for size in [20, 50, 100, 200, 500, 1000]:
    # Load dataset
    X = np.load('./dataset/test{}_dataset.npy'.format(size))
    objs = []
    print("Evaluating heuristic for size {} with {} instances".format(size, len(X)))
    for node_positions in X:
        obj = eval_heuristic(node_positions, select_next_node)
        objs.append(obj)
    print('Average objective value for size {}: {}'.format(size, np.mean(objs)), "\n")