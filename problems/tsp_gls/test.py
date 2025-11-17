import numpy as np
import logging
from gen_inst import TSPInstance, load_dataset
from gls import guided_local_search
import time
import elkai
from tqdm import tqdm

perturbation_moves_map = {
    20: 5,
    50: 30,
    100: 40,
    200: 40,
}
iter_limit_map = {
    20: 73,
    50: 175,
    100: 1800,
    200: 800,
}
SCALE = 1000000
test_sizes = list(iter_limit_map.keys())

def calculate_cost(inst: TSPInstance, path: np.ndarray):
    return inst.distmat[path, np.roll(path, 1)].sum().item()

optimal_objs_dict = {20: 3.8362853943492015, 50: 5.68457994395107, 100: 7.778580370400294, 200: 10.71194600194464}

def solve(inst: TSPInstance, heuristics):
    start_time = time.time()
    heu = heuristics(inst.distmat.copy())
    result = guided_local_search(inst.distmat, heu, perturbation_moves_map[inst.n], iter_limit_map[inst.n])
    duration = time.time() - start_time
    return calculate_cost(inst, result), duration

def evaluate(function):
    print("[*] Function:", function.__name__, "\n")
    for problem_size in iter_limit_map.keys():
        dataset_path = f"dataset/test{problem_size}_dataset.npy"
        dataset = load_dataset(dataset_path)
        logging.info(f"[*] Evaluating {dataset_path}")

        objs = []
        durations = []
        for instance in dataset:
            obj, duration = solve(instance, function)
            objs.append(obj)
            durations.append(duration)

        mean_obj = np.mean(objs).item()
        mean_optimal_obj = optimal_objs_dict[problem_size]
        gap = mean_obj / mean_optimal_obj - 1
        print(f"[*] Average for {problem_size}: {mean_obj:.6f} ({mean_optimal_obj:.6f})")
        print(f"[*] Optimality gap: {gap*100:.6f}%")
        print(f"[*] Total/Average duration: {sum(durations):.6f}s {sum(durations)/len(durations):.6f}s")
        print()

if __name__ == "__main__":
    from gpt import heuristics
    evaluate(heuristics)