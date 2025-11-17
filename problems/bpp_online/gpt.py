import numpy as np

def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """Sigmoid-weighted scoring with probabilistic acceptance."""
    exact_fit_score = 5.0 * (bins_remain_cap == item)
    near_fit_range = max(0.2 * item, 0.1)
    near_fit_penalty = 1 / (1 + np.exp(-10*(bins_remain_cap - item)/near_fit_range))
    valid_bins = (bins_remain_cap >= item)
    scores = np.where(valid_bins, exact_fit_score + 2.0 * near_fit_penalty, -np.inf)
    return scores
