"""
EPEC Reference Implementation
Computes Efficiency–Precision Equilibrium Coefficient (EPEC) for time series forecasting models.
"""

import numpy as np

def epec(errors, avg_train_times, lamda=0.25):
    """
    Compute EPEC scores and rankings.

    Parameters
    ----------
    errors : np.ndarray, shape (n_models, n_tasks)
        Prediction error for each model (row) and each task (column).
    avg_train_times : np.ndarray, shape (n_models,)
        Average training time (seconds) for each model across all tasks.
    lamda : float, default=0.25
        Exponential decay parameter (higher = stronger penalty for accuracy loss).

    Returns
    -------
    epec_scores : np.ndarray, shape (n_models,)
        EPEC score for each model.
    rankings : np.ndarray, shape (n_models,)
        Rank of each model (1 = highest EPEC).
    """
    min_per_task = np.min(errors, axis=0, keepdims=True)
    min_per_task = np.where(min_per_task == 0, 1e-12, min_per_task)
    loss = (errors - min_per_task) / min_per_task
    weight = np.exp(-lamda * loss)
    avg_weight = np.mean(weight, axis=1)
    epec_scores = avg_weight / avg_train_times

    sorted_indices = np.argsort(epec_scores)[::-1]
    rankings = np.zeros_like(epec_scores, dtype=int)

    for rank, idx in enumerate(sorted_indices, start=1):
        rankings[idx] = rank

    return epec_scores, rankings
