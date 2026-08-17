"""Common result dataclass for all comparison algorithms."""

from dataclasses import dataclass
import numpy as np


@dataclass
class OptResult:
    """Standardized result container compatible with experiment runner."""

    best_position: np.ndarray
    best_fitness: float
    convergence_curve: np.ndarray
    # NCRO-specific fields (None for other algorithms)
    exploration_counts: np.ndarray | None = None
    exploitation_counts: np.ndarray | None = None
    counterfactual_success_counts: np.ndarray | None = None
    regret_values: np.ndarray | None = None
    q_values: np.ndarray | None = None
