"""
Grey Wolf Optimizer (GWO).

Nature-inspired metaheuristic based on grey wolf social hierarchy.
References:
  - Mirjalili, S. et al. (2014). "Grey Wolf Optimizer." Advances in
    Engineering Software, 69, 46-61.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class GWOOptimizer:
    """Grey Wolf Optimizer."""

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.seed = seed

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U = self.L, self.U

        X = rng.uniform(L, U, size=(N, D))
        fitness = np.array([self.func(X[i]) for i in range(N)])

        # Track global best across all iterations
        g_idx = int(np.argmin(fitness))
        g_best = X[g_idx].copy()
        g_fit = fitness[g_idx]

        convergence = np.zeros(T)

        for t in range(T):
            order = np.argsort(fitness)
            Alpha, Beta, Delta = X[order[0]], X[order[1]], X[order[2]]

            a = 2 - 2 * t / max(1, T - 1)

            Y = np.zeros_like(X)
            for i in range(N):
                parts = []
                for leader in (Alpha, Beta, Delta):
                    r1 = rng.random(D)
                    r2 = rng.random(D)
                    A_coeff = 2 * a * r1 - a
                    C_coeff = 2 * r2
                    dist = np.abs(C_coeff * leader - X[i])
                    parts.append(leader - A_coeff * dist)
                Y[i] = np.mean(parts, axis=0)

            X = np.clip(Y, L, U)
            fitness = np.array([self.func(X[i]) for i in range(N)])

            # Update global best (not just iteration best)
            k = int(np.argmin(fitness))
            if fitness[k] < g_fit:
                g_best = X[k].copy()
                g_fit = fitness[k]

            convergence[t] = g_fit

        return OptResult(best_position=g_best, best_fitness=float(g_fit),
                         convergence_curve=convergence)
