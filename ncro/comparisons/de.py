"""
Differential Evolution (DE/rand/1/bin) — Classic baseline.

Standard DE with rand/1 mutation and binomial crossover.
References:
  - Storn, R. & Price, K. (1997). Differential Evolution. Journal of Global Optimization.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class DEOptimizer:
    """Standard Differential Evolution (DE/rand/1/bin)."""

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_iter: int = 1000,
        F: float = 0.5,
        CR: float = 0.9,
        seed: int | None = None,
        **kwargs,
    ):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.F = F
        self.CR = CR
        self.seed = seed

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U = self.L, self.U

        # Initialize
        X = rng.uniform(L, U, size=(N, D))
        fitness = np.array([self.func(X[i]) for i in range(N)])

        g_idx = int(np.argmin(fitness))
        g = X[g_idx].copy()
        g_fit = fitness[g_idx]

        convergence = np.zeros(T)

        for t in range(T):
            for i in range(N):
                # Mutation: DE/rand/1
                candidates = np.delete(np.arange(N), i)
                r = rng.choice(candidates, size=3, replace=False)
                mutant = X[r[0]] + self.F * (X[r[1]] - X[r[2]])
                mutant = np.clip(mutant, L, U)

                # Binomial crossover
                trial = X[i].copy()
                j_rand = rng.integers(D)
                for j in range(D):
                    if rng.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Selection
                f_trial = self.func(trial)
                if f_trial <= fitness[i]:
                    X[i] = trial
                    fitness[i] = f_trial

            g_idx = int(np.argmin(fitness))
            if fitness[g_idx] < g_fit:
                g = X[g_idx].copy()
                g_fit = fitness[g_idx]

            convergence[t] = g_fit

        return OptResult(
            best_position=g,
            best_fitness=g_fit,
            convergence_curve=convergence,
        )
