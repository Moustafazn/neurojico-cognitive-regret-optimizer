"""
Sine Cosine Algorithm (SCA).

Population-based metaheuristic that uses sine and cosine trigonometric
functions to create oscillatory search patterns for exploration and
exploitation.
References:
  - Mirjalili, S. (2016). "SCA: A Sine Cosine Algorithm for solving
    optimization problems." Knowledge-Based Systems, 96, 120-133.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class SCAOptimizer:
    """Sine Cosine Algorithm."""

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500,
                 a: float = 2.0, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.a = a  # constant for controlling search range
        self.seed = seed

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U = self.L, self.U

        # Initialize population
        X = rng.uniform(L, U, size=(N, D))
        fitness = np.array([self.func(X[i]) for i in range(N)])

        # Destination (best solution found)
        k = int(np.argmin(fitness))
        dest = X[k].copy()
        dest_fit = fitness[k]

        convergence = np.zeros(T)

        for t in range(T):
            # Linearly decreasing parameter r1
            r1 = self.a - self.a * t / T

            for i in range(N):
                r2 = rng.random(D) * 2 * np.pi  # random in [0, 2*pi]
                r3 = rng.random(D) * 2           # random weight in [0, 2]
                r4 = rng.random(D)               # sine or cosine selector

                # Update position using sine or cosine
                new_pos = np.where(
                    r4 < 0.5,
                    X[i] + r1 * np.sin(r2) * np.abs(r3 * dest - X[i]),
                    X[i] + r1 * np.cos(r2) * np.abs(r3 * dest - X[i]),
                )
                X[i] = np.clip(new_pos, L, U)
                fitness[i] = self.func(X[i])

            # Update destination
            k = int(np.argmin(fitness))
            if fitness[k] < dest_fit:
                dest = X[k].copy()
                dest_fit = fitness[k]

            convergence[t] = dest_fit

        return OptResult(best_position=dest, best_fitness=float(dest_fit),
                         convergence_curve=convergence)
