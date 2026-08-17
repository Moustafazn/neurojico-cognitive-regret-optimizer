"""
Whale Optimization Algorithm (WOA).

Nature-inspired metaheuristic based on humpback whale bubble-net hunting.
References:
  - Mirjalili, S. & Lewis, A. (2016). "The Whale Optimization Algorithm."
    Advances in Engineering Software, 95, 51-67.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class WOAOptimizer:
    """Whale Optimization Algorithm."""

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
        k = int(np.argmin(fitness))
        G = X[k].copy()
        GF = fitness[k]
        convergence = np.zeros(T)

        for t in range(T):
            a = 2 - 2 * t / max(1, T - 1)
            Y = X.copy()

            for i in range(N):
                r1 = rng.random()
                r2 = rng.random()
                A = 2 * a * r1 - a
                Cc = 2 * r2
                p = rng.random()
                l = rng.uniform(-1, 1)

                if p < 0.5:
                    if abs(A) < 1:
                        dist = np.abs(Cc * G - X[i])
                        Y[i] = G - A * dist
                    else:
                        j = rng.integers(N)
                        xr = X[j]
                        dist = np.abs(Cc * xr - X[i])
                        Y[i] = xr - A * dist
                else:
                    dist = np.abs(G - X[i])
                    Y[i] = dist * np.exp(l) * np.cos(2 * np.pi * l) + G

            X = np.clip(Y, L, U)
            fitness = np.array([self.func(X[i]) for i in range(N)])
            k = int(np.argmin(fitness))
            if fitness[k] < GF:
                G = X[k].copy()
                GF = fitness[k]
            convergence[t] = GF

        return OptResult(best_position=G, best_fitness=float(GF),
                         convergence_curve=convergence)
