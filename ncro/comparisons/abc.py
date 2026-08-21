"""
Artificial Bee Colony (ABC) Algorithm.

Swarm intelligence metaheuristic inspired by the foraging behavior of
honey bee colonies (employed bees, onlooker bees, scout bees).
References:
  - Karaboga, D. (2005). "An idea based on honey bee swarm for numerical
    optimization." Technical Report TR06, Erciyes University.
  - Karaboga, D. & Basturk, B. (2007). "A powerful and efficient algorithm
    for numerical function optimization: artificial bee colony (ABC)
    algorithm." Journal of Global Optimization, 39(3), 459-471.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class ABCOptimizer:
    """Artificial Bee Colony Optimization."""

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500,
                 limit: int | None = None, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        # Limit for abandonment: standard recommendation is N*D
        self.limit = limit if limit is not None else population_size * dimension
        self.seed = seed

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U = self.L, self.U
        SN = N // 2  # number of food sources (half the colony)

        # Initialize food sources
        X = rng.uniform(L, U, size=(SN, D))
        fitness = np.array([self.func(X[i]) for i in range(SN)])
        trial = np.zeros(SN, dtype=int)  # abandonment counter

        # Global best
        k = int(np.argmin(fitness))
        g_best = X[k].copy()
        g_fit = fitness[k]

        convergence = np.zeros(T)

        for t in range(T):
            # ── Employed bee phase ──
            for i in range(SN):
                # Pick a random partner different from i
                partner = rng.integers(SN - 1)
                if partner >= i:
                    partner += 1

                # Pick a random dimension
                j = rng.integers(D)

                # Generate candidate solution
                v = X[i].copy()
                phi = rng.uniform(-1, 1)
                v[j] = X[i][j] + phi * (X[i][j] - X[partner][j])
                v = np.clip(v, L, U)

                f_v = self.func(v)
                # Greedy selection
                if f_v < fitness[i]:
                    X[i] = v
                    fitness[i] = f_v
                    trial[i] = 0
                else:
                    trial[i] += 1

            # ── Onlooker bee phase ──
            # Compute selection probabilities (fitness-proportional)
            fit_vals = np.where(fitness >= 0,
                                1.0 / (1.0 + fitness),
                                1.0 + np.abs(fitness))
            probs = fit_vals / np.sum(fit_vals)

            for _ in range(SN):
                # Select food source via roulette wheel
                i = rng.choice(SN, p=probs)

                partner = rng.integers(SN - 1)
                if partner >= i:
                    partner += 1

                j = rng.integers(D)

                v = X[i].copy()
                phi = rng.uniform(-1, 1)
                v[j] = X[i][j] + phi * (X[i][j] - X[partner][j])
                v = np.clip(v, L, U)

                f_v = self.func(v)
                if f_v < fitness[i]:
                    X[i] = v
                    fitness[i] = f_v
                    trial[i] = 0
                else:
                    trial[i] += 1

            # ── Scout bee phase ──
            # Abandon exhausted food sources
            for i in range(SN):
                if trial[i] > self.limit:
                    X[i] = rng.uniform(L, U, D)
                    fitness[i] = self.func(X[i])
                    trial[i] = 0

            # Update global best
            k = int(np.argmin(fitness))
            if fitness[k] < g_fit:
                g_best = X[k].copy()
                g_fit = fitness[k]

            convergence[t] = g_fit

        return OptResult(best_position=g_best, best_fitness=float(g_fit),
                         convergence_curve=convergence)
