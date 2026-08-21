"""
Harris Hawks Optimization (HHO).

Nature-inspired metaheuristic based on the cooperative hunting behavior
of Harris hawks and their surprise pounce strategy.
References:
  - Heidari, A.A. et al. (2019). "Harris hawks optimization: Algorithm
    and applications." Future Generation Computer Systems, 97, 849-872.
"""

import math
import numpy as np
from typing import Callable
from .base import OptResult


class HHOOptimizer:
    """Harris Hawks Optimization."""

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

        # Initialize population
        X = rng.uniform(L, U, size=(N, D))
        fitness = np.array([self.func(X[i]) for i in range(N)])

        # Rabbit (prey) — best solution
        k = int(np.argmin(fitness))
        rabbit = X[k].copy()
        rabbit_fit = fitness[k]

        convergence = np.zeros(T)

        for t in range(T):
            # Escaping energy of rabbit
            E0 = 2 * rng.random() - 1  # initial energy in [-1, 1]
            J_factor = 2 * (1 - t / T)  # decreasing factor
            E = E0 * J_factor  # escaping energy

            for i in range(N):
                q = rng.random()

                if abs(E) >= 1:
                    # ── Exploration phase ──
                    if q >= 0.5:
                        # Random tall tree strategy
                        r_idx = rng.integers(N)
                        X_rand = X[r_idx]
                        r1 = rng.random(D)
                        X[i] = X_rand - r1 * np.abs(X_rand - 2 * rng.random(D) * X[i])
                    else:
                        # Perch on random tall trees
                        r3 = rng.random(D)
                        r4 = rng.random()
                        X_mean = np.mean(X, axis=0)
                        X[i] = (rabbit - X_mean) - r3 * (L + rng.random(D) * (U - L)) * r4
                else:
                    # ── Exploitation phase ──
                    r5 = rng.random()

                    if r5 >= 0.5 and abs(E) >= 0.5:
                        # Soft besiege
                        delta_X = rabbit - X[i]
                        X[i] = delta_X - E * np.abs(J_factor * rabbit - X[i])

                    elif r5 >= 0.5 and abs(E) < 0.5:
                        # Hard besiege
                        delta_X = rabbit - X[i]
                        X[i] = rabbit - E * np.abs(delta_X)

                    elif r5 < 0.5 and abs(E) >= 0.5:
                        # Soft besiege with progressive rapid dives
                        Y = rabbit - E * np.abs(J_factor * rabbit - X[i])
                        Y = np.clip(Y, L, U)
                        if self.func(Y) < fitness[i]:
                            X[i] = Y
                        else:
                            # Levy flight
                            S = self._levy_flight(D, rng)
                            Z = Y + rng.random(D) * S
                            Z = np.clip(Z, L, U)
                            if self.func(Z) < fitness[i]:
                                X[i] = Z

                    else:
                        # Hard besiege with progressive rapid dives
                        X_mean = np.mean(X, axis=0)
                        Y = rabbit - E * np.abs(J_factor * rabbit - X_mean)
                        Y = np.clip(Y, L, U)
                        if self.func(Y) < fitness[i]:
                            X[i] = Y
                        else:
                            S = self._levy_flight(D, rng)
                            Z = Y + rng.random(D) * S
                            Z = np.clip(Z, L, U)
                            if self.func(Z) < fitness[i]:
                                X[i] = Z

                X[i] = np.clip(X[i], L, U)
                fitness[i] = self.func(X[i])

            # Update rabbit
            k = int(np.argmin(fitness))
            if fitness[k] < rabbit_fit:
                rabbit = X[k].copy()
                rabbit_fit = fitness[k]

            convergence[t] = rabbit_fit

        return OptResult(best_position=rabbit, best_fitness=float(rabbit_fit),
                         convergence_curve=convergence)

    @staticmethod
    def _levy_flight(D: int, rng) -> np.ndarray:
        """Generate Levy flight step using Mantegna's algorithm."""
        beta = 1.5
        sigma_u = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = rng.normal(0, sigma_u, D)
        v = rng.normal(0, 1, D)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step
