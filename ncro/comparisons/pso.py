"""
Particle Swarm Optimization (PSO) — Classic baseline.

Standard PSO with inertia weight (linearly decreasing from 0.9 to 0.4).
References:
  - Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization. IEEE ICNN.
  - Shi, Y. & Eberhart, R. (1998). A modified particle swarm optimizer. IEEE CEC.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class PSOOptimizer:
    """Standard Particle Swarm Optimization with inertia weight."""

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_iter: int = 1000,
        c1: float = 2.0,
        c2: float = 2.0,
        w_max: float = 0.9,
        w_min: float = 0.4,
        seed: int | None = None,
        **kwargs,
    ):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.seed = seed

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U = self.L, self.U

        # Initialize positions and velocities
        X = rng.uniform(L, U, size=(N, D))
        V = rng.uniform(-(U - L), (U - L), size=(N, D)) * 0.1

        # Evaluate
        fitness = np.array([self.func(X[i]) for i in range(N)])
        P = X.copy()
        P_fit = fitness.copy()

        g_idx = int(np.argmin(P_fit))
        g = P[g_idx].copy()
        g_fit = P_fit[g_idx]

        convergence = np.zeros(T)

        for t in range(T):
            w = self.w_max - (self.w_max - self.w_min) * t / T

            for i in range(N):
                r1 = rng.random(D)
                r2 = rng.random(D)
                V[i] = w * V[i] + self.c1 * r1 * (P[i] - X[i]) + self.c2 * r2 * (g - X[i])
                X[i] = np.clip(X[i] + V[i], L, U)

                f = self.func(X[i])
                if f < P_fit[i]:
                    P[i] = X[i].copy()
                    P_fit[i] = f

            best_idx = int(np.argmin(P_fit))
            if P_fit[best_idx] < g_fit:
                g = P[best_idx].copy()
                g_fit = P_fit[best_idx]

            convergence[t] = g_fit

        return OptResult(
            best_position=g,
            best_fitness=g_fit,
            convergence_curve=convergence,
        )
