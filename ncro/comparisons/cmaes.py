"""
CMA-ES — Covariance Matrix Adaptation Evolution Strategy.

Gold-standard evolutionary strategy for continuous optimization.
Uses a multivariate normal distribution to sample candidate solutions,
adapting both the mean and the full covariance matrix over generations.

References:
  - Hansen, N. & Ostermeier, A. (2001). "Completely Derandomized
    Self-Adaptation in Evolution Strategies." Evolutionary Computation,
    9(2), 159-195.
  - Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial."
    arXiv:1604.00772.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class CMAESOptimizer:
    """
    CMA-ES: Covariance Matrix Adaptation Evolution Strategy.

    A derandomized evolution strategy that adapts a full covariance
    matrix to capture pairwise variable dependencies. Widely regarded
    as the gold-standard for derivative-free continuous optimization
    on problems up to moderate dimensionality.

    Key mechanisms:
      - Step-size control via cumulative sigma path (CSA)
      - Rank-one update via evolution path
      - Rank-mu update via weighted recombination of selected steps
      - Automatic adaptation of all internal parameters from (D, lambda)
    """

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_iter: int = 1000,
        seed: int | None = None,
        **kwargs,
    ):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.seed = seed
        # For interface compatibility
        self.population_size = population_size
        self.max_iter = max_iter

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        D, T = self.D, self.T
        L, U = self.L, self.U
        lambda_size = self.N
        mu = lambda_size // 2

        # ── Recombination weights ──
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # ── Adaptation constants (Hansen 2016 defaults) ──
        c_c = (4.0 + mu_eff / D) / (D + 4.0 + 2.0 * mu_eff / D)

        c_sigma = (mu_eff + 2.0) / (D + mu_eff + 5.0)

        c_1 = 2.0 / ((D + 1.3) ** 2 + mu_eff)

        c_mu = min(
            1.0 - c_1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((D + 2.0) ** 2 + mu_eff),
        )

        d_sigma = (
            1.0
            + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (D + 1.0)) - 1.0)
            + c_sigma
        )

        chi_n = np.sqrt(D) * (
            1.0 - 1.0 / (4.0 * D) + 1.0 / (21.0 * D ** 2)
        )

        # ── State initialisation ──
        mean = np.clip(rng.uniform(L, U, size=D), L, U)

        if np.isscalar(U):
            sigma = 0.25 * abs(U - L)
        else:
            sigma = 0.25 * np.mean(np.abs(np.asarray(U) - np.asarray(L)))

        C = np.eye(D)
        p_c = np.zeros(D)
        p_sigma = np.zeros(D)

        g_best = mean.copy()
        g_fit = self.func(np.clip(mean, L, U))
        convergence = np.zeros(T)

        for t in range(T):
            # ── Eigen-decomposition of C ──
            C = 0.5 * (C + C.T)
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-14)
            D_diag = np.sqrt(eigenvalues)

            # ── Sample lambda offspring ──
            z = rng.normal(size=(lambda_size, D))
            y = (z * D_diag) @ B.T                      # ~ N(0, C)
            X = np.clip(mean + sigma * y, L, U)

            # ── Evaluate ──
            fitness = np.array([self.func(X[i]) for i in range(lambda_size)])

            # ── Selection and recombination ──
            order = np.argsort(fitness)
            sel = order[:mu]

            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * X[sel], axis=0)

            # ── Update evolution paths ──
            delta_mean = (mean - old_mean) / max(sigma, 1e-15)

            inv_sqrt_C = (B * (1.0 / D_diag)) @ B.T

            p_sigma = (
                (1.0 - c_sigma) * p_sigma
                + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff)
                * (inv_sqrt_C @ delta_mean)
            )

            generation = t + 1
            norm_factor = np.sqrt(
                1.0 - (1.0 - c_sigma) ** (2.0 * generation)
            )
            h_sigma = float(
                np.linalg.norm(p_sigma) / max(norm_factor, 1e-15) / chi_n
                < (1.4 + 2.0 / (D + 1.0))
            )

            p_c = (
                (1.0 - c_c) * p_c
                + h_sigma
                * np.sqrt(c_c * (2.0 - c_c) * mu_eff)
                * delta_mean
            )

            # ── Covariance matrix update ──
            selected_steps = (X[sel] - old_mean) / max(sigma, 1e-15)

            rank_mu_update = np.zeros((D, D))
            for idx in range(mu):
                rank_mu_update += weights[idx] * np.outer(
                    selected_steps[idx], selected_steps[idx]
                )

            C = (
                (1.0 - c_1 - c_mu) * C
                + c_1 * (
                    np.outer(p_c, p_c)
                    + (1.0 - h_sigma) * c_c * (2.0 - c_c) * C
                )
                + c_mu * rank_mu_update
            )

            # ── Step-size update ──
            sigma *= np.exp(
                (c_sigma / d_sigma)
                * (np.linalg.norm(p_sigma) / chi_n - 1.0)
            )

            # ── Track global best ──
            best_idx = order[0]
            if fitness[best_idx] < g_fit:
                g_best = X[best_idx].copy()
                g_fit = fitness[best_idx]

            convergence[t] = g_fit

        return OptResult(
            best_position=g_best,
            best_fitness=g_fit,
            convergence_curve=convergence,
        )
