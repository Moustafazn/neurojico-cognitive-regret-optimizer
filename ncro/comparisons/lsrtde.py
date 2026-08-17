"""
L-SRTDE — Success Rate-based Adaptive Differential Evolution.

CEC 2024 competition algorithm. Extends L-SHADE with success rate
monitoring to adaptively reset parameters when the algorithm stagnates.

Key innovation: monitors the success rate of mutations and resets
parameter memories when success rate drops below a threshold,
preventing premature convergence.

References:
  - Stanovov, V. et al. (2024). "Success rate-based adaptive differential
    evolution L-SRTDE for CEC 2024 competition."
"""

import numpy as np
from typing import Callable
from .base import OptResult


class LSRTDEOptimizer:
    """
    L-SRTDE: Success Rate-based adaptive DE for CEC 2024.

    Extends L-SHADE with:
      - Success rate monitoring per generation
      - Parameter memory reset when success rate drops
      - Improved stagnation detection and recovery
    """

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_iter: int = 1000,
        H: int = 5,
        N_min: int = 4,
        sr_threshold: float = 0.1,  # Success rate threshold for reset
        seed: int | None = None,
        **kwargs,
    ):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N_init = population_size
        self.T = max_iter
        self.H = H
        self.N_min = N_min
        self.sr_threshold = sr_threshold
        self.seed = seed
        self.population_size = population_size
        self.max_iter = max_iter

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        D, T = self.D, self.T
        L, U = self.L, self.U
        H = self.H
        N = self.N_init

        X = rng.uniform(L, U, size=(N, D))
        fitness = np.array([self.func(X[i]) for i in range(N)])

        archive = []
        archive_max = int(2.0 * N)

        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0

        g_idx = int(np.argmin(fitness))
        g = X[g_idx].copy()
        g_fit = fitness[g_idx]

        convergence = np.zeros(T)

        # Success rate tracking
        recent_success_rates = []
        stagnation_counter = 0

        for t in range(T):
            tau = (t + 1) / T
            S_F = []
            S_CR = []
            delta_f = []
            n_success = 0

            for i in range(N):
                r_idx = rng.integers(H)

                F_i = -1.0
                attempts = 0
                while F_i <= 0 and attempts < 100:
                    F_i = M_F[r_idx] + 0.1 * rng.standard_cauchy()
                    attempts += 1
                if F_i <= 0:
                    F_i = 0.01
                F_i = min(F_i, 1.0)

                if M_CR[r_idx] < 0:
                    CR_i = 0.0
                else:
                    CR_i = np.clip(rng.normal(M_CR[r_idx], 0.1), 0.0, 1.0)

                # Dynamic p-best with success rate influence
                p_ratio = max(0.05, 0.2 - 0.15 * tau)
                p = max(2, int(p_ratio * N))
                sorted_idx = np.argsort(fitness[:N])
                pbest_idx = sorted_idx[rng.integers(p)]

                candidates = np.delete(np.arange(N), i)
                r1 = rng.choice(candidates)

                n_archive = len(archive)
                combined_size = N + n_archive
                combined_cands = [c for c in range(combined_size) if c != i and c != r1]
                if len(combined_cands) == 0:
                    combined_cands = [c for c in range(combined_size) if c != i]
                r2_idx = rng.choice(combined_cands)

                if r2_idx < N:
                    x_r2 = X[r2_idx]
                else:
                    x_r2 = archive[r2_idx - N]

                mutant = X[i] + F_i * (X[pbest_idx] - X[i]) + F_i * (X[r1] - x_r2)

                # Midpoint boundary repair
                for j in range(D):
                    if mutant[j] < L:
                        mutant[j] = (L + X[i][j]) / 2.0
                    elif mutant[j] > U:
                        mutant[j] = (U + X[i][j]) / 2.0

                trial = X[i].copy()
                j_rand = rng.integers(D)
                for j in range(D):
                    if rng.random() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                f_trial = self.func(trial)
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        archive.append(X[i].copy())
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        delta_f.append(abs(fitness[i] - f_trial))
                        n_success += 1
                    X[i] = trial
                    fitness[i] = f_trial

            # Compute success rate
            success_rate = n_success / N if N > 0 else 0.0
            recent_success_rates.append(success_rate)
            if len(recent_success_rates) > 10:
                recent_success_rates.pop(0)

            # L-SRTDE: Check if success rate has dropped — reset memories
            avg_sr = np.mean(recent_success_rates) if recent_success_rates else 0.5
            if avg_sr < self.sr_threshold and t > 10:
                stagnation_counter += 1
                if stagnation_counter >= 5:
                    # Reset parameter memories to encourage diversity
                    M_F = np.full(H, 0.3 + 0.4 * rng.random())
                    M_CR = np.full(H, 0.3 + 0.4 * rng.random())
                    k = 0
                    stagnation_counter = 0
            else:
                stagnation_counter = max(0, stagnation_counter - 1)

            # Trim archive
            while len(archive) > archive_max:
                archive.pop(rng.integers(len(archive)))

            # Standard memory update
            if len(S_F) > 0:
                delta_f_arr = np.array(delta_f)
                weights = delta_f_arr / (np.sum(delta_f_arr) + 1e-30)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)

                M_F[k] = float(np.sum(weights * S_F_arr ** 2) / (np.sum(weights * S_F_arr) + 1e-30))

                if np.max(S_CR_arr) == 0:
                    M_CR[k] = -1.0
                else:
                    M_CR[k] = float(np.sum(weights * S_CR_arr ** 2) / (np.sum(weights * S_CR_arr) + 1e-30))

                k = (k + 1) % H

            g_idx_t = int(np.argmin(fitness[:N]))
            if fitness[g_idx_t] < g_fit:
                g = X[g_idx_t].copy()
                g_fit = fitness[g_idx_t]

            convergence[t] = g_fit

            # Linear population size reduction
            N_new = max(self.N_min, round(self.N_init + (self.N_min - self.N_init) * (t + 1) / T))
            if N_new < N:
                sorted_idx = np.argsort(fitness[:N])
                keep = sorted_idx[:N_new]
                X = X[keep]
                fitness = fitness[keep]
                N = N_new
                archive_max = int(2.0 * N)
                while len(archive) > archive_max:
                    archive.pop(rng.integers(len(archive)))

        return OptResult(best_position=g, best_fitness=g_fit, convergence_curve=convergence)
