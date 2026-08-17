"""
jSO — Self-adaptive Differential Evolution with Weighted Mutation.

An advanced L-SHADE variant that won CEC 2017 and remains competitive
through CEC 2024+. Uses weighted mutation strategy and improved
parameter adaptation.

References:
  - Brest, J., Maucec, M.S., & Boskovic, B. (2017). "Single Objective
    Real-Parameter Optimization: Algorithm jSO." IEEE CEC 2017.
  - Brest, J. et al. (2021). Updated jSO variants for CEC competitions.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class jSOOptimizer:
    """
    jSO: CEC competition-winning adaptive DE (2017+).

    Key differences from L-SHADE:
      - Weighted mutation (current-to-pbest-w/1)
      - Improved F and CR adaptation with generation-dependent bounds
      - Dynamic p-best percentage
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
        archive_max = N

        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.8)
        k = 0

        g_idx = int(np.argmin(fitness))
        g = X[g_idx].copy()
        g_fit = fitness[g_idx]

        convergence = np.zeros(T)

        for t in range(T):
            tau = (t + 1) / T
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                r_idx = rng.integers(H)

                # jSO: Generation-dependent F bounds
                F_i = -1.0
                while F_i <= 0:
                    F_i = M_F[r_idx] + 0.1 * rng.standard_cauchy()
                if F_i > 1.0:
                    F_i = 1.0

                # jSO modification: cap F at 0.7 in early generations
                if tau < 0.6 and F_i > 0.7:
                    F_i = 0.7

                # jSO: Generation-dependent CR
                if M_CR[r_idx] < 0:
                    CR_i = 0.0
                else:
                    CR_i = np.clip(rng.normal(M_CR[r_idx], 0.1), 0.0, 1.0)

                # jSO modification: force high CR early, low CR late
                if tau < 0.25 and CR_i < 0.7:
                    CR_i = 0.7
                if tau < 0.5 and CR_i < 0.6:
                    CR_i = 0.6

                # Dynamic p-best
                p_ratio = max(0.05, 0.25 - 0.2 * tau)
                p = max(2, int(p_ratio * N))
                sorted_idx = np.argsort(fitness[:N])
                pbest_idx = sorted_idx[rng.integers(p)]

                # Select r1 != i
                candidates = np.delete(np.arange(N), i)
                r1 = rng.choice(candidates)

                # r2 from pop + archive
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

                # jSO: Weighted mutation (current-to-pbest-w/1)
                # Weight Fw depends on generation
                if tau < 0.2:
                    Fw = 0.7 * F_i
                elif tau < 0.4:
                    Fw = 0.8 * F_i
                else:
                    Fw = 1.2 * F_i

                mutant = X[i] + Fw * (X[pbest_idx] - X[i]) + F_i * (X[r1] - x_r2)
                mutant = np.clip(mutant, L, U)

                # Binomial crossover
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
                    X[i] = trial
                    fitness[i] = f_trial

            # Trim archive
            while len(archive) > archive_max:
                archive.pop(rng.integers(len(archive)))

            # Update historical memory
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

            # Linear Population Size Reduction (same as L-SHADE)
            N_new = max(self.N_min, round(self.N_init + (self.N_min - self.N_init) * (t + 1) / T))
            if N_new < N:
                sorted_idx = np.argsort(fitness[:N])
                keep = sorted_idx[:N_new]
                X = X[keep]
                fitness = fitness[keep]
                N = N_new
                archive_max = N
                while len(archive) > archive_max:
                    archive.pop(rng.integers(len(archive)))

        return OptResult(best_position=g, best_fitness=g_fit, convergence_curve=convergence)
