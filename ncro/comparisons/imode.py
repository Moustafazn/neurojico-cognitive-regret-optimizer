"""
IMODE — Improved Multi-Operator Differential Evolution.

CEC 2020 competition winner. Uses multiple mutation strategies
simultaneously, with operator probabilities adapted based on
recent success history.

Key innovations over single-operator DE:
  - Three mutation operators competing within each generation
  - Operator selection probability adapted via success-rate feedback
  - Combines DE/current-to-pbest/1, DE/current-to-pbest-w/1,
    and DE/rand-to-pbest/1 strategies
  - Success-history based F and CR adaptation (SHADE-style)
  - Linear population size reduction (L-SHADE-style)

References:
  - Sallam, K.M. et al. (2020). "Improved Multi-operator Differential
    Evolution Algorithm for Solving Unconstrained Problems." IEEE CEC 2020.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class IMODEOptimizer:
    """
    IMODE: CEC 2020 competition-winning multi-operator DE.

    Maintains three mutation strategies and adapts their selection
    probabilities based on which operators produce the most
    successful trial vectors.

    Operators:
      1. DE/current-to-pbest/1 (standard L-SHADE mutation)
      2. DE/current-to-pbest-w/1 (jSO weighted mutation)
      3. DE/rand-to-pbest/1 (alternative exploration mutation)
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

    def _mutate(self, op, X, i, F_i, pbest_idx, r1, x_r2, candidates, tau, rng):
        """Apply selected mutation operator."""
        if op == 0:
            # Operator 1: DE/current-to-pbest/1 (standard L-SHADE)
            return X[i] + F_i * (X[pbest_idx] - X[i]) + F_i * (X[r1] - x_r2)
        elif op == 1:
            # Operator 2: DE/current-to-pbest-w/1 (jSO weighted)
            Fw = 0.7 * F_i if tau < 0.2 else (0.8 * F_i if tau < 0.4 else 1.2 * F_i)
            return X[i] + Fw * (X[pbest_idx] - X[i]) + F_i * (X[r1] - x_r2)
        else:
            # Operator 3: DE/rand-to-pbest/1 (exploration)
            remaining = [idx for idx in candidates if idx != r1]
            if len(remaining) == 0:
                remaining = list(candidates)
            r3 = rng.choice(remaining)
            return X[pbest_idx] + F_i * (X[r1] - X[r3])

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
        M_CR = np.full(H, 0.5)
        k = 0

        # Multi-operator probabilities (3 operators)
        n_ops = 3
        op_probs = np.ones(n_ops) / n_ops
        op_success = np.zeros(n_ops)
        op_usage = np.zeros(n_ops)
        adapt_period = max(5, N // 5)

        g_idx = int(np.argmin(fitness))
        g = X[g_idx].copy()
        g_fit = fitness[g_idx]
        convergence = np.zeros(T)

        for t in range(T):
            tau = (t + 1) / T
            S_F, S_CR, delta_f = [], [], []

            for i in range(N):
                # Sample F and CR from historical memory
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

                # Select operator
                op = int(rng.choice(n_ops, p=op_probs))
                op_usage[op] += 1

                # Dynamic p-best
                p_ratio = max(0.05, 0.25 - 0.20 * tau)
                p = max(2, int(p_ratio * N))
                sorted_idx = np.argsort(fitness[:N])
                pbest_idx = sorted_idx[rng.integers(p)]

                # Select r1 != i
                candidates = np.delete(np.arange(N), i)
                r1 = rng.choice(candidates)

                # Select r2 from population + archive
                n_arch = len(archive)
                combined_cands = [
                    c for c in range(N + n_arch) if c != i and c != r1
                ]
                if len(combined_cands) == 0:
                    combined_cands = [c for c in range(N + n_arch) if c != i]
                r2_idx = rng.choice(combined_cands)
                x_r2 = X[r2_idx] if r2_idx < N else archive[r2_idx - N]

                # Mutation
                mutant = self._mutate(
                    op, X, i, F_i, pbest_idx, r1, x_r2, candidates, tau, rng,
                )

                # Boundary repair: midpoint strategy
                for j in range(D):
                    if mutant[j] < L:
                        mutant[j] = (L + X[i][j]) / 2.0
                    elif mutant[j] > U:
                        mutant[j] = (U + X[i][j]) / 2.0

                # Binomial crossover
                trial = X[i].copy()
                j_rand = rng.integers(D)
                for j in range(D):
                    if rng.random() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                # Selection
                f_trial = self.func(trial)
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        archive.append(X[i].copy())
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        delta_f.append(abs(fitness[i] - f_trial))
                        op_success[op] += 1
                    X[i] = trial
                    fitness[i] = f_trial


            # Adapt operator probabilities periodically
            if (t + 1) % adapt_period == 0:
                total_usage = np.sum(op_usage)
                if total_usage > 0:
                    sr = np.zeros(n_ops)
                    for oi in range(n_ops):
                        if op_usage[oi] > 0:
                            sr[oi] = op_success[oi] / op_usage[oi]
                    sr_sum = np.sum(sr)
                    if sr_sum > 1e-30:
                        op_probs = sr / sr_sum
                    else:
                        op_probs = np.ones(n_ops) / n_ops
                    # Enforce minimum probability to avoid starvation
                    min_prob = 0.1 / n_ops
                    op_probs = np.maximum(op_probs, min_prob)
                    op_probs /= np.sum(op_probs)
                op_success[:] = 0
                op_usage[:] = 0

            # Trim archive
            while len(archive) > archive_max:
                archive.pop(rng.integers(len(archive)))

            # Update historical memory (weighted Lehmer mean)
            if len(S_F) > 0:
                df_arr = np.array(delta_f)
                w = df_arr / (np.sum(df_arr) + 1e-30)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)

                M_F[k] = float(
                    np.sum(w * S_F_arr ** 2) / (np.sum(w * S_F_arr) + 1e-30)
                )
                if np.max(S_CR_arr) == 0:
                    M_CR[k] = -1.0
                else:
                    M_CR[k] = float(
                        np.sum(w * S_CR_arr ** 2) / (np.sum(w * S_CR_arr) + 1e-30)
                    )
                k = (k + 1) % H

            # Update global best
            g_idx_t = int(np.argmin(fitness[:N]))
            if fitness[g_idx_t] < g_fit:
                g = X[g_idx_t].copy()
                g_fit = fitness[g_idx_t]

            convergence[t] = g_fit

            # Linear population size reduction (L-SHADE style)
            N_new = max(
                self.N_min,
                round(self.N_init + (self.N_min - self.N_init) * (t + 1) / T),
            )
            if N_new < N:
                sorted_idx = np.argsort(fitness[:N])
                keep = sorted_idx[:N_new]
                X = X[keep]
                fitness = fitness[keep]
                N = N_new
                archive_max = N
                while len(archive) > archive_max:
                    archive.pop(rng.integers(len(archive)))

        return OptResult(
            best_position=g,
            best_fitness=g_fit,
            convergence_curve=convergence,
        )

