"""
Ablation Study Variants for NCRO.

Each variant uses the SAME motion framework as the full optimizer,
with exactly ONE component disabled to isolate its contribution.

Full motion equation (reference):
  candidate = x_i + w·V_i + α(t)(1+M_R)·E_i + β(t)(1-M_R)·H_i + γ(t)·C·D_C

Professor's required ablation variants (Section 6.8):
  NCRO-R:    Regret Memory removed           → NCRO_NoRegret
  NCRO-C:    Counterfactual Success Memory    → NCRO_NoCFMem
  NCRO-L:    Counterfactual Learning removed  → NCRO_NoCounterfactual
  NCRO-M:    Regret-Aware Momentum removed    → NCRO_NoMomentum
  NCRO-Full: Complete Model                   → NCROOptimizer
"""

import numpy as np
from typing import Callable
from .optimizer import NCROResult


# ──────────────────────────────────────────────────────────────────
# Shared NCRO default parameters (must match optimizer.py exactly)
# ──────────────────────────────────────────────────────────────────
_NCRO_DEFAULTS = dict(
    c1=2.0, c2=2.0,
    alpha_max=2.0, alpha_min=0.15,
    beta_min=0.15, beta_max=1.8,
    gamma_max=1.0, gamma_min=0.05,
    rho=0.90, rho_c=0.90,
    eta_R=0.30, eta_C=0.20, eta_P=0.20,
    q_min=0.10, q_max=0.90,
    epsilon=1e-12,
)


class NCRO_NoRegret:
    """
    NCRO with regret signal DISABLED.

    Changes from Full:
      - M_R is never updated (stays 0)
      - Motion equation becomes: x_i + α·E + β·H + γ·C·D_C
        (no regret amplification/dampening)
      - q_i = q0(t) + η_C·C - η_P·P  (no η_R·M_R term)

    Everything else (counterfactual generation, C_mem, blend,
    greedy 4-way selection, progress memory) remains identical to Full.
    """

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.seed = seed
        for k, v in _NCRO_DEFAULTS.items():
            setattr(self, k, v)

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U, eps = self.L, self.U, self.epsilon

        X = rng.uniform(L, U, size=(N, D))
        F = np.array([self.func(X[i]) for i in range(N)])
        P = X.copy(); PF = F.copy()
        g_idx = int(np.argmin(F)); G = P[g_idx].copy(); GF = F[g_idx]

        # M_R is NOT used — stays zero.  C_mem and progress are kept.
        C_mem = np.zeros(N)
        progress_mem = np.zeros(N)

        convergence = np.zeros(T + 1); convergence[0] = GF
        expl_cnt = np.zeros(T); xplt_cnt = np.zeros(T)
        cf_cnt = np.zeros(T); regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        for t in range(T):
            tau = t / max(1, T - 1)
            alpha_t = self.alpha_max * (1 - tau) + self.alpha_min * tau
            beta_t = self.beta_min * (1 - tau) + self.beta_max * tau
            gamma_t = self.gamma_max * (1 - tau) + self.gamma_min * tau
            sigma_t = 1.0 * (1 - tau) + 0.01 * tau
            q0_t = self.q_max * (1 - tau) + self.q_min * tau

            newX = np.empty_like(X); newF = np.empty(N)
            it_expl = it_xplt = it_cf = 0

            for i in range(N):
                r1, r2 = rng.choice(N, 2, replace=False)
                E_i = X[r1] - X[r2]
                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                # q WITHOUT regret (η_R * M_R removed)
                q_i = q0_t + self.eta_C * C_mem[i] - self.eta_P * progress_mem[i]
                q_i = float(np.clip(q_i, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                # Actual and counterfactual candidates
                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i
                # Regret-modulated position-relative noise + dimension-selective perturbation
                dist_to_best = np.linalg.norm(X[i] - G)
                regret_boost = 1.0
                noise_scale = min(max(dist_to_best * regret_boost, eps),
                                  (U - L) / np.sqrt(D))
                max_dims = max(1, int(D * (1 - 0.7 * tau)))
                n_dims = rng.integers(1, max_dims + 1)
                dims = rng.choice(D, n_dims, replace=False)
                noise = np.zeros(D)
                noise[dims] = (0.02 / np.sqrt(D)) * sigma_t * noise_scale * rng.standard_normal(n_dims)
                Y_A = np.clip(Y_A + noise, L, U)
                Y_C = np.clip(Y_C, L, U)
                F_A = self.func(Y_A); F_C = self.func(Y_C)

                # CF success — regret is computed but NOT accumulated
                success = float(F_C < F_A - eps)
                C_mem[i] = self.rho_c * C_mem[i] + (1 - self.rho_c) * success

                D_C = Y_C - X[i]

                # Motion WITHOUT regret modulation (M_R=0)
                candidate = (
                    X[i]
                    + alpha_t * E_i          # no (1+M_R) multiplier
                    + beta_t * H_i           # no (1-M_R) multiplier
                    + gamma_t * C_mem[i] * D_C
                )

                if rng.random() < q_i:
                    candidate = 0.75 * candidate + 0.25 * Y_A; it_expl += 1
                else:
                    candidate = 0.75 * candidate + 0.25 * Y_C; it_xplt += 1

                candidate = np.clip(candidate, L, U)
                F_cand = self.func(candidate)

                # Greedy 4-way selection
                candidates = [X[i], Y_A, Y_C, candidate]
                values = [F[i], F_A, F_C, F_cand]
                best_idx = int(np.argmin(values))
                newX[i] = candidates[best_idx]; newF[i] = values[best_idx]
                if success > 0: it_cf += 1

            F_old = F.copy(); X = newX; F = newF
            improved = F < PF; P[improved] = X[improved]; PF[improved] = F[improved]
            k = int(np.argmin(PF))
            if PF[k] < GF: G = P[k].copy(); GF = PF[k]

            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0)

            convergence[t + 1] = GF
            expl_cnt[t] = it_expl; xplt_cnt[t] = it_xplt; cf_cnt[t] = it_cf

        return NCROResult(G, float(GF), convergence[1:], expl_cnt, xplt_cnt,
                          cf_cnt, regret_avg, q_hist)


class NCRO_NoCFMem:
    """
    NCRO with Counterfactual Success Memory (C_mem) DISABLED.
    Maps to professor's NCRO-C variant (Section 6.8).

    Changes from Full:
      - C_mem is forced to 0 (never accumulated)
      - Y_C IS still generated, regret IS still computed and accumulated in M_R
      - But γ·C_mem·D_C = γ·0·D_C = 0 (no counterfactual direction force)
      - q still uses η_R·M_R but η_C·C_mem = 0
      - Regret still modulates exploration/exploitation forces via (1+M_R) and (1-M_R)

    This isolates the contribution of the counterfactual success memory
    while preserving regret computation from CF comparison.
    """

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.seed = seed
        for k, v in _NCRO_DEFAULTS.items():
            setattr(self, k, v)

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U, eps = self.L, self.U, self.epsilon

        X = rng.uniform(L, U, size=(N, D))
        F = np.array([self.func(X[i]) for i in range(N)])
        P = X.copy(); PF = F.copy()
        g_idx = int(np.argmin(F)); G = P[g_idx].copy(); GF = F[g_idx]

        M_R = np.zeros(N)
        # C_mem is NOT used — stays zero throughout
        progress_mem = np.zeros(N)

        convergence = np.zeros(T + 1); convergence[0] = GF
        expl_cnt = np.zeros(T); xplt_cnt = np.zeros(T)
        cf_cnt = np.zeros(T); regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        for t in range(T):
            tau = t / max(1, T - 1)
            alpha_t = self.alpha_max * (1 - tau) + self.alpha_min * tau
            beta_t = self.beta_min * (1 - tau) + self.beta_max * tau
            gamma_t = self.gamma_max * (1 - tau) + self.gamma_min * tau
            sigma_t = 1.0 * (1 - tau) + 0.01 * tau
            q0_t = self.q_max * (1 - tau) + self.q_min * tau

            newX = np.empty_like(X); newF = np.empty(N)
            it_expl = it_xplt = it_cf = 0; it_reg = 0.0

            for i in range(N):
                r1, r2 = rng.choice(N, 2, replace=False)
                E_i = X[r1] - X[r2]
                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                # q uses regret but NOT C_mem (η_C * 0 = 0)
                q_i = q0_t + self.eta_R * M_R[i] - self.eta_P * progress_mem[i]
                q_i = float(np.clip(q_i, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                # Y_C IS generated (for regret computation)
                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i
                # Regret-modulated position-relative noise + dimension-selective perturbation
                dist_to_best = np.linalg.norm(X[i] - G)
                regret_boost = 1.0 + M_R[i] * 5.0
                noise_scale = min(max(dist_to_best * regret_boost, eps),
                                  (U - L) / np.sqrt(D))
                max_dims = max(1, int(D * (1 - 0.7 * tau)))
                n_dims = rng.integers(1, max_dims + 1)
                dims = rng.choice(D, n_dims, replace=False)
                noise = np.zeros(D)
                noise[dims] = (0.02 / np.sqrt(D)) * sigma_t * noise_scale * rng.standard_normal(n_dims)
                Y_A = np.clip(Y_A + noise, L, U)
                Y_C = np.clip(Y_C, L, U)
                F_A = self.func(Y_A); F_C = self.func(Y_C)

                # Regret IS computed and accumulated
                regret = max(0.0, F_A - F_C) / (abs(F_A) + abs(F_C) + eps)
                regret = float(np.clip(regret, 0, 1))
                success = float(F_C < F_A - eps)
                M_R[i] = self.rho * M_R[i] + (1 - self.rho) * regret
                # C_mem NOT accumulated (stays 0)

                D_C = Y_C - X[i]

                # Motion with regret modulation but NO CF direction (C_mem=0)
                # γ·0·D_C = 0
                candidate = (
                    X[i]
                    + alpha_t * (1 + M_R[i]) * E_i
                    + beta_t * (1 - M_R[i]) * H_i
                    # + gamma_t * 0 * D_C  (C_mem is always 0)
                )

                if rng.random() < q_i:
                    candidate = 0.75 * candidate + 0.25 * Y_A; it_expl += 1
                else:
                    candidate = 0.75 * candidate + 0.25 * Y_C; it_xplt += 1

                candidate = np.clip(candidate, L, U)
                F_cand = self.func(candidate)

                # Greedy 4-way selection (Y_C still available)
                candidates = [X[i], Y_A, Y_C, candidate]
                values = [F[i], F_A, F_C, F_cand]
                best_idx = int(np.argmin(values))
                newX[i] = candidates[best_idx]; newF[i] = values[best_idx]
                if success > 0: it_cf += 1
                it_reg += M_R[i]

            F_old = F.copy(); X = newX; F = newF
            improved = F < PF; P[improved] = X[improved]; PF[improved] = F[improved]
            k = int(np.argmin(PF))
            if PF[k] < GF: G = P[k].copy(); GF = PF[k]

            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0)

            convergence[t + 1] = GF
            expl_cnt[t] = it_expl; xplt_cnt[t] = it_xplt
            cf_cnt[t] = it_cf; regret_avg[t] = it_reg / N

        return NCROResult(G, float(GF), convergence[1:], expl_cnt, xplt_cnt,
                          cf_cnt, regret_avg, q_hist)


class NCRO_NoCounterfactual:
    """
    NCRO with counterfactual candidate DISABLED.

    Changes from Full:
      - No Y_C is generated (no counterfactual thinking)
      - No D_C direction, no regret, no C_mem
      - Motion simplifies to: x_i + α·E + β·H  (no γ·C·D_C)
      - q follows fixed schedule: q = q0(t)
      - Selection is greedy among {x_i, Y_A, candidate} (3-way)
      - Uses fewer function evaluations (no F_C)

    This tests the core contribution of counterfactual reasoning.
    """

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.seed = seed
        for k, v in _NCRO_DEFAULTS.items():
            setattr(self, k, v)

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U, eps = self.L, self.U, self.epsilon

        X = rng.uniform(L, U, size=(N, D))
        F = np.array([self.func(X[i]) for i in range(N)])
        P = X.copy(); PF = F.copy()
        g_idx = int(np.argmin(F)); G = P[g_idx].copy(); GF = F[g_idx]

        progress_mem = np.zeros(N)

        convergence = np.zeros(T + 1); convergence[0] = GF
        expl_cnt = np.zeros(T); xplt_cnt = np.zeros(T)
        cf_cnt = np.zeros(T); regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        for t in range(T):
            tau = t / max(1, T - 1)
            alpha_t = self.alpha_max * (1 - tau) + self.alpha_min * tau
            beta_t = self.beta_min * (1 - tau) + self.beta_max * tau
            sigma_t = 1.0 * (1 - tau) + 0.01 * tau
            q0_t = self.q_max * (1 - tau) + self.q_min * tau

            newX = np.empty_like(X); newF = np.empty(N)
            it_expl = it_xplt = 0

            for i in range(N):
                r1, r2 = rng.choice(N, 2, replace=False)
                E_i = X[r1] - X[r2]
                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                # Fixed q — no feedback (no CF, no regret, no C_mem)
                q_i = float(np.clip(q0_t, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                # ONLY actual candidate — NO counterfactual
                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
                # Regret-modulated position-relative noise + dimension-selective perturbation
                dist_to_best = np.linalg.norm(X[i] - G)
                regret_boost = 1.0
                noise_scale = min(max(dist_to_best * regret_boost, eps),
                                  (U - L) / np.sqrt(D))
                max_dims = max(1, int(D * (1 - 0.7 * tau)))
                n_dims = rng.integers(1, max_dims + 1)
                dims = rng.choice(D, n_dims, replace=False)
                noise = np.zeros(D)
                noise[dims] = (0.02 / np.sqrt(D)) * sigma_t * noise_scale * rng.standard_normal(n_dims)
                Y_A = np.clip(Y_A + noise, L, U)
                F_A = self.func(Y_A)

                # Motion WITHOUT counterfactual direction (no γ·C·D_C, no M_R)
                candidate = X[i] + alpha_t * E_i + beta_t * H_i
                candidate = np.clip(candidate, L, U)
                F_cand = self.func(candidate)

                # Greedy 3-way selection (no Y_C)
                candidates = [X[i], Y_A, candidate]
                values = [F[i], F_A, F_cand]
                best_idx = int(np.argmin(values))
                newX[i] = candidates[best_idx]; newF[i] = values[best_idx]

                if q_i > 0.5: it_expl += 1
                else: it_xplt += 1

            F_old = F.copy(); X = newX; F = newF
            improved = F < PF; P[improved] = X[improved]; PF[improved] = F[improved]
            k = int(np.argmin(PF))
            if PF[k] < GF: G = P[k].copy(); GF = PF[k]

            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0)

            convergence[t + 1] = GF
            expl_cnt[t] = it_expl; xplt_cnt[t] = it_xplt

        return NCROResult(G, float(GF), convergence[1:], expl_cnt, xplt_cnt,
                          cf_cnt, regret_avg, q_hist)


class NCRO_NoAdaptiveEE:
    """
    NCRO with adaptive exploration-exploitation balance DISABLED.

    Changes from Full:
      - q = q0(t) = q_max·(1-τ) + q_min·τ   (fixed linear schedule)
      - η_R, η_C, η_P have NO influence on q
      - M_R and C_mem ARE still computed and DO modulate the forces
      - Counterfactual IS still generated

    This tests whether the ADAPTIVE q mechanism contributes
    beyond the time-based schedule alone.
    """

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.seed = seed
        for k, v in _NCRO_DEFAULTS.items():
            setattr(self, k, v)

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U, eps = self.L, self.U, self.epsilon

        X = rng.uniform(L, U, size=(N, D))
        F = np.array([self.func(X[i]) for i in range(N)])
        P = X.copy(); PF = F.copy()
        g_idx = int(np.argmin(F)); G = P[g_idx].copy(); GF = F[g_idx]

        M_R = np.zeros(N); C_mem = np.zeros(N)

        convergence = np.zeros(T + 1); convergence[0] = GF
        expl_cnt = np.zeros(T); xplt_cnt = np.zeros(T)
        cf_cnt = np.zeros(T); regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        for t in range(T):
            tau = t / max(1, T - 1)
            alpha_t = self.alpha_max * (1 - tau) + self.alpha_min * tau
            beta_t = self.beta_min * (1 - tau) + self.beta_max * tau
            gamma_t = self.gamma_max * (1 - tau) + self.gamma_min * tau
            sigma_t = 1.0 * (1 - tau) + 0.01 * tau
            q0_t = self.q_max * (1 - tau) + self.q_min * tau

            newX = np.empty_like(X); newF = np.empty(N)
            it_expl = it_xplt = it_cf = 0; it_reg = 0.0

            for i in range(N):
                r1, r2 = rng.choice(N, 2, replace=False)
                E_i = X[r1] - X[r2]
                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                # FIXED q — no adaptive terms at all
                q_i = float(np.clip(q0_t, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i
                # Regret-modulated position-relative noise + dimension-selective perturbation
                dist_to_best = np.linalg.norm(X[i] - G)
                regret_boost = 1.0 + M_R[i] * 5.0
                noise_scale = min(max(dist_to_best * regret_boost, eps),
                                  (U - L) / np.sqrt(D))
                max_dims = max(1, int(D * (1 - 0.7 * tau)))
                n_dims = rng.integers(1, max_dims + 1)
                dims = rng.choice(D, n_dims, replace=False)
                noise = np.zeros(D)
                noise[dims] = (0.02 / np.sqrt(D)) * sigma_t * noise_scale * rng.standard_normal(n_dims)
                Y_A = np.clip(Y_A + noise, L, U)
                Y_C = np.clip(Y_C, L, U)
                F_A = self.func(Y_A); F_C = self.func(Y_C)

                # Regret + CF success computed and used in FORCES (but NOT in q)
                regret = max(0.0, F_A - F_C) / (abs(F_A) + abs(F_C) + eps)
                regret = float(np.clip(regret, 0, 1))
                success = float(F_C < F_A - eps)
                M_R[i] = self.rho * M_R[i] + (1 - self.rho) * regret
                C_mem[i] = self.rho_c * C_mem[i] + (1 - self.rho_c) * success

                D_C = Y_C - X[i]

                # Motion with regret-modulated forces (M_R still active here)
                candidate = (
                    X[i]
                    + alpha_t * (1 + M_R[i]) * E_i
                    + beta_t * (1 - M_R[i]) * H_i
                    + gamma_t * C_mem[i] * D_C
                )

                if rng.random() < q_i:
                    candidate = 0.75 * candidate + 0.25 * Y_A; it_expl += 1
                else:
                    candidate = 0.75 * candidate + 0.25 * Y_C; it_xplt += 1

                candidate = np.clip(candidate, L, U)
                F_cand = self.func(candidate)

                candidates = [X[i], Y_A, Y_C, candidate]
                values = [F[i], F_A, F_C, F_cand]
                best_idx = int(np.argmin(values))
                newX[i] = candidates[best_idx]; newF[i] = values[best_idx]
                if success > 0: it_cf += 1
                it_reg += M_R[i]

            F_old = F.copy(); X = newX; F = newF
            improved = F < PF; P[improved] = X[improved]; PF[improved] = F[improved]
            k = int(np.argmin(PF))
            if PF[k] < GF: G = P[k].copy(); GF = PF[k]

            convergence[t + 1] = GF
            expl_cnt[t] = it_expl; xplt_cnt[t] = it_xplt
            cf_cnt[t] = it_cf; regret_avg[t] = it_reg / N

        return NCROResult(G, float(GF), convergence[1:], expl_cnt, xplt_cnt,
                          cf_cnt, regret_avg, q_hist)


class NCRO_NoMomentum:
    """
    NCRO with regret-aware momentum DISABLED.

    Changes from Full:
      - The momentum term w·V_i is removed from the motion equation
      - Motion becomes: x_i + α(1+M_R)·E + β(1-M_R)·H + γ·C·D_C
        (identical to original professor equation without momentum addition)
      - All other components (regret, counterfactual, adaptive q, EMA memory,
        scout mode, diversity recovery) remain identical to Full

    This tests whether the regret-aware momentum contributes to
    directional continuity and convergence quality.
    """

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.seed = seed
        for k, v in _NCRO_DEFAULTS.items():
            setattr(self, k, v)

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U, eps = self.L, self.U, self.epsilon
        search_range = U - L

        X = rng.uniform(L, U, size=(N, D))
        F = np.array([self.func(X[i]) for i in range(N)])
        P = X.copy(); PF = F.copy()
        g_idx = int(np.argmin(F)); G = P[g_idx].copy(); GF = F[g_idx]

        M_R = np.zeros(N)
        C_mem = np.zeros(N)
        progress_mem = np.zeros(N)

        convergence = np.zeros(T + 1); convergence[0] = GF
        expl_cnt = np.zeros(T); xplt_cnt = np.zeros(T)
        cf_cnt = np.zeros(T); regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        for t in range(T):
            tau = t / max(1, T - 1)
            alpha_t = self.alpha_max * (1 - tau) + self.alpha_min * tau
            beta_t = self.beta_min * (1 - tau) + self.beta_max * tau
            gamma_t = self.gamma_max * (1 - tau) + self.gamma_min * tau
            sigma_t = 1.0 * (1 - tau) + 0.01 * tau
            q0_t = self.q_max * (1 - tau) + self.q_min * tau

            pop_diversity = np.mean(np.std(X, axis=0)) / (search_range + eps)

            newX = np.empty_like(X); newF = np.empty(N)
            it_expl = it_xplt = it_cf = 0; it_reg = 0.0

            for i in range(N):
                # Regret-driven exploration (same as Full)
                agent_stuck = (M_R[i] > 0.25 and progress_mem[i] < 0.01)
                diversity_collapsed = (pop_diversity < 0.005)

                if agent_stuck or diversity_collapsed:
                    scout_type = rng.integers(4)
                    center = (L + U) / 2.0
                    if scout_type == 0:
                        target = np.clip(2.0 * center - G, L, U)
                    elif scout_type == 1:
                        target = rng.uniform(L, U, D)
                    elif scout_type == 2:
                        target = np.clip(2.0 * center - P[i], L, U)
                    else:
                        r_agent = rng.integers(N)
                        target = P[r_agent].copy()
                    E_i = target - X[i]
                    regret_scale = 0.3 + 0.7 * M_R[i]
                    E_i *= regret_scale
                else:
                    r1, r2 = rng.choice(N, 2, replace=False)
                    E_i = X[r1] - X[r2]
                    min_step = M_R[i] * 0.02 * search_range
                    E_norm = np.linalg.norm(E_i)
                    if E_norm < min_step and E_norm > eps:
                        E_i = E_i * (min_step / E_norm)

                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                q_i = q0_t + self.eta_R * M_R[i] + self.eta_C * C_mem[i] - self.eta_P * progress_mem[i]
                q_i = float(np.clip(q_i, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i
                # Regret-modulated position-relative noise + dimension-selective perturbation
                dist_to_best = np.linalg.norm(X[i] - G)
                regret_boost = 1.0 + M_R[i] * 5.0
                noise_scale = min(max(dist_to_best * regret_boost, eps),
                                  search_range / np.sqrt(D))
                max_dims = max(1, int(D * (1 - 0.7 * tau)))
                n_dims = rng.integers(1, max_dims + 1)
                dims = rng.choice(D, n_dims, replace=False)
                noise = np.zeros(D)
                noise[dims] = (0.02 / np.sqrt(D)) * sigma_t * noise_scale * rng.standard_normal(n_dims)
                Y_A = np.clip(Y_A + noise, L, U)
                Y_C = np.clip(Y_C, L, U)
                F_A = self.func(Y_A); F_C = self.func(Y_C)

                regret = max(0.0, F_A - F_C) / (abs(F_A) + abs(F_C) + eps)
                regret = float(np.clip(regret, 0, 1))
                success = float(F_C < F_A - eps)
                M_R[i] = self.rho * M_R[i] + (1 - self.rho) * regret
                C_mem[i] = self.rho_c * C_mem[i] + (1 - self.rho_c) * success

                D_C = Y_C - X[i]

                # ====================================================
                # MOTION EQUATION WITHOUT MOMENTUM (key difference)
                # No w·V_i term — everything else identical to Full
                # ====================================================
                candidate = (
                    X[i]
                    # NO momentum: + w_momentum * V_i
                    + alpha_t * (1 + M_R[i]) * E_i
                    + beta_t * (1 - M_R[i]) * H_i
                    + gamma_t * C_mem[i] * D_C
                )

                if rng.random() < q_i:
                    candidate = 0.75 * candidate + 0.25 * Y_A; it_expl += 1
                else:
                    candidate = 0.75 * candidate + 0.25 * Y_C; it_xplt += 1

                candidate = np.clip(candidate, L, U)
                F_cand = self.func(candidate)

                # Diversity-aware selection (same as Full)
                if pop_diversity < 0.01 and M_R[i] > 0.2:
                    move_candidates = [Y_A, Y_C, candidate]
                    move_values = [F_A, F_C, F_cand]
                    best_move = int(np.argmin(move_values))
                    newX[i] = move_candidates[best_move]
                    newF[i] = move_values[best_move]
                else:
                    candidates = [X[i], Y_A, Y_C, candidate]
                    values = [F[i], F_A, F_C, F_cand]
                    best_idx = int(np.argmin(values))
                    newX[i] = candidates[best_idx]
                    newF[i] = values[best_idx]

                if success > 0: it_cf += 1
                it_reg += M_R[i]

            F_old = F.copy()
            X = newX; F = newF
            improved = F < PF; P[improved] = X[improved]; PF[improved] = F[improved]
            k = int(np.argmin(PF))
            if PF[k] < GF: G = P[k].copy(); GF = PF[k]

            # Diversity recovery (same as Full)
            if pop_diversity < 0.002 and tau < 0.8:
                n_reset = max(1, N // 5)
                worst_idx = np.argsort(F)[-n_reset:]
                for wi in worst_idx:
                    X[wi] = rng.uniform(L, U, D)
                    F[wi] = self.func(X[wi])

            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0)

            convergence[t + 1] = GF
            expl_cnt[t] = it_expl; xplt_cnt[t] = it_xplt
            cf_cnt[t] = it_cf; regret_avg[t] = it_reg / N

        return NCROResult(G, float(GF), convergence[1:], expl_cnt, xplt_cnt,
                          cf_cnt, regret_avg, q_hist)


class NCRO_NoRegretMemory:
    """
    NCRO with regret memory accumulation DISABLED.

    Changes from Full:
      - Uses INSTANTANEOUS regret and CF success (from previous iteration)
        instead of EMA-smoothed memories
      - Motion uses instant_R instead of M_R in force modulation
      - q uses instant signals instead of accumulated memories

    This tests whether the EMA memory mechanism (learning from history)
    adds value over reacting to only the most recent feedback.
    """

    def __init__(self, objective_function: Callable, dimension: int, bounds: tuple,
                 population_size: int = 30, max_iter: int = 500, seed: int | None = None, **kwargs):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.seed = seed
        for k, v in _NCRO_DEFAULTS.items():
            setattr(self, k, v)

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U, eps = self.L, self.U, self.epsilon

        X = rng.uniform(L, U, size=(N, D))
        F = np.array([self.func(X[i]) for i in range(N)])
        P = X.copy(); PF = F.copy()
        g_idx = int(np.argmin(F)); G = P[g_idx].copy(); GF = F[g_idx]

        # INSTANTANEOUS signals (replace EMA memories)
        instant_R = np.zeros(N)    # Last iteration's normalized regret
        instant_SC = np.zeros(N)   # Last iteration's CF success (0 or 1)
        progress_mem = np.zeros(N)

        convergence = np.zeros(T + 1); convergence[0] = GF
        expl_cnt = np.zeros(T); xplt_cnt = np.zeros(T)
        cf_cnt = np.zeros(T); regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        for t in range(T):
            tau = t / max(1, T - 1)
            alpha_t = self.alpha_max * (1 - tau) + self.alpha_min * tau
            beta_t = self.beta_min * (1 - tau) + self.beta_max * tau
            gamma_t = self.gamma_max * (1 - tau) + self.gamma_min * tau
            sigma_t = 1.0 * (1 - tau) + 0.01 * tau
            q0_t = self.q_max * (1 - tau) + self.q_min * tau

            newX = np.empty_like(X); newF = np.empty(N)
            it_expl = it_xplt = it_cf = 0; it_reg = 0.0

            for i in range(N):
                r1, r2 = rng.choice(N, 2, replace=False)
                E_i = X[r1] - X[r2]
                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                # INSTANTANEOUS signals in q (no EMA smoothing)
                q_i = (q0_t
                       + self.eta_R * instant_R[i]
                       + self.eta_C * instant_SC[i]
                       - self.eta_P * progress_mem[i])
                q_i = float(np.clip(q_i, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i
                # Regret-modulated position-relative noise + dimension-selective perturbation
                dist_to_best = np.linalg.norm(X[i] - G)
                regret_boost = 1.0 + instant_R[i] * 5.0
                noise_scale = min(max(dist_to_best * regret_boost, eps),
                                  (U - L) / np.sqrt(D))
                max_dims = max(1, int(D * (1 - 0.7 * tau)))
                n_dims = rng.integers(1, max_dims + 1)
                dims = rng.choice(D, n_dims, replace=False)
                noise = np.zeros(D)
                noise[dims] = (0.02 / np.sqrt(D)) * sigma_t * noise_scale * rng.standard_normal(n_dims)
                Y_A = np.clip(Y_A + noise, L, U)
                Y_C = np.clip(Y_C, L, U)
                F_A = self.func(Y_A); F_C = self.func(Y_C)

                # Compute instantaneous regret + success (no EMA, stored directly)
                regret = max(0.0, F_A - F_C) / (abs(F_A) + abs(F_C) + eps)
                regret = float(np.clip(regret, 0, 1))
                success = float(F_C < F_A - eps)

                # Store raw instant values (NO EMA accumulation)
                instant_R[i] = regret
                instant_SC[i] = success

                D_C = Y_C - X[i]

                # Motion using INSTANTANEOUS regret in forces
                candidate = (
                    X[i]
                    + alpha_t * (1 + instant_R[i]) * E_i
                    + beta_t * (1 - instant_R[i]) * H_i
                    + gamma_t * instant_SC[i] * D_C
                )

                if rng.random() < q_i:
                    candidate = 0.75 * candidate + 0.25 * Y_A; it_expl += 1
                else:
                    candidate = 0.75 * candidate + 0.25 * Y_C; it_xplt += 1

                candidate = np.clip(candidate, L, U)
                F_cand = self.func(candidate)

                candidates = [X[i], Y_A, Y_C, candidate]
                values = [F[i], F_A, F_C, F_cand]
                best_idx = int(np.argmin(values))
                newX[i] = candidates[best_idx]; newF[i] = values[best_idx]
                if success > 0: it_cf += 1
                it_reg += regret

            F_old = F.copy(); X = newX; F = newF
            improved = F < PF; P[improved] = X[improved]; PF[improved] = F[improved]
            k = int(np.argmin(PF))
            if PF[k] < GF: G = P[k].copy(); GF = PF[k]

            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0)

            convergence[t + 1] = GF
            expl_cnt[t] = it_expl; xplt_cnt[t] = it_xplt
            cf_cnt[t] = it_cf; regret_avg[t] = it_reg / N

        return NCROResult(G, float(GF), convergence[1:], expl_cnt, xplt_cnt,
                          cf_cnt, regret_avg, q_hist)
