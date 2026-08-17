"""
Ablation Study Variants for NCRO.

Each variant uses the SAME V2 motion framework as the full optimizer,
with exactly ONE component disabled to isolate its contribution.

V2 Full motion equation (reference):
  candidate = x_i + α(t)(1+M_R)·E_i + β(t)(1-M_R)·H_i + γ(t)·C·D_C
  Blend: 0.75·candidate + 0.25·Y_A/Y_C
  Selection: greedy among {x_i, Y_A, Y_C, candidate}

Variants:
  1. NCRO_NoRegret           — M_R forced to 0 everywhere (forces + q)
  2. NCRO_NoCounterfactual   — No Y_C, no D_C, no regret, no C_mem
  3. NCRO_NoAdaptiveEE       — q = q0(t) only (no η_R, η_C, η_P)
  4. NCRO_NoRegretMemory     — Instantaneous regret (no EMA smoothing)
"""

import numpy as np
from typing import Callable
from .optimizer import NCROResult


# ──────────────────────────────────────────────────────────────────
# Shared V2 default parameters (must match optimizer.py exactly)
# ──────────────────────────────────────────────────────────────────
_V2_DEFAULTS = dict(
    c1=1.5, c2=1.5,
    alpha_max=2.0, alpha_min=0.15,
    beta_min=0.15, beta_max=1.8,
    gamma_max=1.0, gamma_min=0.05,
    rho=0.70, rho_c=0.60,
    eta_R=0.20, eta_C=0.12, eta_P=0.70,
    q_min=0.05, q_max=0.95,
    epsilon=1e-12,
)


class NCRO_NoRegret:
    """
    V2 with regret signal DISABLED.

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
        for k, v in _V2_DEFAULTS.items():
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
                noise = 0.02 * sigma_t * (U - L) / np.sqrt(D) * rng.standard_normal(D)
                Y_A = np.clip(Y_A + noise, L, U)
                Y_C = np.clip(Y_C, L, U)
                F_A = self.func(Y_A); F_C = self.func(Y_C)

                # CF success — regret is computed but NOT accumulated
                success = float(F_C < F_A - eps)
                C_mem[i] = self.rho_c * C_mem[i] + (1 - self.rho_c) * success

                D_C = Y_C - X[i]

                # V2 motion WITHOUT regret modulation (M_R=0)
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


class NCRO_NoCounterfactual:
    """
    V2 with counterfactual candidate DISABLED.

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
        for k, v in _V2_DEFAULTS.items():
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
                noise = 0.02 * sigma_t * (U - L) / np.sqrt(D) * rng.standard_normal(D)
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
    V2 with adaptive exploration-exploitation balance DISABLED.

    Changes from Full:
      - q = q0(t) = q_max·(1-τ) + q_min·τ   (fixed linear schedule)
      - η_R, η_C, η_P have NO influence on q
      - M_R and C_mem ARE still computed and DO modulate the V2 forces
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
        for k, v in _V2_DEFAULTS.items():
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
                noise = 0.02 * sigma_t * (U - L) / np.sqrt(D) * rng.standard_normal(D)
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

                # V2 motion with regret-modulated forces (M_R still active here)
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


class NCRO_NoRegretMemory:
    """
    V2 with regret memory accumulation DISABLED.

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
        for k, v in _V2_DEFAULTS.items():
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
                noise = 0.02 * sigma_t * (U - L) / np.sqrt(D) * rng.standard_normal(D)
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

                # V2 motion using INSTANTANEOUS regret in forces
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
