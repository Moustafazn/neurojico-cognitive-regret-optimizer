"""
Neurojico Cognitive Regret Optimizer (NCRO)

Motion equation:

  candidate = x_i
            + w * V_i                                 # regret-aware momentum
            + alpha(t) * (1 + M_R(i)) * E_i          # regret amplifies exploration
            + beta(t)  * (1 - M_R(i)) * H_i          # regret dampens exploitation
            + gamma(t) * C(i) * D_C(i)               # counterfactual force

Key mechanisms:
  - OBL initialization: opposition-based population seeding (Tizhoosh 2005)
  - Regret-driven exploration with Levy flight scout mode
  - Regret-modulated position-relative noise with D-adaptive coefficient
  - Dimension-selective counterfactual perturbation (Van Hoeck 2015)
  - Stagnation-triggered opposition jump (Rahnamayan et al. 2008)
  - Late-stage noise suppression for machine-precision convergence
  - Diversity-aware selection to prevent premature convergence
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class NCROResult:
    """Container for NCRO optimization results."""
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: np.ndarray
    exploration_counts: np.ndarray
    exploitation_counts: np.ndarray
    counterfactual_success_counts: np.ndarray
    regret_values: np.ndarray
    q_values: np.ndarray


class NCROOptimizer:
    """
    Neurojico Cognitive Regret Optimizer (NCRO).

    Key innovations:
      - Regret directly modulates exploration/exploitation forces
      - Counterfactual direction (D_C) acts as an additional force
      - Greedy selection among 4 candidates per iteration
      - Enhanced exploration with minimum step size and long-distance vision
      - Diversity-aware selection to prevent premature convergence
    """

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_iter: int = 500,
        c1: float = 2.0,
        c2: float = 2.0,
        alpha_max: float = 2.0,
        alpha_min: float = 0.15,
        beta_min: float = 0.15,
        beta_max: float = 1.8,
        gamma_max: float = 1.0,
        gamma_min: float = 0.05,
        rho: float = 0.90,
        rho_c: float = 0.90,
        eta_R: float = 0.30,
        eta_C: float = 0.20,
        eta_P: float = 0.20,
        q_min: float = 0.10,
        q_max: float = 0.90,
        w_max: float = 0.90,
        epsilon: float = 1e-12,
        d_min: float = 0.05,
        seed: int | None = None,
    ):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.c1 = c1
        self.c2 = c2
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.rho = rho
        self.rho_c = rho_c
        self.eta_R = eta_R
        self.eta_C = eta_C
        self.eta_P = eta_P
        self.q_min = q_min
        self.q_max = q_max
        self.epsilon = epsilon
        self.seed = seed

    # ------------------------------------------------------------------
    #  Initialization
    # ------------------------------------------------------------------

    def _initialize_population(self, rng):
        """OBL init: N random + N opposite, keep best N (Tizhoosh 2005)."""
        N, D, L, U = self.N, self.D, self.L, self.U
        X_rand = rng.uniform(L, U, size=(N, D))
        X_opp = L + U - X_rand
        X_all = np.vstack([X_rand, X_opp])
        F_all = np.array([self.func(X_all[j]) for j in range(2 * N)])
        idx = np.argsort(F_all)[:N]
        return X_all[idx].copy(), F_all[idx].copy()

    def _compute_schedules(self, tau):
        """Time-dependent coefficients for iteration tau in [0,1]."""
        alpha_t = self.alpha_max * (1 - tau) + self.alpha_min * tau
        beta_t = self.beta_min * (1 - tau) + self.beta_max * tau
        gamma_t = self.gamma_max * (1 - tau) + self.gamma_min * tau
        sigma_t = 1.0 * (1 - tau) + 0.01 * tau
        q0_t = self.q_max * (1 - tau) + self.q_min * tau
        return alpha_t, beta_t, gamma_t, sigma_t, q0_t

    # ------------------------------------------------------------------
    #  Per-agent helpers
    # ------------------------------------------------------------------

    def _compute_exploration(self, i, X, P, G, M_R, progress_mem,
                             pop_diversity, search_range, rng):
        """Exploration direction E_i with multi-directional scout mode."""
        N, D = self.N, self.D
        L, U, eps = self.L, self.U, self.epsilon
        agent_stuck = (M_R[i] > 0.25 and progress_mem[i] < 0.01)

        if agent_stuck or pop_diversity < 0.005:
            scout = rng.integers(5)
            center = (L + U) / 2.0
            if scout == 0:
                target = np.clip(2.0 * center - G, L, U)
            elif scout == 1:
                target = rng.uniform(L, U, D)
            elif scout == 2:
                target = np.clip(2.0 * center - P[i], L, U)
            elif scout == 3:
                target = P[rng.integers(N)].copy()
            else:
                target = self._levy_flight_target(G, search_range, D, rng)
            E_i = (target - X[i]) * (0.3 + 0.7 * M_R[i])
        else:
            r1, r2 = rng.choice(N, 2, replace=False)
            E_i = X[r1] - X[r2]
            min_step = M_R[i] * 0.02 * search_range
            E_norm = np.linalg.norm(E_i)
            if E_norm < min_step and E_norm > eps:
                E_i = E_i * (min_step / E_norm)
        return E_i

    @staticmethod
    def _levy_flight_target(G, search_range, D, rng):
        """Levy flight step from G (Mantegna 1994)."""
        beta_levy = 1.5
        sigma_u = (
            math.gamma(1 + beta_levy) * np.sin(np.pi * beta_levy / 2)
            / (math.gamma((1 + beta_levy) / 2)
               * beta_levy * 2 ** ((beta_levy - 1) / 2))
        ) ** (1 / beta_levy)
        u = rng.normal(0, sigma_u, D)
        v = rng.normal(0, 1, D)
        step = 0.01 * u / (np.abs(v) ** (1 / beta_levy))
        return G + step * search_range

    def _apply_noise(self, Y_A, X_i, G, M_R_i, sigma_t, tau,
                     pop_diversity, search_range, rng):
        """Regret-modulated, D-adaptive, dimension-selective noise.

        Three interacting mechanisms:
          1. Position-relative scale: agents near G get smaller noise.
          2. D-adaptive coefficient: 0.02/sqrt(D) normalizes across D.
          3. Late-stage suppression: noise=0 when converged (tau>0.9).
        """
        D = self.D
        L, U, eps = self.L, self.U, self.epsilon

        # Late-stage suppression
        if tau > 0.9 and pop_diversity < 0.01:
            return np.clip(Y_A, L, U)

        # Position-relative noise scale with regret boost
        dist_to_best = np.linalg.norm(X_i - G)
        noise_scale = min(max(dist_to_best * (1.0 + M_R_i * 5.0), eps),
                          search_range / np.sqrt(D))

        # Dimension-selective perturbation
        max_dims = max(1, int(D * (1 - 0.7 * tau)))
        n_dims = rng.integers(1, max_dims + 1)
        dims = rng.choice(D, n_dims, replace=False)

        # D-adaptive coefficient
        noise_coeff = 0.02 / np.sqrt(D)
        noise = np.zeros(D)
        noise[dims] = noise_coeff * sigma_t * noise_scale * rng.standard_normal(n_dims)
        return np.clip(Y_A + noise, L, U)

    def _update_memories(self, i, F_A, F_C, M_R, C_mem):
        """Cognitive regret computation and EMA memory update."""
        eps = self.epsilon
        regret = float(np.clip(
            max(0.0, F_A - F_C) / (abs(F_A) + abs(F_C) + eps), 0, 1))
        success = float(F_C < F_A - eps)
        M_R[i] = self.rho * M_R[i] + (1 - self.rho) * regret
        C_mem[i] = self.rho_c * C_mem[i] + (1 - self.rho_c) * success
        return success

    def _build_candidate(self, X_i, X_prev_i, E_i, H_i, D_C, M_R_i,
                         C_mem_i, alpha_t, beta_t, gamma_t, tau):
        """Build motion-equation candidate with regret-aware momentum."""
        V_i = X_i - X_prev_i
        w_momentum = (1 - M_R_i) * 0.4 * (1 - tau)
        return (X_i
                + w_momentum * V_i
                + alpha_t * (1 + M_R_i) * E_i
                + beta_t * (1 - M_R_i) * H_i
                + gamma_t * C_mem_i * D_C)

    def _select_survivor(self, X_i, F_i, Y_A, F_A, Y_C, F_C,
                         candidate, F_cand, M_R_i, pop_diversity):
        """Greedy 4-way selection with diversity-aware forced movement."""
        if pop_diversity < 0.01 and M_R_i > 0.2:
            cs = [Y_A, Y_C, candidate]
            vs = [F_A, F_C, F_cand]
        else:
            cs = [X_i, Y_A, Y_C, candidate]
            vs = [F_i, F_A, F_C, F_cand]
        best = int(np.argmin(vs))
        return cs[best], vs[best]

    # ------------------------------------------------------------------
    #  Population-level helpers
    # ------------------------------------------------------------------

    def _stagnation_opposition_jump(self, G, GF, X, F, P, PF,
                                    stag_counter, last_GF, tau):
        """Try opposite of G when stagnated >=50 iters (Rahnamayan 2008)."""
        L, U = self.L, self.U
        if GF < last_GF:
            stag_counter = 0
            last_GF = GF
        else:
            stag_counter += 1
        if stag_counter >= 50 and tau < 0.95:
            G_opp = np.clip(L + U - G, L, U)
            F_opp = self.func(G_opp)
            if F_opp < GF:
                G = G_opp.copy()
                GF = F_opp
                worst_i = int(np.argmax(F))
                X[worst_i] = G_opp.copy()
                F[worst_i] = F_opp
                P[worst_i] = G_opp.copy()
                PF[worst_i] = F_opp
            stag_counter = 0
        return G, GF, stag_counter, last_GF

    def _recover_diversity(self, X, F, pop_diversity, tau, rng):
        """Reinitialize worst 20% when diversity drops critically."""
        if pop_diversity < 0.002 and tau < 0.8:
            n_reset = max(1, self.N // 5)
            for wi in np.argsort(F)[-n_reset:]:
                X[wi] = rng.uniform(self.L, self.U, self.D)
                F[wi] = self.func(X[wi])

    # ------------------------------------------------------------------
    #  Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U = self.L, self.U
        eps = self.epsilon
        search_range = U - L

        # Initialize population with OBL
        X, F = self._initialize_population(rng)
        P = X.copy()
        PF = F.copy()
        g_idx = int(np.argmin(F))
        G = X[g_idx].copy()
        GF = F[g_idx]

        # Memories
        M_R = np.zeros(N)
        C_mem = np.zeros(N)
        progress_mem = np.zeros(N)
        X_prev = X.copy()

        # Stagnation tracking
        stag_counter = 0
        last_GF = GF

        # Metrics
        convergence = np.zeros(T + 1)
        convergence[0] = GF
        expl_cnt = np.zeros(T)
        xplt_cnt = np.zeros(T)
        cf_cnt = np.zeros(T)
        regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        for t in range(T):
            tau = t / max(1, T - 1)
            alpha_t, beta_t, gamma_t, sigma_t, q0_t = self._compute_schedules(tau)
            pop_diversity = np.mean(np.std(X, axis=0)) / (search_range + eps)

            newX = np.empty_like(X)
            newF = np.empty(N)
            it_expl = it_xplt = it_cf = 0
            it_regret = 0.0

            for i in range(N):
                # Exploration direction
                E_i = self._compute_exploration(
                    i, X, P, G, M_R, progress_mem,
                    pop_diversity, search_range, rng)

                # Exploitation direction (PSO-style)
                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                # Adaptive q
                q_i = (q0_t
                       + self.eta_R * M_R[i]
                       + self.eta_C * C_mem[i]
                       - self.eta_P * progress_mem[i])
                q_i = float(np.clip(q_i, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                # Generate actual & counterfactual candidates
                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i

                # Apply noise to actual candidate
                Y_A = self._apply_noise(
                    Y_A, X[i], G, M_R[i], sigma_t, tau,
                    pop_diversity, search_range, rng)
                Y_C = np.clip(Y_C, L, U)
                F_A = self.func(Y_A)
                F_C = self.func(Y_C)

                # Update memories
                success = self._update_memories(i, F_A, F_C, M_R, C_mem)

                # Build candidate via motion equation
                D_C = Y_C - X[i]
                candidate = self._build_candidate(
                    X[i], X_prev[i], E_i, H_i, D_C, M_R[i],
                    C_mem[i], alpha_t, beta_t, gamma_t, tau)

                # Candidate blend
                if rng.random() < q_i:
                    candidate = 0.75 * candidate + 0.25 * Y_A
                    it_expl += 1
                else:
                    candidate = 0.75 * candidate + 0.25 * Y_C
                    it_xplt += 1

                candidate = np.clip(candidate, L, U)
                F_cand = self.func(candidate)

                # Greedy selection
                newX[i], newF[i] = self._select_survivor(
                    X[i], F[i], Y_A, F_A, Y_C, F_C,
                    candidate, F_cand, M_R[i], pop_diversity)

                if success > 0:
                    it_cf += 1
                it_regret += M_R[i]

            # Update population
            F_old = F.copy()
            X_prev = X.copy()
            X = newX
            F = newF

            # Update personal & global bests
            improved = F < PF
            P[improved] = X[improved]
            PF[improved] = F[improved]
            k = int(np.argmin(PF))
            if PF[k] < GF:
                G = P[k].copy()
                GF = PF[k]

            # Stagnation-triggered opposition jump
            G, GF, stag_counter, last_GF = self._stagnation_opposition_jump(
                G, GF, X, F, P, PF, stag_counter, last_GF, tau)

            # Diversity recovery
            self._recover_diversity(X, F, pop_diversity, tau, rng)

            # Update progress memory
            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(
                accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0)

            # Record metrics
            convergence[t + 1] = GF
            expl_cnt[t] = it_expl
            xplt_cnt[t] = it_xplt
            cf_cnt[t] = it_cf
            regret_avg[t] = it_regret / N

        return NCROResult(
            best_position=G,
            best_fitness=float(GF),
            convergence_curve=convergence[1:],
            exploration_counts=expl_cnt,
            exploitation_counts=xplt_cnt,
            counterfactual_success_counts=cf_cnt,
            regret_values=regret_avg,
            q_values=q_hist,
        )
