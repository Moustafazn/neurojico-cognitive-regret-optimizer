"""
Neurojico Cognitive Regret Optimizer (NCRO).

Motion equation:
  x_i' = x_i + w·V_i + α(1+M_R)·E_i + β(1-M_R)·H_i + γ·C·D_C

Architecture:
  1. OBL initialization (Tizhoosh 2005)
  2. Budget-driven main loop: while FE_count < MaxFEs (CEC convention)
  3. Adaptive hybrid evaluation:
       Phase 1 (CV ≥ 1/√N) — rank-based regret, 1 FE/agent
       Phase 2 (CV < 1/√N) — direct counterfactual, 3 FEs/agent
  4. Regret-modulated exploration with Lévy flight scout mode
  5. Dimension-selective noise (D-adaptive, position-relative)
  6. Stagnation-triggered opposition jump (Rahnamayan et al. 2008)
  7. Late-stage noise suppression for machine-precision convergence
  8. Diversity-aware selection and recovery
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


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
    """Neurojico Cognitive Regret Optimizer (NCRO).

    Budget-driven ``while FE_count < MaxFEs`` loop with adaptive hybrid
    evaluation.  Phase 1 uses rank-based regret (1 FE/agent); Phase 2
    switches to direct counterfactual evaluation (3 FEs/agent) when
    CV(F) drops below 1/√N (David & Nagaraja 2003).
    """

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_fes: int | None = None,
        max_iter: int | None = None,
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
        w_base: float = 0.40,
        noise_base: float = 0.02,
        noise_regret_boost: float = 5.0,
        dim_decay: float = 0.7,
        stag_limit: int = 50,
        convergence_threshold: float = 0.01,
        epsilon: float = 1e-12,
        seed: int | None = None,
        repair_fn: Optional[Callable] = None,
    ):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.repair_fn = repair_fn
        self.N = population_size
        # Budget: prefer max_fes; fall back to max_iter-based estimate.
        if max_fes is not None:
            self.max_fes = max_fes
        elif max_iter is not None:
            self.max_fes = 2 * population_size * max_iter + 2 * population_size
        else:
            self.max_fes = 45000  # sensible default
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
        self.w_base = w_base
        self.noise_base = noise_base
        self.noise_regret_boost = noise_regret_boost
        self.dim_decay = dim_decay
        self.stag_limit = stag_limit
        self.convergence_threshold = convergence_threshold
        self.epsilon = epsilon
        self.seed = seed

    # ------------------------------------------------------------------
    #  Internal FE counter
    # ------------------------------------------------------------------

    def _eval(self, x):
        """Evaluate objective and increment the internal FE counter."""
        self._fe_count += 1
        return self.func(x)

    def _budget_left(self):
        """True if at least one FE remains in the budget."""
        return self._fe_count < self.max_fes

    # ------------------------------------------------------------------
    #  Constraint handling
    # ------------------------------------------------------------------

    def _repair(self, x):
        """Project candidate(s) onto the feasible set.

        Uses the injected repair_fn when available (e.g. microgrid
        equality-constraint projection), otherwise falls back to
        simple box-clipping.  Works for both single vectors and 2-D
        population arrays.
        """
        if self.repair_fn is not None:
            return self.repair_fn(x)
        return np.clip(x, self.L, self.U)

    # ------------------------------------------------------------------
    #  Initialization
    # ------------------------------------------------------------------

    def _initialize_population(self, rng):
        """OBL init: N random + N opposite, keep best N (Tizhoosh 2005)."""
        N, D, L, U = self.N, self.D, self.L, self.U
        X_rand = self._repair(rng.uniform(L, U, size=(N, D)))
        X_opp = self._repair(L + U - X_rand)
        X_all = np.vstack([X_rand, X_opp])
        F_all = np.array([self._eval(X_all[j]) for j in range(2 * N)])
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

    @staticmethod
    def _scalar_range(search_range):
        """Collapse per-dimension ranges to a single scalar."""
        sr = np.asarray(search_range)
        return float(np.mean(sr)) if sr.ndim > 0 else float(sr)

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
                target = self._repair(2.0 * center - G)
            elif scout == 1:
                target = self._repair(rng.uniform(L, U, D))
            elif scout == 2:
                target = self._repair(2.0 * center - P[i])
            elif scout == 3:
                target = P[rng.integers(N)].copy()
            else:
                sr = self._scalar_range(search_range)
                target = self._levy_flight_target(G, sr, D, rng)
            E_i = (target - X[i]) * (0.3 + 0.7 * M_R[i])
        else:
            r1, r2 = rng.choice(N, 2, replace=False)
            E_i = X[r1] - X[r2]
            sr = self._scalar_range(search_range)
            min_step = M_R[i] * 0.02 * sr
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

    def _apply_noise(self, Y, X_i, G, M_R_i, sigma_t, tau,
                     pop_diversity, search_range, rng):
        """Regret-modulated, D-adaptive, dimension-selective noise.

        Three interacting mechanisms:
          1. Position-relative scale: agents near G get smaller noise.
          2. D-adaptive coefficient: 0.02/sqrt(D) normalizes across D.
          3. Late-stage suppression: noise=0 when converged (tau>0.9).
        """
        D = self.D
        eps = self.epsilon

        # Late-stage suppression
        if tau > 0.9 and pop_diversity < self.convergence_threshold:
            return self._repair(Y)

        # Position-relative noise scale with regret boost
        sr = self._scalar_range(search_range)
        dist_to_best = np.linalg.norm(X_i - G)
        noise_scale = min(
            max(dist_to_best * (1.0 + M_R_i * self.noise_regret_boost), eps),
            sr / np.sqrt(D))

        # Dimension-selective perturbation
        max_dims = max(1, int(D * (1 - self.dim_decay * tau)))
        n_dims = rng.integers(1, max_dims + 1)
        dims = rng.choice(D, n_dims, replace=False)

        # D-adaptive coefficient
        noise_coeff = self.noise_base / np.sqrt(D)
        noise = np.zeros(D)
        noise[dims] = noise_coeff * sigma_t * noise_scale * rng.standard_normal(n_dims)
        return self._repair(Y + noise)

    @staticmethod
    def _rank_based_regret(i, F_old, F_new, N):
        """Estimate cognitive regret from ordinal rank change."""
        from scipy.stats import rankdata
        ranks_old = rankdata(F_old)
        ranks_new = rankdata(F_new)
        # Positive rank_change → agent got worse → regret
        rank_change = (ranks_new[i] - ranks_old[i]) / max(N - 1, 1)
        regret = float(np.clip(rank_change, 0.0, 1.0))
        cf_success = 1.0 if ranks_new[i] < ranks_old[i] else 0.0
        return regret, cf_success

    def _update_memories_rank(self, i, F_old, F_new, M_R, C_mem):
        """Update regret and CF-success memories from rank feedback."""
        regret, cf_success = self._rank_based_regret(
            i, F_old, F_new, self.N)
        M_R[i] = self.rho * M_R[i] + (1 - self.rho) * regret
        C_mem[i] = self.rho_c * C_mem[i] + (1 - self.rho_c) * cf_success
        return cf_success

    def _build_candidate(self, X_i, X_prev_i, E_i, H_i, D_C, M_R_i,
                         C_mem_i, alpha_t, beta_t, gamma_t, tau):
        """Build motion-equation candidate with regret-aware momentum."""
        V_i = X_i - X_prev_i
        w_momentum = (1 - M_R_i) * self.w_base * (1 - tau)
        return (X_i
                + w_momentum * V_i
                + alpha_t * (1 + M_R_i) * E_i
                + beta_t * (1 - M_R_i) * H_i
                + gamma_t * C_mem_i * D_C)

    def _select_survivor(self, X_i, F_i, candidate, F_cand,
                         M_R_i, pop_diversity):
        """Greedy 2-way selection with diversity-aware forced movement."""
        if pop_diversity < self.convergence_threshold and M_R_i > 0.2:
            # Force acceptance to escape stagnation
            return candidate, F_cand
        if F_cand <= F_i:
            return candidate, F_cand
        return X_i, F_i

    # ------------------------------------------------------------------
    #  Phase 2 (direct counterfactual) helpers
    # ------------------------------------------------------------------

    def _build_Y_A_Y_C(self, X_i, q_i, alpha_t, beta_t, E_i, H_i):
        """Construct explore-biased Y_A and exploit-biased Y_C vectors."""
        Y_A = self._repair(X_i + q_i * alpha_t * E_i
                           + (1 - q_i) * beta_t * H_i)
        Y_C = self._repair(X_i + (1 - q_i) * alpha_t * E_i
                           + q_i * beta_t * H_i)
        return Y_A, Y_C

    @staticmethod
    def _direct_regret(F_A, F_C, eps):
        """Exact regret from evaluated explore / exploit candidates."""
        regret = max(0.0, F_A - F_C) / (abs(F_A) + abs(F_C) + eps)
        return float(np.clip(regret, 0.0, 1.0))

    def _update_memories_direct(self, i, F_A, F_C, M_R, C_mem):
        """Update memories using direct counterfactual comparison."""
        eps = self.epsilon
        regret = self._direct_regret(F_A, F_C, eps)
        cf_success = 1.0 if F_C < F_A - eps else 0.0
        M_R[i] = self.rho * M_R[i] + (1 - self.rho) * regret
        C_mem[i] = self.rho_c * C_mem[i] + (1 - self.rho_c) * cf_success
        return cf_success

    @staticmethod
    def _select_best_of_four(X_i, F_i, Y_A, F_A, Y_C, F_C,
                             candidate, F_cand):
        """Greedy 4-way selection — return best (position, fitness)."""
        positions = [X_i, Y_A, Y_C, candidate]
        values = [F_i, F_A, F_C, F_cand]
        best_idx = int(np.argmin(values))
        return positions[best_idx], values[best_idx]

    @staticmethod
    def _population_cv(F):
        """Coefficient of variation of population fitness values."""
        mean_f = np.mean(F)
        if abs(mean_f) < 1e-30:
            return 0.0
        return float(np.std(F) / abs(mean_f))

    # ------------------------------------------------------------------
    #  Population-level helpers
    # ------------------------------------------------------------------

    def _stagnation_opposition_jump(self, G, GF, X, F, P, PF,
                                    stag_counter, last_GF, tau):
        """Try opposite of G when stagnated (Rahnamayan 2008)."""
        L, U = self.L, self.U
        if GF < last_GF:
            stag_counter = 0
            last_GF = GF
        else:
            stag_counter += 1
        if stag_counter >= self.stag_limit and tau < 0.95:
            G_opp = self._repair(L + U - G)
            F_opp = self._eval(G_opp)
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
                X[wi] = self._repair(rng.uniform(self.L, self.U, self.D))
                F[wi] = self._eval(X[wi])

    # ------------------------------------------------------------------
    #  Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D = self.N, self.D
        L, U = self.L, self.U
        eps = self.epsilon
        max_fes = self.max_fes
        search_range = np.mean(U - L)

        # CV threshold for phase transition (David & Nagaraja 2003).
        # When CV(F) < 1/√N the expected spacing between adjacent
        # order statistics is dominated by noise → ranks unreliable.
        cv_threshold = 1.0 / math.sqrt(N)

        # Reset the internal FE counter for this optimization run.
        # Every evaluation routed through _eval() increments it.
        self._fe_count = 0

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

        # Phase flag — once switched to Phase 2, stays there.
        in_phase2 = False

        # Dynamic metric lists (iteration count unknown in advance).
        convergence_list = [GF]
        expl_list = []
        xplt_list = []
        cf_list = []
        regret_list = []
        q_list = []

        while self._budget_left():
            # tau tracks fraction of budget consumed (Turgut 2026).
            tau = min(self._fe_count / max(1, max_fes), 1.0)
            alpha_t, beta_t, gamma_t, sigma_t, q0_t = (
                self._compute_schedules(tau))
            pop_diversity = (
                np.mean(np.std(X, axis=0)) / (search_range + eps))

            # Adaptive phase transition (irreversible).
            if not in_phase2:
                cv = self._population_cv(F)
                if cv < cv_threshold:
                    in_phase2 = True

            newX = np.empty_like(X)
            newF = np.empty(N)
            it_expl = it_xplt = it_cf = 0
            it_regret = 0.0

            # Snapshot for Phase 1 rank regret
            F_old_snapshot = F.copy()
            q_row = np.zeros(N)

            for i in range(N):
                if not self._budget_left():
                    newX[i] = X[i]
                    newF[i] = F[i]
                    continue

                # Exploration direction
                E_i = self._compute_exploration(
                    i, X, P, G, M_R, progress_mem,
                    pop_diversity, search_range, rng)

                # Exploitation direction (PSO-style)
                u1, u2 = rng.random(), rng.random()
                H_i = (self.c1 * u1 * (P[i] - X[i])
                       + self.c2 * u2 * (G - X[i]))

                # Adaptive q
                q_i = (q0_t
                       + self.eta_R * M_R[i]
                       + self.eta_C * C_mem[i]
                       - self.eta_P * progress_mem[i])
                q_i = float(np.clip(q_i, self.q_min, self.q_max))
                q_row[i] = q_i

                if rng.random() < q_i:
                    it_expl += 1
                else:
                    it_xplt += 1

                if in_phase2:
                    # ── Phase 2: direct counterfactual (3 FEs) ──
                    Y_A, Y_C = self._build_Y_A_Y_C(
                        X[i], q_i, alpha_t, beta_t, E_i, H_i)
                    Y_A = self._apply_noise(
                        Y_A, X[i], G, M_R[i], sigma_t, tau,
                        pop_diversity, search_range, rng)
                    Y_A = self._repair(Y_A)
                    F_A = self._eval(Y_A)
                    F_C = self._eval(Y_C)

                    D_C = Y_C - X[i]
                    candidate = self._build_candidate(
                        X[i], X_prev[i], E_i, H_i, D_C, M_R[i],
                        C_mem[i], alpha_t, beta_t, gamma_t, tau)
                    candidate = self._apply_noise(
                        candidate, X[i], G, M_R[i], sigma_t, tau,
                        pop_diversity, search_range, rng)
                    candidate = self._repair(candidate)
                    F_cand = self._eval(candidate)

                    cf_success = self._update_memories_direct(
                        i, F_A, F_C, M_R, C_mem)
                    if cf_success > 0:
                        it_cf += 1
                    it_regret += M_R[i]

                    newX[i], newF[i] = self._select_best_of_four(
                        X[i], F[i], Y_A, F_A, Y_C, F_C,
                        candidate, F_cand)
                else:
                    # ── Phase 1: rank-based regret (1 FE) ──
                    D_C = (((1 - q_i) - q_i) * alpha_t * E_i
                           + (q_i - (1 - q_i)) * beta_t * H_i)
                    candidate = self._build_candidate(
                        X[i], X_prev[i], E_i, H_i, D_C, M_R[i],
                        C_mem[i], alpha_t, beta_t, gamma_t, tau)
                    candidate = self._apply_noise(
                        candidate, X[i], G, M_R[i], sigma_t, tau,
                        pop_diversity, search_range, rng)
                    candidate = self._repair(candidate)
                    F_cand = self._eval(candidate)
                    newX[i], newF[i] = self._select_survivor(
                        X[i], F[i], candidate, F_cand,
                        M_R[i], pop_diversity)

            # Phase 1: rank-based regret update (whole population)
            if not in_phase2:
                for i in range(N):
                    cf_success = self._update_memories_rank(
                        i, F_old_snapshot, newF, M_R, C_mem)
                    if cf_success > 0:
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

            # Record metrics for this iteration.
            convergence_list.append(GF)
            expl_list.append(it_expl)
            xplt_list.append(it_xplt)
            cf_list.append(it_cf)
            regret_list.append(it_regret / max(N, 1))
            q_list.append(q_row)

        return NCROResult(
            best_position=G,
            best_fitness=float(GF),
            convergence_curve=np.array(convergence_list[1:]),
            exploration_counts=np.array(expl_list),
            exploitation_counts=np.array(xplt_list),
            counterfactual_success_counts=np.array(cf_list),
            regret_values=np.array(regret_list),
            q_values=np.array(q_list) if q_list else np.empty((0, N)),
        )
