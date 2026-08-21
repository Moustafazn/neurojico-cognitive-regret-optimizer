"""
Neurojico Cognitive Regret Optimizer (NCRO)

Motion equation:

  candidate = x_i
            + alpha(t) * (1 + M_R(i)) * E_i        # regret amplifies exploration
            + beta(t)  * (1 - M_R(i)) * H_i         # regret dampens exploitation
            + gamma(t) * C(i) * D_C(i)               # counterfactual force

  Where:
    E_i  = x_r1 - x_r2                              (differential exploration)
    H_i  = c1*u1*(p_i - x_i) + c2*u2*(g - x_i)     (PSO-style exploitation)
    D_C  = Y_C - x_i                                (counterfactual direction)

  Enhanced mechanisms:
    - Regret-driven exploration: stuck agents scout distant regions, converging agents use standard differential
    - Regret-aware momentum: agents continue in successful direction (modulated by regret)
    - Diversity-aware selection: allows escape from local optima when diversity is low
"""

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

    def optimize(self) -> NCROResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U = self.L, self.U
        eps = self.epsilon
        search_range = U - L  # scalar (symmetric bounds assumed)

        # Initialize population
        X = rng.uniform(L, U, size=(N, D))
        F = np.array([self.func(X[i]) for i in range(N)])

        P = X.copy()          # Personal best positions
        PF = F.copy()         # Personal best fitness
        g_idx = int(np.argmin(F))
        G = X[g_idx].copy()
        GF = F[g_idx]

        # Memories
        M_R = np.zeros(N)           # Regret memory
        C_mem = np.zeros(N)         # Counterfactual success memory
        progress_mem = np.zeros(N)  # Fitness progress memory

        # Momentum: track previous positions for velocity
        X_prev = X.copy()           # Previous iteration positions

        # Tracking
        convergence = np.zeros(T + 1)
        convergence[0] = GF
        expl_cnt = np.zeros(T)
        xplt_cnt = np.zeros(T)
        cf_cnt = np.zeros(T)
        regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        for t in range(T):
            tau = t / max(1, T - 1)

            # Time-varying parameters
            alpha_t = self.alpha_max * (1 - tau) + self.alpha_min * tau
            beta_t = self.beta_min * (1 - tau) + self.beta_max * tau
            gamma_t = self.gamma_max * (1 - tau) + self.gamma_min * tau
            sigma_t = 1.0 * (1 - tau) + 0.01 * tau
            q0_t = self.q_max * (1 - tau) + self.q_min * tau

            # ─── Population diversity measurement ───
            # Normalized diversity: mean std across dimensions / search range
            pop_diversity = np.mean(np.std(X, axis=0)) / (search_range + eps)

            # Minimum exploration step size (decays with time, but always > 0)
            # This prevents the differential vector from collapsing to zero
            min_step_fraction = 0.05 * (1 - tau) + 0.001 * tau  # 5% early → 0.1% late
            min_step = min_step_fraction * search_range

            newX = np.empty_like(X)
            newF = np.empty(N)
            it_expl = 0
            it_xplt = 0
            it_cf = 0
            it_regret = 0.0

            for i in range(N):
                # ─── Regret-Driven Exploration Strategy ───
                # The exploration distance adapts to each agent's regret state:
                #   High regret + no progress = stuck agent → scout distant areas
                #   Low regret + good progress = converging → standard differential
                #
                # This uses the algorithm's own regret signal to decide WHEN
                # and HOW FAR to explore — consistent with NCRO's philosophy.

                agent_stuck = (M_R[i] > 0.25 and progress_mem[i] < 0.01)
                diversity_collapsed = (pop_diversity < 0.005)

                if agent_stuck or diversity_collapsed:
                    # ─── Long-Distance Exploration (multi-directional) ───
                    # Stuck agents scout DIFFERENT distant areas, not just one:
                    scout_type = rng.integers(4)

                    if scout_type == 0:
                        # Direction 1: Opposite of global best
                        center = (L + U) / 2.0
                        target = np.clip(2.0 * center - G, L, U)
                    elif scout_type == 1:
                        # Direction 2: Random region of search space
                        target = rng.uniform(L, U, D)
                    elif scout_type == 2:
                        # Direction 3: Opposite of personal best
                        center = (L + U) / 2.0
                        target = np.clip(2.0 * center - P[i], L, U)
                    else:
                        # Direction 4: Toward a randomly chosen agent's
                        # personal best (information sharing)
                        r_agent = rng.integers(N)
                        target = P[r_agent].copy()

                    E_i = target - X[i]

                    # Scale exploration step by regret intensity:
                    # higher regret = larger exploration radius
                    regret_scale = 0.3 + 0.7 * M_R[i]  # range [0.3, 1.0]
                    E_i *= regret_scale

                else:
                    # ─── Standard Differential Exploration ───
                    # Professor's original: E_i = x_r1 - x_r2
                    r1, r2 = rng.choice(N, 2, replace=False)
                    E_i = X[r1] - X[r2]

                    # Regret-proportional minimum step:
                    # Low regret → allow E_i to be tiny (fine convergence)
                    # Moderate regret → ensure minimum exploration magnitude
                    min_step = M_R[i] * 0.02 * search_range  # proportional to regret
                    E_norm = np.linalg.norm(E_i)
                    if E_norm < min_step and E_norm > eps:
                        E_i = E_i * (min_step / E_norm)

                # Exploitation direction (Section 3.2)
                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                # Adaptive exploration probability (Section 7.1)
                q_i = q0_t + self.eta_R * M_R[i] + self.eta_C * C_mem[i] - self.eta_P * progress_mem[i]
                q_i = float(np.clip(q_i, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                # Actual candidate (Section 4.1)
                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i

                # Counterfactual candidate (Section 4.2)
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i

                # Small time-decaying noise perturbation on actual candidate
                noise = 0.02 * sigma_t * search_range / np.sqrt(D) * rng.standard_normal(D)
                Y_A = np.clip(Y_A + noise, L, U)
                Y_C = np.clip(Y_C, L, U)

                F_A = self.func(Y_A)
                F_C = self.func(Y_C)

                # Cognitive regret (Section 6.1)
                regret = max(0.0, F_A - F_C) / (abs(F_A) + abs(F_C) + eps)
                regret = float(np.clip(regret, 0, 1))

                # Counterfactual success with ε-tolerance (Section 6.3)
                success = float(F_C < F_A - eps)

                # Update memories (Section 6.2 & 6.3)
                M_R[i] = self.rho * M_R[i] + (1 - self.rho) * regret
                C_mem[i] = self.rho_c * C_mem[i] + (1 - self.rho_c) * success

                # Counterfactual direction (Section 3.3)
                D_C = Y_C - X[i]

                # ─── Regret-Aware Momentum ───
                # V_i = direction agent actually moved last iteration
                # Momentum weight modulated by regret:
                #   Low regret → high momentum (keep going, direction is good)
                #   High regret → low momentum (abandon direction, try new)
                V_i = X[i] - X_prev[i]
                w_momentum = (1 - M_R[i]) * 0.4 * (1 - tau)  # decays with time + regret

                # ====================================================
                # NCRO MOTION EQUATION (Section 5)
                # Four forces: momentum + exploration + exploitation + counterfactual
                # ====================================================
                candidate = (
                    X[i]
                    + w_momentum * V_i                  # momentum (follow successful direction)
                    + alpha_t * (1 + M_R[i]) * E_i      # exploration (regret-amplified)
                    + beta_t * (1 - M_R[i]) * H_i       # exploitation (regret-dampened)
                    + gamma_t * C_mem[i] * D_C           # counterfactual direction
                )

                # Blend with exploration/exploitation
                if rng.random() < q_i:
                    candidate = 0.75 * candidate + 0.25 * Y_A
                    it_expl += 1
                else:
                    candidate = 0.75 * candidate + 0.25 * Y_C
                    it_xplt += 1

                candidate = np.clip(candidate, L, U)
                F_cand = self.func(candidate)

                # ─── Diversity-Aware Selection ───
                # When population diversity is critically low, relax greedy
                # selection to allow agents to escape local optima
                if pop_diversity < 0.01 and M_R[i] > 0.2:
                    # High regret + low diversity: force movement
                    # Select best among new candidates only (exclude X[i])
                    move_candidates = [Y_A, Y_C, candidate]
                    move_values = [F_A, F_C, F_cand]
                    best_move = int(np.argmin(move_values))
                    newX[i] = move_candidates[best_move]
                    newF[i] = move_values[best_move]
                else:
                    # Standard greedy 4-way selection
                    candidates = [X[i], Y_A, Y_C, candidate]
                    values = [F[i], F_A, F_C, F_cand]
                    best_idx = int(np.argmin(values))
                    newX[i] = candidates[best_idx]
                    newF[i] = values[best_idx]

                if success > 0:
                    it_cf += 1
                it_regret += M_R[i]

            # Update population
            F_old = F.copy()
            X_prev = X.copy()  # Save for momentum before updating
            X = newX
            F = newF

            # Update personal bests (Section 8.2)
            improved = F < PF
            P[improved] = X[improved]
            PF[improved] = F[improved]

            # Update global best (Section 8.3)
            k = int(np.argmin(PF))
            if PF[k] < GF:
                G = P[k].copy()
                GF = PF[k]

            # ─── Diversity Recovery ───
            # If diversity drops critically, reinitialize worst agents
            # to explore distant regions (keeps top 80%, resets worst 20%)
            if pop_diversity < 0.002 and tau < 0.8:
                n_reset = max(1, N // 5)
                worst_idx = np.argsort(F)[-n_reset:]
                for wi in worst_idx:
                    # Reset to random position in search space
                    X[wi] = rng.uniform(L, U, D)
                    F[wi] = self.func(X[wi])

            # Update progress memory for next iteration (Section 7.2)
            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(
                accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0
            )

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
