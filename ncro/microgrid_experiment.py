"""
NCRO Smart Microgrid Optimization Experiment
==============================================

Algorithms
----------
1. NCRO
2. CMA-ES
3. L-SHADE implementation
4. jSO-style adaptive DE
5. IMODE-style multi-operator DE
6. SLSQP deterministic reference

Statistical analysis
--------------------
- Wilcoxon signed-rank test
- Holm correction
- Vargha-Delaney A12 effect size
- Friedman test
- Mean ranking
- Success rate
- Mean function evaluations to tolerance
- Runtime and feasibility rate

Generated files
---------------
- run_level_results.csv
- summary_results.csv
- allocation_results.csv
- reference_solutions.csv
- objective_verification.csv
- wilcoxon_holm_a12.csv
- friedman_results.csv
- convergence_vs_fes.png
- final_objective_boxplot.png
- best_allocation.png
- mean_rank_plot.png

Required packages
-----------------
pip install numpy scipy pandas matplotlib

Quick validation
----------------
python -m ncro.microgrid_experiment --quick

Full experiment
---------------
python -m ncro.microgrid_experiment
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from ncro.optimizer import NCROOptimizer


# ============================================================
# 1. General experiment settings
# ============================================================

OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "results"
TABLES_DIR = OUTPUT_DIRECTORY / "tables"
FIGURES_DIR = OUTPUT_DIRECTORY / "figures"

ALGORITHMS = [
    "NCRO",
    "CMA-ES",
    "L-SHADE",
    "jSO",
    "IMODE",
    "NL-SHADE-LBC",
    "L-SRTDE",
]

SOURCE_NAMES = [
    "Solar",
    "Wind",
    "Battery",
    "Grid",
]


# ============================================================
# 2. Smart microgrid problem
# ============================================================

@dataclass
class MicrogridProblem:
    """
    Single-period smart microgrid optimization problem.

    Decision vector:
        X = [P_solar, P_wind, P_battery, P_grid]
    """

    name: str = "Base"
    load: float = 100.0

    upper_bounds: Tuple[float, float, float, float] = (
        60.0,
        50.0,
        35.0,
        100.0,
    )

    energy_cost_coefficients: Tuple[
        float, float, float, float
    ] = (
        0.02,
        0.03,
        0.08,
        0.15,
    )

    emission_coefficients: Tuple[
        float, float, float, float
    ] = (
        0.01,
        0.02,
        0.05,
        0.40,
    )

    objective_weights: Tuple[float, float, float] = (
        0.50,
        0.30,
        0.20,
    )

    instability_scale: float = 0.75

    def __post_init__(self) -> None:

        self.upper_bounds = np.asarray(
            self.upper_bounds,
            dtype=float,
        )

        self.lower_bounds = np.zeros(
            4,
            dtype=float,
        )

        self.energy_cost_coefficients = np.asarray(
            self.energy_cost_coefficients,
            dtype=float,
        )

        self.emission_coefficients = np.asarray(
            self.emission_coefficients,
            dtype=float,
        )

        self.objective_weights = np.asarray(
            self.objective_weights,
            dtype=float,
        )

        if self.upper_bounds.shape != (4,):
            raise ValueError(
                "upper_bounds must contain four values."
            )

        if self.energy_cost_coefficients.shape != (4,):
            raise ValueError(
                "energy_cost_coefficients must contain four values."
            )

        if self.emission_coefficients.shape != (4,):
            raise ValueError(
                "emission_coefficients must contain four values."
            )

        if self.objective_weights.shape != (3,):
            raise ValueError(
                "objective_weights must contain three values."
            )

        if not np.isclose(
            np.sum(self.objective_weights),
            1.0,
        ):
            raise ValueError(
                "Objective weights must sum to one."
            )

        if np.any(self.objective_weights < 0.0):
            raise ValueError(
                "Objective weights must be nonnegative."
            )

        if self.load <= 0.0:
            raise ValueError(
                "Load demand must be positive."
            )

        if self.load > np.sum(self.upper_bounds):
            raise ValueError(
                "Available capacity cannot satisfy the load."
            )

        self.energy_scale = float(
            np.dot(
                self.energy_cost_coefficients,
                self.upper_bounds,
            )
        )

        self.emission_scale = float(
            np.dot(
                self.emission_coefficients,
                self.upper_bounds,
            )
        )

        self.scales = np.array(
            [
                self.energy_scale,
                self.emission_scale,
                self.instability_scale,
            ],
            dtype=float,
        )

        if np.any(self.scales <= 0.0):
            raise ValueError(
                "All normalization scales must be positive."
            )

    def repair(
        self,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Project candidates onto:

            sum(X) = load
            lower_bounds <= X <= upper_bounds

        Projection:
            y_j = clip(x_j - lambda, lb_j, ub_j)
        """

        is_vector = np.asarray(candidates).ndim == 1

        candidates = np.atleast_2d(
            np.asarray(candidates, dtype=float)
        )

        lambda_low = np.min(
            candidates - self.upper_bounds,
            axis=1,
        )

        lambda_high = np.max(
            candidates - self.lower_bounds,
            axis=1,
        )

        for _ in range(70):

            lambda_middle = 0.5 * (
                lambda_low + lambda_high
            )

            repaired = np.clip(
                candidates - lambda_middle[:, None],
                self.lower_bounds,
                self.upper_bounds,
            )

            excessive_power = (
                np.sum(repaired, axis=1)
                > self.load
            )

            lambda_low = np.where(
                excessive_power,
                lambda_middle,
                lambda_low,
            )

            lambda_high = np.where(
                excessive_power,
                lambda_high,
                lambda_middle,
            )

        repaired = np.clip(
            candidates
            - 0.5
            * (lambda_low + lambda_high)[:, None],
            self.lower_bounds,
            self.upper_bounds,
        )

        if is_vector:
            return repaired[0]

        return repaired

    def objective_components(
        self,
        candidates: np.ndarray,
    ) -> np.ndarray:

        candidates = np.atleast_2d(
            np.asarray(candidates, dtype=float)
        )

        energy_cost = (
            candidates
            @ self.energy_cost_coefficients
        )

        emission_cost = (
            candidates
            @ self.emission_coefficients
        )

        source_ratios = (
            candidates / self.load
        )

        allocation_imbalance = np.sum(
            (source_ratios - 0.25) ** 2,
            axis=1,
        )

        return np.column_stack(
            [
                energy_cost,
                emission_cost,
                allocation_imbalance,
            ]
        )

    def evaluate(
        self,
        candidates: np.ndarray,
    ) -> np.ndarray:

        components = self.objective_components(
            candidates
        )

        normalized_components = (
            components / self.scales
        )

        return (
            normalized_components
            @ self.objective_weights
        )

    def is_feasible(
        self,
        candidate: np.ndarray,
        tolerance: float = 1e-7,
    ) -> bool:

        candidate = np.asarray(
            candidate,
            dtype=float,
        )

        bounds_satisfied = (
            np.all(
                candidate
                >= self.lower_bounds - tolerance
            )
            and
            np.all(
                candidate
                <= self.upper_bounds + tolerance
            )
        )

        balance_satisfied = (
            abs(
                np.sum(candidate) - self.load
            )
            <= tolerance
        )

        return bool(
            bounds_satisfied
            and balance_satisfied
        )


# ============================================================
# 3. Operating scenarios
# ============================================================

def create_scenarios() -> List[MicrogridProblem]:
    """
    Same four-variable model evaluated under different inputs.
    """

    return [
        MicrogridProblem(
            name="Base",
        ),

        MicrogridProblem(
            name="HighLoad",
            load=120.0,
        ),

        MicrogridProblem(
            name="LowSolar",
            upper_bounds=(
                35.0,
                50.0,
                35.0,
                100.0,
            ),
        ),

        MicrogridProblem(
            name="LowWind",
            upper_bounds=(
                60.0,
                25.0,
                35.0,
                100.0,
            ),
        ),

        MicrogridProblem(
            name="SmallBattery",
            upper_bounds=(
                60.0,
                50.0,
                15.0,
                100.0,
            ),
        ),

        MicrogridProblem(
            name="PeakTariff",
            energy_cost_coefficients=(
                0.02,
                0.03,
                0.09,
                0.25,
            ),
        ),

        MicrogridProblem(
            name="CleanerGrid",
            emission_coefficients=(
                0.01,
                0.02,
                0.05,
                0.18,
            ),
        ),

        MicrogridProblem(
            name="EmissionPriority",
            objective_weights=(
                0.30,
                0.55,
                0.15,
            ),
        ),
    ]


# ============================================================
# 4. Function-evaluation recorder
# ============================================================

class EvaluationRecorder:

    def __init__(
        self,
        problem: MicrogridProblem,
        evaluation_budget: int,
        reference_value: float,
        success_tolerance: float,
    ) -> None:

        self.problem = problem
        self.evaluation_budget = evaluation_budget
        self.reference_value = reference_value
        self.success_tolerance = success_tolerance

        self.function_evaluations = 0
        self.best_value = np.inf

        self.best_solution: Optional[
            np.ndarray
        ] = None

        self.first_success_evaluation: Optional[
            int
        ] = None

        self.fe_history: List[int] = []
        self.best_history: List[float] = []

    def evaluate(
        self,
        candidates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        repaired = self.problem.repair(
            candidates
        )

        values = self.problem.evaluate(
            repaired
        )

        for candidate, value in zip(
            np.atleast_2d(repaired),
            np.atleast_1d(values),
        ):

            self.function_evaluations += 1

            if value < self.best_value:

                self.best_value = float(value)
                self.best_solution = candidate.copy()

            if (
                self.first_success_evaluation
                is None
                and value
                <= self.reference_value
                + self.success_tolerance
            ):
                self.first_success_evaluation = (
                    self.function_evaluations
                )

            self.fe_history.append(
                self.function_evaluations
            )

            self.best_history.append(
                self.best_value
            )

        return repaired, np.asarray(values)


def initialize_population(
    recorder: EvaluationRecorder,
    rng: np.random.Generator,
    population_size: int,
) -> Tuple[np.ndarray, np.ndarray]:

    raw_population = rng.uniform(
        recorder.problem.lower_bounds,
        recorder.problem.upper_bounds,
        size=(population_size, 4),
    )

    return recorder.evaluate(
        raw_population
    )


# ============================================================
# 5. NCRO  (delegates to the shared ncro/optimizer.py)
# ============================================================

def _run_ncro_with_class(
    optimizer_class,
    recorder: EvaluationRecorder,
    rng: np.random.Generator,
    population_size: int = 30,
) -> None:
    """Run any NCROOptimizer variant through the EvaluationRecorder.

    The optimizer uses ``recorder.evaluate`` as its objective function
    and ``problem.repair`` as its constraint handler, so every function
    evaluation is tracked identically to all other algorithms.
    """
    problem = recorder.problem
    dimension = problem.lower_bounds.shape[0]

    # Wrapper: evaluate a single candidate through the recorder.
    def objective(x):
        _, values = recorder.evaluate(x)
        return float(np.atleast_1d(values)[0])

    seed = int(rng.integers(0, 2**31))

    # Pass max_fes directly — the optimizer manages its own FE
    # budget internally via a while-loop counter (CEC convention).
    opt = optimizer_class(
        objective_function=objective,
        dimension=dimension,
        bounds=(problem.lower_bounds, problem.upper_bounds),
        population_size=population_size,
        max_fes=recorder.evaluation_budget,
        seed=seed,
        repair_fn=problem.repair,
    )
    opt.optimize()


def run_ncro(
    recorder: EvaluationRecorder,
    rng: np.random.Generator,
    population_size: int = 30,
) -> None:
    """Run NCRO optimizer through the EvaluationRecorder."""
    _run_ncro_with_class(
        NCROOptimizer, recorder, rng, population_size,
    )





# ============================================================
# 6. CMA-ES
# ============================================================

def run_cma_es(
    recorder: EvaluationRecorder,
    rng: np.random.Generator,
    population_size: int = 30,
) -> None:

    dimension = 4
    lambda_size = population_size
    mu = lambda_size // 2

    weights = (
        np.log(mu + 0.5)
        - np.log(
            np.arange(1, mu + 1)
        )
    )

    weights /= np.sum(weights)

    effective_mu = (
        1.0
        / np.sum(weights ** 2)
    )

    c_c = (
        4.0
        + effective_mu / dimension
    ) / (
        dimension
        + 4.0
        + 2.0
        * effective_mu
        / dimension
    )

    c_sigma = (
        effective_mu + 2.0
    ) / (
        dimension
        + effective_mu
        + 5.0
    )

    c_1 = (
        2.0
        /
        (
            (dimension + 1.3) ** 2
            + effective_mu
        )
    )

    c_mu = min(
        1.0 - c_1,
        2.0
        * (
            effective_mu
            - 2.0
            + 1.0 / effective_mu
        )
        /
        (
            (dimension + 2.0) ** 2
            + effective_mu
        ),
    )

    damping = (
        1.0
        + 2.0
        * max(
            0.0,
            np.sqrt(
                (
                    effective_mu - 1.0
                )
                /
                (
                    dimension + 1.0
                )
            )
            - 1.0,
        )
        + c_sigma
    )

    expected_norm = (
        np.sqrt(dimension)
        * (
            1.0
            - 1.0
            / (4.0 * dimension)
            + 1.0
            / (
                21.0
                * dimension ** 2
            )
        )
    )

    mean = recorder.problem.repair(
        rng.uniform(
            recorder.problem.lower_bounds,
            recorder.problem.upper_bounds,
        )
    )

    sigma = (
        0.25
        * np.mean(
            recorder.problem.upper_bounds
        )
    )

    covariance = np.eye(
        dimension
    )

    covariance_path = np.zeros(
        dimension
    )

    sigma_path = np.zeros(
        dimension
    )

    generation = 0

    while (
        recorder.function_evaluations
        + lambda_size
        <= recorder.evaluation_budget
    ):

        covariance = 0.5 * (
            covariance + covariance.T
        )

        eigenvalues, eigenvectors = (
            np.linalg.eigh(covariance)
        )

        eigenvalues = np.maximum(
            eigenvalues,
            1e-14,
        )

        scaling = np.sqrt(
            eigenvalues
        )

        normal_samples = rng.normal(
            size=(
                lambda_size,
                dimension,
            )
        )

        mutation_steps = (
            normal_samples * scaling
        ) @ eigenvectors.T

        candidates, fitness = (
            recorder.evaluate(
                mean
                + sigma
                * mutation_steps
            )
        )

        selected = np.argsort(
            fitness
        )[:mu]

        old_mean = mean.copy()

        mean = np.sum(
            weights[:, None]
            * candidates[selected],
            axis=0,
        )

        normalized_step = (
            mean - old_mean
        ) / max(sigma, 1e-15)

        inverse_sqrt_covariance = (
            eigenvectors
            * (1.0 / scaling)
        ) @ eigenvectors.T

        sigma_path = (
            (1.0 - c_sigma)
            * sigma_path
            +
            np.sqrt(
                c_sigma
                * (2.0 - c_sigma)
                * effective_mu
            )
            * (
                inverse_sqrt_covariance
                @ normalized_step
            )
        )

        generation += 1

        path_normalizer = np.sqrt(
            1.0
            - (
                1.0 - c_sigma
            ) ** (
                2.0 * generation
            )
        )

        path_condition = float(
            np.linalg.norm(sigma_path)
            / max(
                path_normalizer,
                1e-15,
            )
            / expected_norm
            <
            (
                1.4
                + 2.0
                / (
                    dimension + 1.0
                )
            )
        )

        covariance_path = (
            (1.0 - c_c)
            * covariance_path
            +
            path_condition
            * np.sqrt(
                c_c
                * (2.0 - c_c)
                * effective_mu
            )
            * normalized_step
        )

        selected_steps = (
            candidates[selected]
            - old_mean
        ) / max(sigma, 1e-15)

        rank_mu_update = np.zeros(
            (
                dimension,
                dimension,
            )
        )

        for index in range(mu):

            rank_mu_update += (
                weights[index]
                * np.outer(
                    selected_steps[index],
                    selected_steps[index],
                )
            )

        covariance = (
            (
                1.0
                - c_1
                - c_mu
            )
            * covariance
            +
            c_1
            * (
                np.outer(
                    covariance_path,
                    covariance_path,
                )
                +
                (
                    1.0
                    - path_condition
                )
                * c_c
                * (2.0 - c_c)
                * covariance
            )
            +
            c_mu
            * rank_mu_update
        )

        sigma *= np.exp(
            (
                c_sigma / damping
            )
            * (
                np.linalg.norm(sigma_path)
                / expected_norm
                - 1.0
            )
        )


# ============================================================
# 7. Adaptive DE competitors
# ============================================================

def sample_scale_factor(
    rng: np.random.Generator,
    memory_value: float,
) -> float:

    for _ in range(100):

        scale_factor = (
            memory_value
            + 0.10
            * np.tan(
                np.pi
                * (
                    rng.random() - 0.5
                )
            )
        )

        if scale_factor > 0.0:
            return float(
                min(scale_factor, 1.0)
            )

    return 0.50


def run_adaptive_de(
    recorder: EvaluationRecorder,
    rng: np.random.Generator,
    variant: str,
    initial_population_size: int = 30,
) -> None:

    population_size = (
        initial_population_size
    )

    population, fitness = initialize_population(
        recorder,
        rng,
        population_size,
    )

    archive: List[np.ndarray] = []

    memory_size = 6

    scale_memory = np.full(
        memory_size,
        0.50,
    )

    crossover_memory = np.full(
        memory_size,
        0.50,
    )

    memory_index = 0
    minimum_population_size = 4

    # L-SRTDE state
    recent_success_rates: List[float] = []
    stagnation_counter = 0
    generation_counter = 0

    while (
        recorder.function_evaluations
        + population_size
        <= recorder.evaluation_budget
        and population_size >= 4
    ):

        budget_progress = (
            recorder.function_evaluations
            / recorder.evaluation_budget
        )

        sorted_indices = np.argsort(
            fitness
        )

        trial_population = np.empty_like(
            population
        )

        sampled_scale_factors = np.empty(
            population_size
        )

        sampled_crossover_rates = np.empty(
            population_size
        )

        for agent in range(
            population_size
        ):

            memory_slot = int(
                rng.integers(
                    memory_size
                )
            )

            scale_factor = sample_scale_factor(
                rng,
                scale_memory[memory_slot],
            )

            crossover_rate = float(
                np.clip(
                    rng.normal(
                        crossover_memory[
                            memory_slot
                        ],
                        0.10,
                    ),
                    0.0,
                    1.0,
                )
            )

            if variant == "jSO":

                if budget_progress < 0.60:
                    scale_factor *= 0.70

                scale_factor = float(
                    np.clip(
                        scale_factor,
                        0.05,
                        1.0,
                    )
                )

                if budget_progress < 0.25:
                    crossover_rate = max(
                        crossover_rate,
                        0.70,
                    )

                elif budget_progress < 0.50:
                    crossover_rate = max(
                        crossover_rate,
                        0.60,
                    )

            if variant == "NL-SHADE-LBC":

                cr_bias = (
                    0.9 * (1.0 - budget_progress)
                    + 0.1 * budget_progress
                )

                crossover_rate = float(
                    np.clip(
                        crossover_rate * cr_bias
                        + 0.5 * (1.0 - cr_bias),
                        0.0,
                        1.0,
                    )
                )

            if variant == "L-SHADE":
                p_rate = 0.11
            else:
                p_rate = 0.20

            pbest_count = max(
                2,
                int(
                    np.ceil(
                        p_rate
                        * population_size
                    )
                ),
            )

            pbest_index = int(
                rng.choice(
                    sorted_indices[
                        :pbest_count
                    ]
                )
            )

            pbest = population[
                pbest_index
            ]

            available = list(
                range(population_size)
            )

            available.remove(agent)

            random_index_1 = int(
                rng.choice(available)
            )

            if archive:

                combined_pool = np.vstack(
                    [
                        population,
                        np.asarray(archive),
                    ]
                )

            else:
                combined_pool = population

            random_index_2 = int(
                rng.integers(
                    len(combined_pool)
                )
            )

            if (
                variant == "IMODE"
                and rng.random() >= 0.50
            ):

                remaining = [
                    index
                    for index in available
                    if index != random_index_1
                ]

                random_index_3 = int(
                    rng.choice(remaining)
                )

                mutant = (
                    pbest
                    + scale_factor
                    * (
                        population[random_index_1]
                        - population[random_index_3]
                    )
                )

            else:

                mutant = (
                    population[agent]
                    + scale_factor
                    * (
                        pbest
                        - population[agent]
                    )
                    + scale_factor
                    * (
                        population[random_index_1]
                        - combined_pool[random_index_2]
                    )
                )

            crossover_mask = (
                rng.random(4)
                < crossover_rate
            )

            crossover_mask[
                rng.integers(4)
            ] = True

            trial_population[agent] = np.where(
                crossover_mask,
                mutant,
                population[agent],
            )

            sampled_scale_factors[agent] = (
                scale_factor
            )

            sampled_crossover_rates[agent] = (
                crossover_rate
            )

        (
            trial_population,
            trial_fitness,
        ) = recorder.evaluate(
            trial_population
        )

        success_mask = (
            trial_fitness < fitness
        )

        if np.any(success_mask):

            successful_indices = np.where(
                success_mask
            )[0]

            improvements = (
                fitness[successful_indices]
                - trial_fitness[
                    successful_indices
                ]
            )

            for index in successful_indices:
                archive.append(
                    population[index].copy()
                )

            population[
                successful_indices
            ] = trial_population[
                successful_indices
            ]

            fitness[
                successful_indices
            ] = trial_fitness[
                successful_indices
            ]

            success_weights = (
                improvements
                / np.sum(improvements)
            )

            successful_f = (
                sampled_scale_factors[
                    successful_indices
                ]
            )

            successful_cr = (
                sampled_crossover_rates[
                    successful_indices
                ]
            )

            denominator_f = np.sum(
                success_weights
                * successful_f
            )

            if denominator_f > 1e-12:

                scale_memory[
                    memory_index
                ] = (
                    np.sum(
                        success_weights
                        * successful_f ** 2
                    )
                    / denominator_f
                )

            denominator_cr = np.sum(
                success_weights
                * successful_cr
            )

            if denominator_cr > 1e-12:

                crossover_memory[
                    memory_index
                ] = (
                    np.sum(
                        success_weights
                        * successful_cr ** 2
                    )
                    / denominator_cr
                )

            memory_index = (
                memory_index + 1
            ) % memory_size

        # L-SRTDE: success rate tracking and memory reset
        if variant == "L-SRTDE":

            success_rate = (
                float(np.sum(success_mask))
                / population_size
                if population_size > 0
                else 0.0
            )

            recent_success_rates.append(
                success_rate
            )

            if len(recent_success_rates) > 10:
                recent_success_rates.pop(0)

            avg_sr = (
                float(np.mean(recent_success_rates))
                if recent_success_rates
                else 0.5
            )

            if (
                avg_sr < 0.10
                and generation_counter > 10
            ):
                stagnation_counter += 1

                if stagnation_counter >= 5:
                    scale_memory = np.full(
                        memory_size,
                        0.3 + 0.4 * rng.random(),
                    )
                    crossover_memory = np.full(
                        memory_size,
                        0.3 + 0.4 * rng.random(),
                    )
                    memory_index = 0
                    stagnation_counter = 0
            else:
                stagnation_counter = max(
                    0,
                    stagnation_counter - 1,
                )

        generation_counter += 1

        if len(archive) > population_size:

            archive_indices = rng.choice(
                len(archive),
                size=population_size,
                replace=False,
            )

            archive_array = np.asarray(
                archive
            )

            archive = [
                archive_array[index].copy()
                for index in archive_indices
            ]

        # Population size reduction
        if variant == "NL-SHADE-LBC":
            # Quadratic (non-linear) reduction
            ratio = budget_progress ** 2
            target_population_size = max(
                minimum_population_size,
                int(round(
                    initial_population_size
                    - (
                        initial_population_size
                        - minimum_population_size
                    ) * ratio
                )),
            )
        else:
            # Linear reduction (L-SHADE, jSO, IMODE, L-SRTDE)
            target_population_size = int(
                round(
                    initial_population_size
                    -
                    (
                        initial_population_size
                        - minimum_population_size
                    )
                    * recorder.function_evaluations
                    / recorder.evaluation_budget
                )
            )

        target_population_size = max(
            minimum_population_size,
            target_population_size,
        )

        if (
            target_population_size
            < population_size
        ):

            retained_indices = np.argsort(
                fitness
            )[:target_population_size]

            population = population[
                retained_indices
            ]

            fitness = fitness[
                retained_indices
            ]

            population_size = (
                target_population_size
            )

            if len(archive) > population_size:
                archive = archive[
                    :population_size
                ]


# ============================================================
# 8. SLSQP reference solution
# ============================================================

def obtain_reference_solution(
    problem: MicrogridProblem,
) -> Tuple[np.ndarray, float]:

    def scalar_objective(
        candidate: np.ndarray,
    ) -> float:

        return float(
            problem.evaluate(candidate)[0]
        )

    equality_constraint = {
        "type": "eq",
        "fun": (
            lambda candidate:
            np.sum(candidate)
            - problem.load
        ),
    }

    initial_solution = problem.repair(
        0.50 * problem.upper_bounds
    )

    result = minimize(
        scalar_objective,
        initial_solution,
        method="SLSQP",
        bounds=list(
            zip(
                problem.lower_bounds,
                problem.upper_bounds,
            )
        ),
        constraints=equality_constraint,
        options={
            "ftol": 1e-14,
            "maxiter": 3000,
            "disp": False,
        },
    )

    if not result.success:

        warnings.warn(
            "SLSQP warning for scenario "
            f"{problem.name}: {result.message}"
        )

    reference_solution = problem.repair(
        result.x
    )

    reference_value = float(
        problem.evaluate(
            reference_solution
        )[0]
    )

    return (
        reference_solution,
        reference_value,
    )


# ============================================================
# 9. Statistical functions
# ============================================================

def holm_adjustment(
    p_values: np.ndarray,
) -> np.ndarray:

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    number_of_tests = len(
        p_values
    )

    ordered_indices = np.argsort(
        p_values
    )

    adjusted_values = np.empty(
        number_of_tests
    )

    running_maximum = 0.0

    for position, original_index in enumerate(
        ordered_indices
    ):

        multiplier = (
            number_of_tests - position
        )

        current_value = (
            multiplier
            * p_values[original_index]
        )

        running_maximum = max(
            running_maximum,
            current_value,
        )

        adjusted_values[
            original_index
        ] = min(
            1.0,
            running_maximum,
        )

    return adjusted_values


def vargha_delaney_a12(
    ncro_values: np.ndarray,
    competitor_values: np.ndarray,
) -> float:
    """
    Minimization interpretation:

        A12 > 0.50: NCRO tends to be better.
        A12 = 0.50: comparable performance.
        A12 < 0.50: competitor tends to be better.
    """

    ncro_values = np.asarray(
        ncro_values,
        dtype=float,
    )

    competitor_values = np.asarray(
        competitor_values,
        dtype=float,
    )

    better = 0
    ties = 0

    for ncro_value in ncro_values:

        for competitor_value in competitor_values:

            if ncro_value < competitor_value:
                better += 1

            elif np.isclose(
                ncro_value,
                competitor_value,
                rtol=0.0,
                atol=1e-14,
            ):
                ties += 1

    total_pairs = (
        len(ncro_values)
        * len(competitor_values)
    )

    return float(
        (
            better
            + 0.5 * ties
        )
        / total_pairs
    )


def effect_size_interpretation(
    a12: float,
) -> str:

    difference = abs(
        a12 - 0.50
    )

    if difference < 0.06:
        return "Negligible"

    if difference < 0.14:
        return "Small"

    if difference < 0.21:
        return "Medium"

    return "Large"


# ============================================================
# 10. Plotting functions
# ============================================================

def plot_convergence(
    traces: Dict[
        str,
        Tuple[np.ndarray, np.ndarray],
    ],
    reference_value: float,
) -> None:

    plt.figure(
        figsize=(8.2, 5.2)
    )

    for algorithm in ALGORITHMS:

        if algorithm not in traces:
            continue

        function_evaluations, best_values = (
            traces[algorithm]
        )

        plt.step(
            function_evaluations,
            best_values,
            where="post",
            linewidth=1.7,
            label=algorithm,
        )

    plt.axhline(
        reference_value,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="SLSQP reference",
    )

    plt.xlabel(
        "Function evaluations"
    )

    plt.ylabel(
        "Best objective value"
    )

    plt.title(
        "Convergence on the Base Microgrid Scenario"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "convergence_vs_fes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_final_boxplot(
    run_results: pd.DataFrame,
) -> None:

    base_results = run_results[
        run_results["Scenario"]
        == "Base"
    ]

    values = [
        base_results[
            base_results["Algorithm"]
            == algorithm
        ]["FinalObjective"].to_numpy()
        for algorithm in ALGORITHMS
    ]

    plt.figure(
        figsize=(8.2, 5.2)
    )

    plt.boxplot(
        values,
        tick_labels=ALGORITHMS,
        showmeans=True,
    )

    plt.xlabel("Algorithm")

    plt.ylabel(
        "Final objective value"
    )

    plt.title(
        "Final Objective Distribution on the Base Scenario"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_objective_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_best_allocation(
    allocation_results: pd.DataFrame,
) -> None:

    base_results = allocation_results[
        allocation_results["Scenario"]
        == "Base"
    ]

    best_row = (
        base_results
        .sort_values("Objective")
        .iloc[0]
    )

    allocation = best_row[
        [
            "Solar",
            "Wind",
            "Battery",
            "Grid",
        ]
    ].to_numpy(
        dtype=float
    )

    colors = [
        "#F4B400",
        "#4285F4",
        "#34A853",
        "#6B7280",
    ]

    plt.figure(
        figsize=(8.2, 5.2)
    )

    bars = plt.bar(
        SOURCE_NAMES,
        allocation,
        color=colors,
    )

    plt.xlabel(
        "Energy source"
    )

    plt.ylabel(
        "Power allocation (kW)"
    )

    plt.title(
        "Best Smart Microgrid Power Allocation"
    )

    upper_limit = max(
        1.0,
        1.20 * np.max(allocation),
    )

    plt.ylim(
        0.0,
        upper_limit,
    )

    for bar, value in zip(
        bars,
        allocation,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value
            + 0.02 * upper_limit,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "best_allocation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_mean_ranks(
    rank_results: pd.DataFrame,
) -> None:

    ordered = rank_results.sort_values(
        "MeanRank",
        ascending=True,
    )

    plt.figure(
        figsize=(8.2, 5.2)
    )

    bars = plt.bar(
        ordered["Algorithm"],
        ordered["MeanRank"],
        color="#4472C4",
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Mean Friedman rank")

    plt.title(
        "Mean Algorithm Rankings Across Scenario-Run Blocks"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        ordered["MeanRank"],
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
        )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "mean_rank_plot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 11. Complete experiment
# ============================================================

def run_experiment(
    quick_mode: bool = False,
) -> None:

    if quick_mode:

        number_of_runs = 3
        evaluation_budget = 3030
        problems = create_scenarios()[:2]

        print(
            "Quick validation: 2 scenarios, "
            "3 runs, 3030 function evaluations."
        )

    else:

        number_of_runs = 30
        evaluation_budget = 45030
        problems = create_scenarios()

        print(
            "Full experiment: 8 scenarios, "
            "30 runs, 45030 function evaluations."
        )

    algorithms_to_run = ALGORITHMS

    population_size = 30
    success_tolerance = 1e-6
    base_seed = 20260827

    # Ensure output directories exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    run_records: List[dict] = []
    allocation_records: List[dict] = []
    reference_records: List[dict] = []

    reference_solutions: Dict[
        str,
        Tuple[np.ndarray, float],
    ] = {}

    convergence_traces: Dict[
        str,
        Tuple[np.ndarray, np.ndarray],
    ] = {}

    for problem in problems:

        (
            reference_solution,
            reference_value,
        ) = obtain_reference_solution(
            problem
        )

        reference_solutions[
            problem.name
        ] = (
            reference_solution,
            reference_value,
        )

        reference_records.append(
            {
                "Scenario": problem.name,
                "Solar": reference_solution[0],
                "Wind": reference_solution[1],
                "Battery": reference_solution[2],
                "Grid": reference_solution[3],
                "Total": np.sum(
                    reference_solution
                ),
                "ReferenceObjective": (
                    reference_value
                ),
                "SuccessThreshold": (
                    reference_value
                    + success_tolerance
                ),
                "Feasible": problem.is_feasible(
                    reference_solution
                ),
            }
        )

        print(
            f"\nScenario: {problem.name}"
        )

        print(
            "Reference objective: "
            f"{reference_value:.12f}"
        )

        for algorithm in algorithms_to_run:

            print(
                f"  Running {algorithm}"
            )

            for run_index in range(
                number_of_runs
            ):

                random_seed = (
                    base_seed + run_index
                )

                rng = np.random.default_rng(
                    random_seed
                )

                recorder = EvaluationRecorder(
                    problem=problem,
                    evaluation_budget=(
                        evaluation_budget
                    ),
                    reference_value=(
                        reference_value
                    ),
                    success_tolerance=(
                        success_tolerance
                    ),
                )

                start_time = (
                    time.perf_counter()
                )

                if algorithm == "NCRO":

                    run_ncro(
                        recorder,
                        rng,
                        population_size,
                    )

                elif algorithm == "CMA-ES":

                    run_cma_es(
                        recorder,
                        rng,
                        population_size,
                    )

                else:

                    run_adaptive_de(
                        recorder,
                        rng,
                        variant=algorithm,
                        initial_population_size=(
                            population_size
                        ),
                    )

                runtime = (
                    time.perf_counter()
                    - start_time
                )

                best_solution = (
                    recorder.best_solution
                )

                final_objective = (
                    recorder.best_value
                )

                success = int(
                    final_objective
                    <= reference_value
                    + success_tolerance
                )

                feasible = int(
                    problem.is_feasible(
                        best_solution
                    )
                )

                run_records.append(
                    {
                        "Scenario": problem.name,
                        "Algorithm": algorithm,
                        "Run": run_index + 1,
                        "Seed": random_seed,
                        "FinalObjective": (
                            final_objective
                        ),
                        "ReferenceObjective": (
                            reference_value
                        ),
                        "AbsoluteGap": abs(
                            final_objective
                            - reference_value
                        ),
                        "SignedGap": (
                            final_objective
                            - reference_value
                        ),
                        "Success": success,
                        "FEsToTolerance": (
                            recorder
                            .first_success_evaluation
                            if recorder
                            .first_success_evaluation
                            is not None
                            else np.nan
                        ),
                        "RuntimeSeconds": runtime,
                        "FunctionEvaluations": (
                            recorder
                            .function_evaluations
                        ),
                        "Feasible": feasible,
                    }
                )

                allocation_records.append(
                    {
                        "Scenario": problem.name,
                        "Algorithm": algorithm,
                        "Run": run_index + 1,
                        "Solar": best_solution[0],
                        "Wind": best_solution[1],
                        "Battery": best_solution[2],
                        "Grid": best_solution[3],
                        "Total": np.sum(
                            best_solution
                        ),
                        "Objective": (
                            final_objective
                        ),
                        "Feasible": feasible,
                    }
                )

                if (
                    problem.name == "Base"
                    and run_index == 0
                ):

                    convergence_traces[
                        algorithm
                    ] = (
                        np.asarray(
                            recorder.fe_history
                        ),
                        np.asarray(
                            recorder.best_history
                        ),
                    )

    run_results = pd.DataFrame(
        run_records
    )

    allocation_results = pd.DataFrame(
        allocation_records
    )

    reference_results = pd.DataFrame(
        reference_records
    )

    run_results.to_csv(
        TABLES_DIR
        / "run_level_results.csv",
        index=False,
    )

    allocation_results.to_csv(
        TABLES_DIR
        / "allocation_results.csv",
        index=False,
    )

    reference_results.to_csv(
        TABLES_DIR
        / "reference_solutions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary_results = (
        run_results
        .groupby(
            [
                "Scenario",
                "Algorithm",
            ]
        )
        .agg(
            Best=(
                "FinalObjective",
                "min",
            ),
            Mean=(
                "FinalObjective",
                "mean",
            ),
            Median=(
                "FinalObjective",
                "median",
            ),
            Worst=(
                "FinalObjective",
                "max",
            ),
            Std=(
                "FinalObjective",
                "std",
            ),
            MeanAbsoluteGap=(
                "AbsoluteGap",
                "mean",
            ),
            MeanFEsToTolerance=(
                "FEsToTolerance",
                "mean",
            ),
            MeanRuntimeSeconds=(
                "RuntimeSeconds",
                "mean",
            ),
            SuccessRate=(
                "Success",
                "mean",
            ),
            FeasibilityRate=(
                "Feasible",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_results[
        "SuccessRatePercent"
    ] = (
        100.0
        * summary_results[
            "SuccessRate"
        ]
    )

    summary_results[
        "FeasibilityRatePercent"
    ] = (
        100.0
        * summary_results[
            "FeasibilityRate"
        ]
    )

    summary_results.to_csv(
        TABLES_DIR
        / "summary_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Pairwise Wilcoxon, Holm, and A12
    # --------------------------------------------------------

    ncro_results = (
        run_results[
            run_results["Algorithm"]
            == "NCRO"
        ]
        .sort_values(
            [
                "Scenario",
                "Run",
            ]
        )
    )

    pairwise_rows = []

    for competitor in ALGORITHMS[1:]:

        competitor_results = (
            run_results[
                run_results["Algorithm"]
                == competitor
            ]
            .sort_values(
                [
                    "Scenario",
                    "Run",
                ]
            )
        )

        ncro_values = ncro_results[
            "FinalObjective"
        ].to_numpy()

        competitor_values = (
            competitor_results[
                "FinalObjective"
            ].to_numpy()
        )

        differences = (
            ncro_values
            - competitor_values
        )

        if np.allclose(
            differences,
            0.0,
            atol=1e-14,
            rtol=0.0,
        ):

            statistic = 0.0
            raw_p_value = 1.0

        else:

            test_result = wilcoxon(
                ncro_values,
                competitor_values,
                zero_method="zsplit",
                alternative="two-sided",
                method="auto",
            )

            statistic = float(
                test_result.statistic
            )

            raw_p_value = float(
                test_result.pvalue
            )

        a12 = vargha_delaney_a12(
            ncro_values,
            competitor_values,
        )

        ncro_wins = int(
            np.sum(
                ncro_values
                < competitor_values
                - 1e-14
            )
        )

        ncro_losses = int(
            np.sum(
                ncro_values
                > competitor_values
                + 1e-14
            )
        )

        ties = int(
            len(ncro_values)
            - ncro_wins
            - ncro_losses
        )

        pairwise_rows.append(
            {
                "Competitor": competitor,
                "WilcoxonStatistic": (
                    statistic
                ),
                "RawP": raw_p_value,
                "A12_NCRO_Better": a12,
                "EffectMagnitude": (
                    effect_size_interpretation(
                        a12
                    )
                ),
                "NCROWins": ncro_wins,
                "NCROLosses": ncro_losses,
                "Ties": ties,
            }
        )

    pairwise_results = pd.DataFrame(
        pairwise_rows
    )

    pairwise_results[
        "HolmAdjustedP"
    ] = holm_adjustment(
        pairwise_results[
            "RawP"
        ].to_numpy()
    )

    pairwise_results[
        "SignificantAt0.05"
    ] = (
        pairwise_results[
            "HolmAdjustedP"
        ]
        < 0.05
    )

    pairwise_results.to_csv(
        TABLES_DIR
        / "wilcoxon_holm_a12.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Friedman test and mean ranks
    # --------------------------------------------------------
    # CEC convention (Liang, Qu & Suganthan 2017):
    #   SE = max(f(x) - f*, 0);  if SE < 1e-8  →  SE = 0.
    # This thresholds numerical noise so that all algorithms
    # reaching the optimum within machine precision are tied.
    # When success rates are equal, convergence speed (FEs to
    # tolerance) becomes the discriminating metric.

    CEC_ERROR_THRESHOLD = 1e-8

    # Build thresholded-SE arrays per algorithm.
    ordered_arrays = []

    for algorithm in ALGORITHMS:

        abs_gap = (
            run_results[
                run_results["Algorithm"]
                == algorithm
            ]
            .sort_values(
                [
                    "Scenario",
                    "Run",
                ]
            )[
                "AbsoluteGap"
            ]
            .to_numpy()
        )

        # CEC thresholding: SE < 1e-8 → 0.
        thresholded = np.where(
            abs_gap < CEC_ERROR_THRESHOLD,
            0.0,
            abs_gap,
        )

        ordered_arrays.append(thresholded)

    (
        friedman_statistic,
        friedman_p_value,
    ) = friedmanchisquare(
        *ordered_arrays
    )

    result_matrix = np.column_stack(
        ordered_arrays
    )

    block_ranks = np.apply_along_axis(
        lambda row:
        rankdata(
            row,
            method="average",
        ),
        axis=1,
        arr=result_matrix,
    )

    mean_ranks = np.mean(
        block_ranks,
        axis=0,
    )

    rank_results = pd.DataFrame(
        {
            "Algorithm": ALGORITHMS,
            "MeanRank": mean_ranks,
        }
    ).sort_values(
        "MeanRank"
    )

    friedman_results = pd.DataFrame(
        {
            "Statistic": [
                "Friedman chi-square",
                "p-value",
                "Number of blocks",
                "Number of algorithms",
                "Significant at 0.05",
            ],
            "Value": [
                # Undefined (0/0) when all blocks are tied; write the
                # literal "NaN" so to_csv() emits no blank cell.
                f"{friedman_statistic:.4f}"
                if np.isfinite(friedman_statistic)
                else "NaN",
                f"{friedman_p_value:.6e}"
                if np.isfinite(friedman_p_value)
                else "NaN",
                result_matrix.shape[0],
                result_matrix.shape[1],
                # "nan < 0.05" is False for every NaN, so guard it.
                bool(friedman_p_value < 0.05)
                if np.isfinite(friedman_p_value)
                else "Undefined (all ranks tied)",
            ],
        }
    )

    friedman_results.to_csv(
        TABLES_DIR
        / "friedman_results.csv",
        index=False,
        na_rep="NaN",
    )

    rank_results.to_csv(
        TABLES_DIR
        / "mean_ranks.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Convergence-speed ranking (FEs to tolerance)
    # --------------------------------------------------------
    # Secondary ranking for problems where all algorithms
    # achieve 100% success rate — discriminates by speed.

    speed_rank = (
        summary_results
        .groupby("Algorithm")["MeanFEsToTolerance"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "MeanFEsToTolerance":
                    "MeanFEsAcrossScenarios",
            }
        )
        .sort_values("MeanFEsAcrossScenarios")
    )
    speed_rank["SpeedRank"] = range(
        1, len(speed_rank) + 1
    )

    speed_rank.to_csv(
        TABLES_DIR
        / "convergence_speed_ranks.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Objective verification
    # --------------------------------------------------------

    base_problem = problems[0]

    (
        reference_solution,
        reference_value,
    ) = reference_solutions["Base"]

    components = (
        base_problem
        .objective_components(
            reference_solution
        )[0]
    )

    normalized_components = (
        components
        / base_problem.scales
    )

    weighted_contributions = (
        base_problem.objective_weights
        * normalized_components
    )

    verification_results = pd.DataFrame(
        {
            "Component": [
                "Energy cost",
                "Emission cost",
                "Allocation imbalance",
                "Unified objective",
            ],
            "RawValue": [
                components[0],
                components[1],
                components[2],
                reference_value,
            ],
            "NormalizationScale": [
                base_problem.scales[0],
                base_problem.scales[1],
                base_problem.scales[2],
                np.nan,
            ],
            "NormalizedValue": [
                normalized_components[0],
                normalized_components[1],
                normalized_components[2],
                np.nan,
            ],
            "Weight": [
                base_problem.objective_weights[0],
                base_problem.objective_weights[1],
                base_problem.objective_weights[2],
                np.nan,
            ],
            "WeightedContribution": [
                weighted_contributions[0],
                weighted_contributions[1],
                weighted_contributions[2],
                np.sum(
                    weighted_contributions
                ),
            ],
        }
    )

    verification_results.to_csv(
        TABLES_DIR
        / "objective_verification.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    plot_convergence(
        convergence_traces,
        reference_value,
    )

    plot_final_boxplot(
        run_results
    )

    plot_best_allocation(
        allocation_results
    )

    plot_mean_ranks(
        rank_results
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("BASE REFERENCE SOLUTION")
    print("=" * 72)

    print(
        "Allocation:",
        np.round(
            reference_solution,
            8,
        ),
    )

    print(
        "Total power:",
        np.sum(reference_solution),
    )

    print(
        "Reference objective:",
        f"{reference_value:.12f}",
    )

    print(
        "Success threshold:",
        f"{reference_value + success_tolerance:.12f}",
    )

    print(
        "Feasible:",
        base_problem.is_feasible(
            reference_solution
        ),
    )

    print("\n" + "=" * 72)
    print("BASE-SCENARIO SUMMARY")
    print("=" * 72)

    print(
        summary_results[
            summary_results["Scenario"]
            == "Base"
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("WILCOXON, HOLM, AND A12")
    print("=" * 72)

    print(
        pairwise_results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("FRIEDMAN TEST")
    print("=" * 72)

    print(
        f"Chi-square = "
        f"{friedman_statistic:.8f}"
    )

    print(
        f"p-value = "
        f"{friedman_p_value:.12g}"
    )

    print(
        "Significant at alpha = 0.05:",
        friedman_p_value < 0.05,
    )

    print("\nMean ranks (thresholded SE, CEC convention):")

    print(
        rank_results.to_string(
            index=False
        )
    )

    print("\nConvergence-speed ranking (FEs to tolerance):")

    print(
        speed_rank.to_string(
            index=False
        )
    )

    print("\nGenerated files:")

    generated_files = [
        ("tables", "run_level_results.csv"),
        ("tables", "summary_results.csv"),
        ("tables", "allocation_results.csv"),
        ("tables", "reference_solutions.csv"),
        ("tables", "objective_verification.csv"),
        ("tables", "wilcoxon_holm_a12.csv"),
        ("tables", "friedman_results.csv"),
        ("tables", "mean_ranks.csv"),
        ("tables", "convergence_speed_ranks.csv"),
        ("figures", "convergence_vs_fes.png"),
        ("figures", "final_objective_boxplot.png"),
        ("figures", "best_allocation.png"),
        ("figures", "mean_rank_plot.png"),
    ]

    for sub_dir, file_name in generated_files:

        print(
            "  "
            + str(
                OUTPUT_DIRECTORY
                / sub_dir
                / file_name
            )
        )


# ============================================================
# 12. Program entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the NCRO smart microgrid "
            "comparative optimization experiment."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a short validation using two scenarios, "
            "three runs, and 3030 function evaluations."
        ),
    )

    arguments = parser.parse_args()

    run_experiment(
        quick_mode=arguments.quick,
    )