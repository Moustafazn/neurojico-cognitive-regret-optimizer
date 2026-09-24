"""
Neurojico Cognitive Regret Optimizer (NCRO).

A metaheuristic optimizer based on cognitive regret theory.
Each agent generates actual and counterfactual decisions,
computes cognitive regret, and uses accumulated regret memory
to adaptively balance exploration and exploitation.

Motion equation:
  x_i' = x_i + w·V_i + α(1+M_R)·E_i + β(1-M_R)·H_i + γ·C·D_C
"""

from .optimizer import NCROOptimizer, NCROResult
from .benchmarks import BENCHMARK_FUNCTIONS, sphere, rastrigin, ackley
from .experiment import run_experiment, plot_results, plot_comparison, plot_ablation
from .ablation import (
    NCRO_NoRegret, NCRO_NoCFMem, NCRO_NoCounterfactual,
    NCRO_NoAdaptiveEE, NCRO_NoRegretMemory,
    NCRO_NoMomentum,
)

__all__ = [
    "NCROOptimizer",
    "NCROResult",
    "BENCHMARK_FUNCTIONS",
    "sphere",
    "rastrigin",
    "ackley",
    "run_experiment",
    "plot_results",
    "plot_comparison",
    "plot_ablation",
    "NCRO_NoRegret",
    "NCRO_NoCFMem",
    "NCRO_NoCounterfactual",
    "NCRO_NoAdaptiveEE",
    "NCRO_NoRegretMemory",
    "NCRO_NoMomentum",
]
