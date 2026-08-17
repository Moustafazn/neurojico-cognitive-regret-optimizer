"""
Neurojico Cognitive Regret Optimizer (NCRO)

A novel metaheuristic optimization algorithm based on cognitive regret theory.
Each agent generates both an actual and a counterfactual (alternative) decision,
computes cognitive regret from their comparison, and uses accumulated regret
memory to adaptively balance exploration and exploitation.

Motion equation:
  x_i(t+1) = x_i(t) + α(t)(1+M_R)·E_i + β(t)(1-M_R)·H_i + γ(t)·C·D_C
"""

from .optimizer import NCROOptimizer, NCROResult
from .benchmarks import BENCHMARK_FUNCTIONS, sphere, rastrigin, ackley
from .experiment import run_experiment, plot_results, plot_comparison, plot_ablation
from .ablation import (
    NCRO_NoRegret, NCRO_NoCounterfactual,
    NCRO_NoAdaptiveEE, NCRO_NoRegretMemory,
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
    "NCRO_NoCounterfactual",
    "NCRO_NoAdaptiveEE",
    "NCRO_NoRegretMemory",
]
