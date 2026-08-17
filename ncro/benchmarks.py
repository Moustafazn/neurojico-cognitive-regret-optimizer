"""
Benchmark test functions for optimization (Minimization).

Includes:
  - 3 original functions: Sphere, Rastrigin, Ackley
  - 7 additional standard functions: Rosenbrock, Schwefel, Griewank,
    Levy, Zakharov, Dixon-Price, Michalewicz

All functions (except Schwefel and Michalewicz) have global optimum F* = 0.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Original 3 Functions (from NCRO poster)
# ──────────────────────────────────────────────────────────────────

def sphere(x: np.ndarray) -> float:
    """F(x) = sum(x_j^2).  Domain: [-100, 100].  F* = 0 at x* = 0."""
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """F(x) = 10D + sum(x_j^2 - 10*cos(2*pi*x_j)).  Domain: [-5.12, 5.12].  F* = 0."""
    D = len(x)
    return float(10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function.  Domain: [-32.768, 32.768].  F* = 0."""
    D = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / D))
        - np.exp(sum_cos / D)
        + 20.0 + np.e
    )


# ──────────────────────────────────────────────────────────────────
# Additional Standard Benchmark Functions
# ──────────────────────────────────────────────────────────────────

def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock (valley-shaped, unimodal).
    F(x) = sum( 100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 )
    Domain: [-30, 30].  F* = 0 at x* = (1, 1, ..., 1).
    """
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def schwefel(x: np.ndarray) -> float:
    """
    Schwefel (multimodal, deceptive -- global optimum far from local optima).
    F(x) = 418.9829*D - sum(x_j * sin(sqrt(|x_j|)))
    Domain: [-500, 500].  F* = 0 at x* = (420.9687, ..., 420.9687).
    """
    D = len(x)
    return float(418.9829 * D - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def griewank(x: np.ndarray) -> float:
    """
    Griewank (multimodal with regular structure).
    F(x) = sum(x_j^2)/4000 - prod(cos(x_j/sqrt(j+1))) + 1
    Domain: [-600, 600].  F* = 0 at x* = 0.
    """
    D = len(x)
    indices = np.arange(1, D + 1)
    sum_part = np.sum(x ** 2) / 4000.0
    prod_part = np.prod(np.cos(x / np.sqrt(indices)))
    return float(sum_part - prod_part + 1.0)


def levy(x: np.ndarray) -> float:
    """
    Levy function (multimodal).
    Domain: [-10, 10].  F* = 0 at x* = (1, 1, ..., 1).
    """
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)


def zakharov(x: np.ndarray) -> float:
    """
    Zakharov function (unimodal).
    Domain: [-5, 10].  F* = 0 at x* = 0.
    """
    D = len(x)
    indices = np.arange(1, D + 1)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(0.5 * indices * x)
    return float(sum1 + sum2 ** 2 + sum2 ** 4)


def dixon_price(x: np.ndarray) -> float:
    """
    Dixon-Price function (unimodal, valley-shaped).
    Domain: [-10, 10].  F* = 0.
    """
    D = len(x)
    term1 = (x[0] - 1.0) ** 2
    indices = np.arange(2, D + 1)
    term2 = np.sum(indices * (2.0 * x[1:] ** 2 - x[:-1]) ** 2)
    return float(term1 + term2)


def michalewicz(x: np.ndarray) -> float:
    """
    Michalewicz function (multimodal, steep ridges).
    Domain: [0, pi].  F* depends on D (approx -1.8013 for D=2, -4.687 for D=5, -9.66 for D=10).
    Note: For simplicity, we set optimal=0 and report absolute values.
    """
    D = len(x)
    m = 10  # Steepness parameter
    indices = np.arange(1, D + 1)
    return float(-np.sum(np.sin(x) * np.sin(indices * x ** 2 / np.pi) ** (2 * m)))


# ──────────────────────────────────────────────────────────────────
# Registry of all benchmark functions
# ──────────────────────────────────────────────────────────────────

BENCHMARK_FUNCTIONS = {
    # Original 3 from NCRO poster
    "Sphere": {
        "function": sphere,
        "bounds": (-100.0, 100.0),
        "optimal": 0.0,
    },
    "Rastrigin": {
        "function": rastrigin,
        "bounds": (-5.12, 5.12),
        "optimal": 0.0,
    },
    "Ackley": {
        "function": ackley,
        "bounds": (-32.768, 32.768),
        "optimal": 0.0,
    },
    # Additional standard benchmarks
    "Rosenbrock": {
        "function": rosenbrock,
        "bounds": (-30.0, 30.0),
        "optimal": 0.0,
    },
    "Schwefel": {
        "function": schwefel,
        "bounds": (-500.0, 500.0),
        "optimal": 0.0,
    },
    "Griewank": {
        "function": griewank,
        "bounds": (-600.0, 600.0),
        "optimal": 0.0,
    },
    "Levy": {
        "function": levy,
        "bounds": (-10.0, 10.0),
        "optimal": 0.0,
    },
    "Zakharov": {
        "function": zakharov,
        "bounds": (-10.0, 10.0),
        "optimal": 0.0,
    },
    "DixonPrice": {
        "function": dixon_price,
        "bounds": (-10.0, 10.0),
        "optimal": 0.0,
    },
    "Michalewicz": {
        "function": michalewicz,
        "bounds": (0.0, np.pi),
        "optimal": 0.0,  # Actual optimum is negative; we report |F_best - F*|
    },
}
