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
    # Hybrid functions
    "Hybrid1": {
        "function": None,  # placeholder, set below
        "bounds": (-100.0, 100.0),
        "optimal": 0.0,
    },
    "Hybrid2": {
        "function": None,  # placeholder, set below
        "bounds": (-100.0, 100.0),
        "optimal": 0.0,
    },
    # Composition functions
    "Composition1": {
        "function": None,  # placeholder, set below
        "bounds": (-100.0, 100.0),
        "optimal": 0.0,
    },
    "Composition2": {
        "function": None,  # placeholder, set below
        "bounds": (-100.0, 100.0),
        "optimal": 0.0,
    },
}


# ──────────────────────────────────────────────────────────────────
# Hybrid Functions
# Partition dimensions into groups, apply a different basic function
# to each group, then sum. Inspired by CEC hybrid functions.
# ──────────────────────────────────────────────────────────────────

def _sphere_component(z):
    return float(np.sum(z ** 2))

def _rastrigin_component(z):
    D = len(z)
    return float(10 * D + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z)))

def _rosenbrock_component(z):
    if len(z) < 2:
        return float(z[0] ** 2)
    return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2))

def _ackley_component(z):
    D = len(z)
    if D == 0:
        return 0.0
    sum_sq = np.sum(z ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * z))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / D))
                 - np.exp(sum_cos / D) + 20.0 + np.e)

def _griewank_component(z):
    D = len(z)
    if D == 0:
        return 0.0
    indices = np.arange(1, D + 1)
    return float(np.sum(z ** 2) / 4000.0
                 - np.prod(np.cos(z / np.sqrt(indices))) + 1.0)

def _levy_component(z):
    if len(z) < 2:
        return float(z[0] ** 2)
    w = 1.0 + (z - 1.0) / 4.0
    t1 = np.sin(np.pi * w[0]) ** 2
    t2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    t3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    return float(t1 + t2 + t3)

def _schwefel_2_22_component(z):
    """Schwefel 2.22: sum(|z|) + prod(|z|)."""
    abs_z = np.abs(z)
    return float(np.sum(abs_z) + np.prod(np.minimum(abs_z, 1e+15)))


def hybrid1(x: np.ndarray) -> float:
    """
    Hybrid Function 1: Sphere + Rastrigin + Rosenbrock.
    Splits D dimensions into 3 groups (30%-30%-40%) and applies a
    different function to each group. Domain: [-100, 100]. F* = 0.
    """
    D = len(x)
    n1 = max(1, int(0.3 * D))
    n2 = max(1, int(0.3 * D))
    z1, z2, z3 = x[:n1], x[n1:n1+n2], x[n1+n2:]
    return _sphere_component(z1) + _rastrigin_component(z2) + _rosenbrock_component(z3)


def hybrid2(x: np.ndarray) -> float:
    """
    Hybrid Function 2: Ackley + Griewank + Levy.
    Splits D dimensions into 3 groups (30%-30%-40%) and applies a
    different function to each group. Domain: [-100, 100]. F* = 0.
    """
    D = len(x)
    n1 = max(1, int(0.3 * D))
    n2 = max(1, int(0.3 * D))
    z1, z2, z3 = x[:n1], x[n1:n1+n2], x[n1+n2:]
    return _ackley_component(z1) + _griewank_component(z2) + _levy_component(z3)


# ──────────────────────────────────────────────────────────────────
# Composition Functions
# Combine multiple shifted basic functions with Gaussian weighting.
# Each sub-function has a different local attractor (shift vector).
# Inspired by CEC composition function design.
# ──────────────────────────────────────────────────────────────────

def _gaussian_weight(x, shift, sigma):
    """Gaussian weight for composition: higher when x is near shift."""
    d = np.sum((x - shift) ** 2)
    return np.exp(-d / (2.0 * len(x) * sigma ** 2))


def composition1(x: np.ndarray) -> float:
    """
    Composition Function 1: Sphere + Rastrigin + Ackley + Rosenbrock + Griewank.
    Five shifted sub-functions combined with Gaussian weighting.
    Domain: [-100, 100]. F* = 0 (when all weights collapse at origin).
    """
    D = len(x)
    rng = np.random.RandomState(42)  # deterministic shifts
    shifts = [rng.uniform(-50, 50, D) for _ in range(5)]
    # Set first shift to origin so global optimum is at 0
    shifts[0] = np.zeros(D)
    sigmas = [10.0, 20.0, 30.0, 20.0, 10.0]
    lambdas = [1.0, 1.0, 1.0, 1.0, 1.0]
    biases = [0.0, 100.0, 200.0, 300.0, 400.0]
    funcs = [_sphere_component, _rastrigin_component, _ackley_component,
             _rosenbrock_component, _griewank_component]

    weights = np.array([_gaussian_weight(x, s, sig)
                        for s, sig in zip(shifts, sigmas)])
    w_sum = np.sum(weights)
    if w_sum == 0:
        weights = np.ones(5) / 5.0
    else:
        weights /= w_sum

    result = 0.0
    for i in range(5):
        z = x - shifts[i]
        result += weights[i] * (lambdas[i] * funcs[i](z) + biases[i])
    return float(result)


def composition2(x: np.ndarray) -> float:
    """
    Composition Function 2: Schwefel_2.22 + Rastrigin + Ackley + Griewank + Sphere.
    Five shifted sub-functions with different sigma and bias values.
    Domain: [-100, 100]. F* = 0.
    """
    D = len(x)
    rng = np.random.RandomState(123)  # deterministic shifts (different seed)
    shifts = [rng.uniform(-50, 50, D) for _ in range(5)]
    shifts[0] = np.zeros(D)
    sigmas = [20.0, 10.0, 30.0, 20.0, 10.0]
    lambdas = [0.1, 1.0, 1.0, 1.0, 1.0]
    biases = [0.0, 100.0, 200.0, 300.0, 400.0]
    funcs = [_schwefel_2_22_component, _rastrigin_component, _ackley_component,
             _griewank_component, _sphere_component]

    weights = np.array([_gaussian_weight(x, s, sig)
                        for s, sig in zip(shifts, sigmas)])
    w_sum = np.sum(weights)
    if w_sum == 0:
        weights = np.ones(5) / 5.0
    else:
        weights /= w_sum

    result = 0.0
    for i in range(5):
        z = x - shifts[i]
        result += weights[i] * (lambdas[i] * funcs[i](z) + biases[i])
    return float(result)


# Wire hybrid/composition functions into registry
BENCHMARK_FUNCTIONS["Hybrid1"]["function"] = hybrid1
BENCHMARK_FUNCTIONS["Hybrid2"]["function"] = hybrid2
BENCHMARK_FUNCTIONS["Composition1"]["function"] = composition1
BENCHMARK_FUNCTIONS["Composition2"]["function"] = composition2
