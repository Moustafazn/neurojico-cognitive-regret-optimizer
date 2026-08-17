"""
Comparison algorithms for benchmarking NCRO against state-of-the-art.

Classic baselines:
  - PSO, DE, GWO, WOA

Adaptive DE family (CEC competition winners):
  - SHADE (2013), L-SHADE (2014), jSO (CEC 2017)
  - NL-SHADE-LBC (CEC 2022 winner), L-SRTDE (CEC 2024)
"""

from .pso import PSOOptimizer
from .de import DEOptimizer
from .shade import SHADEOptimizer
from .lshade import LSHADEOptimizer
from .jso import jSOOptimizer
from .nlshade_lbc import NLSHADELBCOptimizer
from .lsrtde import LSRTDEOptimizer
from .gwo import GWOOptimizer
from .woa import WOAOptimizer

__all__ = [
    "PSOOptimizer", "DEOptimizer",
    "GWOOptimizer", "WOAOptimizer",
    "SHADEOptimizer", "LSHADEOptimizer",
    "jSOOptimizer", "NLSHADELBCOptimizer", "LSRTDEOptimizer",
]
