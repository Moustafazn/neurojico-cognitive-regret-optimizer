"""
Comparison algorithms for benchmarking NCRO against state-of-the-art.

Recent and competitive algorithms (CEC competition winners + gold standard):
  - CMA-ES          — Gold-standard evolution strategy (Hansen 2001/2016)
  - L-SHADE         — CEC 2014 winner, adaptive DE with pop. reduction
  - jSO             — CEC 2017 winner, weighted mutation adaptive DE
  - IMODE           — CEC 2020 winner, multi-operator adaptive DE
  - NL-SHADE-LBC    — CEC 2022 winner, non-linear pop. reduction + bias change
  - L-SRTDE         — CEC 2024, success-rate-based adaptive DE
"""

from .cmaes import CMAESOptimizer
from .lshade import LSHADEOptimizer
from .jso import jSOOptimizer
from .imode import IMODEOptimizer
from .nlshade_lbc import NLSHADELBCOptimizer
from .lsrtde import LSRTDEOptimizer

__all__ = [
    "CMAESOptimizer",
    "LSHADEOptimizer",
    "jSOOptimizer",
    "IMODEOptimizer",
    "NLSHADELBCOptimizer",
    "LSRTDEOptimizer",
]
