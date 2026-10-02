# A Neurojico-Inspired Optimizer based on Cognitive Regret, Counterfactual Reasoning, and Adaptive Memory

## Abstract

Most population-based metaheuristics use what they discover as it comes during the search and often do not consider the potential opportunity cost associated with missing out on finding a better alternative. This paper presents the Neurojico Cognitive Regret Optimizer (NCRO), which is an optimization framework based upon cognition and incorporates counterfactuality reasoning, cognitive regret, adaptive memory, and behavioral adaptation in one searching mechanism. The NCRO creates both candidate solutions actually found by the search algorithm and those not found (the counterfactually generated solutions) under identical search conditions and produces a bounded cognitive-regret signal from the performance difference between them. There are three types of memories used to support this searching: regret memory, counterfactual-success memory, and progress memory. Each type of memory can influence how much exploration occurs versus how much exploitation takes place, how quickly or slowly counterfactual knowledge is learned, and whether the directionality of the search changes over time through a single cognitive-motion operator and a 4-way elite selection process. The theoretical analysis provides a bound on regret memory, shows that regrets vanish as the optimizer approaches behavioral equilibrium, and proves that the fitness function does not decrease monotonically when evaluated deterministically. Benchmark functions representing a variety of search landscapes have been studied using variously sized instances of the Sphere, Rastrigin, Ackley, Griewank, Rosenbrock, Schwefel, Zakharov, and Levy problems. On the Rastrigin and Griewank functions, NCRO found the global optimum. NCRO was able to achieve machine precision convergence on Ackley for all instance sizes up to 100 variables. Ablation studies have identified that regret memory, counterfactual reasoning, and counterfactual-success memory are critical components to the effective operation of the search. Additional validation is provided through scalability, runtime, and statistical analysis of NCRO. Additionally, NCRO has successfully solved a real-world smart microgrid dispatch problem with eight different operating scenarios, yielding 100 percent feasibility and success rate. Overall these results suggest that NCRO can utilize missed opportunities as learning signals that will allow it to optimize a wide range of complex numerical and engineering problems using a robust cognitive approach.

A novel metaheuristic optimization algorithm that uses **cognitive regret** and **counterfactual learning** to adaptively balance exploration and exploitation.

## Project Structure

```
neurojico-cognitive-regret-optimizer/
├── main.py                          # Single entry point
├── ncro/microgrid_experiment.py     # Microgrid comparison
├── ncro/variant_experiment.py       # Controlled-variant
├── requirements.txt
├── ncro/
│   ├── optimizer.py                 # NCRO core algorithm
│   ├── benchmarks.py                # Benchmark functions
│   ├── experiment.py                # Runner + plotting
│   ├── ablation.py                  # Ablation variants
│   └── statistics.py                # Wilcoxon + Friedman
└── results/                         # Auto-generated CSV + PNG
```

## How to Run

```bash
pip install -r requirements.txt

# Full experiment
python main.py

# Quick validation
python main.py --quick
```

## Citation

```bibtex
@misc{hassanien2026neurojico,
  title  = {A Neurojico-Inspired Optimizer based on Cognitive Regret, Counterfactual Reasoning, and Adaptive Memory},
  author = {Hassanien, Aboul Ella and Zein, Moustafa},
  year   = {2026},
  url    = {https://github.com/Moustafazn/neurojico-cognitive-regret-optimizer},
}
```
