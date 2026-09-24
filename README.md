# Neurojico Cognitive Regret Optimizer (NCRO)

A novel metaheuristic optimization algorithm that uses **cognitive regret** and **counterfactual learning** to adaptively balance exploration and exploitation.

## Comparison Algorithms

| Algorithm | Type | Competition |
|-----------|------|-------------|
| CMA-ES | Evolution strategy | Gold standard (Hansen 2001) |
| L-SHADE | Adaptive DE | CEC 2014 winner |
| jSO | Adaptive DE | CEC 2017 winner |
| IMODE | Multi-operator DE | CEC 2020 winner |
| NL-SHADE-LBC | Adaptive DE | CEC 2022 winner |
| L-SRTDE | Adaptive DE | CEC 2024 |

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
