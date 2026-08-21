# Neurojico Cognitive Regret Optimizer (NCRO)

A novel metaheuristic optimization algorithm that uses **cognitive regret** and **counterfactual learning** to adaptively balance exploration and exploitation. Each agent learns not only from the decision it made, but also from the decision it did NOT make.

## Core Motion Equation

```
x_i(t+1) = x_i(t)
          + w(t,M_R) · V_i(t)                   ← regret-aware momentum
          + α(t) · [1 + M_R(i)] · E_i(t)        ← regret amplifies exploration
          + β(t) · [1 - M_R(i)] · H_i(t)        ← regret dampens exploitation
          + γ(t) · C_i(t) · D_C(i,t)             ← counterfactual direction force
```

Where:
- **V_i** = x_i(t) − x_i(t−1): velocity (direction agent moved last step)
- **w** = (1−M_R)·0.4·(1−τ): momentum weight — low regret = keep going, high regret = stop and reconsider
- **E_i**: regret-driven exploration direction (standard differential when converging; multi-directional long-distance scouting when stuck)
- **H_i** = c₁·r₁·(p_i − x_i) + c₂·r₂·(g − x_i) (PSO-style exploitation)
- **D_C** = Y_C − x_i (counterfactual direction — learning from alternatives)
- **M_R** ∈ [0,1]: accumulated regret memory (EMA-smoothed)
- **C_i** ∈ [0,1]: counterfactual success memory (EMA-smoothed)

### Regret-Driven Exploration

The exploration direction E_i adapts based on each agent's regret state:
- **Low regret** (agent doing well): standard `E_i = x_r1 − x_r2` with regret-proportional minimum step
- **High regret + stuck** (M_R > 0.25, no progress): long-distance scouting toward 4 distant areas:
  1. Opposite of global best (search unexplored regions)
  2. Random search space position (pure exploration)
  3. Opposite of personal best (escape personal basin)
  4. Another agent's personal best (information sharing)
- Scout step size scales with regret: `E_i *= (0.3 + 0.7·M_R)` — higher regret = larger radius

## Key Mechanisms

| Component | Formula | Purpose |
|-----------|---------|---------|
| Actual candidate Y_A | x_i + q·α·E + (1−q)·β·H | What the agent decided |
| Counterfactual Y_C | x_i + (1−q)·α·E + q·β·H | What it could have decided |
| Cognitive regret | max(0, F(Y_A) − F(Y_C)) / (\|F_A\| + \|F_C\| + ε) | How much worse was the choice? |
| Regret memory | ρ·M_R + (1−ρ)·R̃_i | Smoothed learning from regret |
| CF success memory | ρ_c·C_i + (1−ρ_c)·SF_i | How often alternatives win |
| Adaptive q | q₀(t) + η_R·M_R + η_C·C_i − η_P·P_i | Per-agent E/E balance |
| Selection | greedy among {x_i, Y_A, Y_C, candidate} | Best of 4 candidates survives |

## Comparison Algorithms

| Algorithm | Year | Type | Reference |
|-----------|------|------|-----------|
| PSO | 1995 | Classic swarm intelligence | Kennedy & Eberhart |
| DE | 1997 | Classic differential evolution | Storn & Price |
| GWO | 2014 | Grey wolf social hierarchy | Mirjalili et al. |
| WOA | 2016 | Whale bubble-net hunting | Mirjalili & Lewis |
| HHO | 2019 | Harris hawks cooperative hunting | Heidari et al. |
| ABC | 2005 | Honey bee foraging behavior | Karaboga & Basturk |
| SCA | 2016 | Sine cosine oscillatory search | Mirjalili |

## Ablation Study

Each variant removes exactly ONE component to measure its contribution:

| Variant | What is removed | Motion equation change |
|---------|-----------------|----------------------|
| NCRO (Full) | Nothing — complete algorithm | w·V + α(1+M_R)·E + β(1−M_R)·H + γ·C·D_C |
| NCRO-NoRegret | Regret signal (M_R forced to 0) | α·E + β·H + γ·C·D_C |
| NCRO-NoCF | Counterfactual candidate + direction | α·E + β·H (no γ·C·D_C) |
| NCRO-NoAdaptEE | Adaptive q (fixed schedule only) | Forces unchanged, q = q₀(t) |
| NCRO-NoMemory | EMA memory (uses instant values) | Uses instant R, instant C |
| NCRO-NoMomentum | Regret-aware momentum | α(1+M_R)·E + β(1−M_R)·H + γ·C·D_C (no w·V) |

## Benchmark Functions (12 total, 4 categories)

| Category | Function | Domain | F* | Key Characteristic |
|----------|----------|--------|----|--------------------|
| **Unimodal** | Sphere | [-100, 100] | 0 | Smooth, convex |
| | Rosenbrock | [-30, 30] | 0 | Narrow curved valley |
| | Zakharov | [-10, 10] | 0 | Non-convex unimodal |
| **Multimodal** | Rastrigin | [-5.12, 5.12] | 0 | Many local optima |
| | Ackley | [-32.768, 32.768] | 0 | Flat outer region |
| | Griewank | [-600, 600] | 0 | Regular multimodal |
| | Schwefel | [-500, 500] | 0 | Deceptive, distant optimum |
| | Levy | [-10, 10] | 0 | Multimodal with ridges |
| **Hybrid** | Hybrid1 | [-100, 100] | 0 | Sphere + Rastrigin + Rosenbrock (split dims) |
| | Hybrid2 | [-100, 100] | 0 | Ackley + Griewank + Levy (split dims) |
| **Composition** | Composition1 | [-100, 100] | 0 | 5 shifted functions (Gaussian-weighted) |
| | Composition2 | [-100, 100] | 0 | 5 shifted functions (different landscape) |

## Experimental Settings

| Parameter | Value |
|-----------|-------|
| Population Size (N) | 30 |
| Maximum Iterations (T_max) | 500 |
| Independent Runs | 30 |
| Dimensionalities (D) | 10, 30, 50, 100 |
| Benchmark Functions | 12 (3 unimodal + 5 multimodal + 2 hybrid + 2 composition) |
| Stopping Criterion | Maximum iterations |
| Fair Comparison | NCRO: 500 iter (3N FEs/iter) = 45,000 FEs; Others: 1500 iter (N FEs/iter) = 45,000 FEs |

## Project Structure

```
neurojico-cognitive-regret-optimizer/
├── main.py                          # Full experiment pipeline
│                                    #   Part 1: Benchmark comparison (8 algs × 12 funcs × 4 dims)
│                                    #   Part 2: Ablation study (6 variants × 12 funcs × 4 dims)
│                                    #   Part 3: Scalability analysis (D=10→100)
│                                    #   Part 4: Runtime evaluation
├── requirements.txt                 # Dependencies (numpy, matplotlib, scipy)
├── pyproject.toml                   # Project config
├── README.md
├── notebook/
│   ├── literature_review.md         # 50 references (1982–2026)
│   └── theoretical_foundation.md    # Mathematical novelty & theory
├── report/
│   └── progress_report.tex          # LaTeX paper
├── results/                         # Auto-generated output (CSV + PNG)
│   ├── benchmark_comparison.csv     # All algorithms × functions × dims
│   ├── ablation_study.csv           # All ablation variants
│   ├── friedman_rankings.csv        # Friedman rankings + overall averages
│   ├── statistical_tests.csv        # Wilcoxon p-values + W/L/T summary
│   ├── runtime_analysis.csv         # Runtime per run & per iteration
│   ├── scalability_analysis.csv     # NCRO performance across D
│   ├── comparison_*.png             # Convergence plots (key functions, D=30)
│   ├── ablation_*.png               # Ablation convergence + bar charts
│   ├── ranking_summary.png          # Friedman ranking bar chart
│   ├── scalability_convergence.png  # NCRO convergence across dimensions
│   ├── runtime_comparison.png       # Runtime bar chart
│   └── statistical_analysis.json    # Full statistical reference
└── ncro/
    ├── __init__.py
    ├── optimizer.py                 # NCRO core algorithm
    ├── benchmarks.py               # 12 benchmark functions (4 categories)
    ├── experiment.py               # Runner + plotting + timing
    ├── ablation.py                 # 5 ablation variants + full (6 total)
    ├── statistics.py               # Wilcoxon signed-rank + Friedman tests
    └── comparisons/
        ├── base.py                 # Shared OptResult dataclass
        ├── pso.py                  # PSO (1995)
        ├── de.py                   # DE/rand/1/bin (1997)
        ├── gwo.py                  # GWO (2014)
        ├── woa.py                  # WOA (2016)
        ├── hho.py                  # HHO (2019)
        ├── abc.py                  # ABC (2005)
        └── sca.py                  # SCA (2016)
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

### Output

The experiment generates **6 CSV tables** and **7+ PNG plots** in the `results/` folder:

**CSV Tables:**
- `benchmark_comparison.csv` — All 8 algorithms × 12 functions × 4 dimensions
- `ablation_study.csv` — 6 NCRO variants with ER, XR, CSR, regret metrics
- `friedman_rankings.csv` — Per-test-case + overall average rankings
- `statistical_tests.csv` — Wilcoxon signed-rank p-values + win/loss/tie summary
- `runtime_analysis.csv` — Average runtime per run and per iteration
- `scalability_analysis.csv` — NCRO quality metrics across D={10,30,50,100}

**Plots:**
- `comparison_*.png` — Convergence curves for key functions (D=30)
- `ablation_*.png` — Ablation convergence + bar charts
- `ranking_summary.png` — Overall Friedman ranking bar chart
- `scalability_convergence.png` — NCRO convergence across all dimensions
- `runtime_comparison.png` — Computational cost comparison

## Performance Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Best Fitness | F_best(t) = min_i F(x_i(t)) | Solution quality |
| Accuracy | Acc = \|F_best − F*\| | Distance from optimum |
| Exploration Ratio | ER = N_exploration / N_total | Fraction exploring |
| Exploitation Ratio | XR = N_exploitation / N_total | Fraction exploiting |
| CF Success Rate | CSR = N(F(Y_C) < F(Y_A)) / N | Counterfactual advantage |
| Average Regret | R̄ = (1/N) Σ R_i | Population regret level |

## Code Availability

The complete source code for NCRO is publicly available at:

🔗 **https://github.com/Moustafazn/neurojico-cognitive-regret-optimizer**
