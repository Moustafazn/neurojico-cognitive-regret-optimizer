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
| Cognitive regret | max(0, F(Y_A) − F(Y_C)) / (|F_A| + |F_C| + ε) | How much worse was the choice? |
| Regret memory | ρ·M_R + (1−ρ)·R̃_i | Smoothed learning from regret |
| CF success memory | ρ_c·C_i + (1−ρ_c)·SF_i | How often alternatives win |
| Adaptive q | q₀(t) + η_R·M_R + η_C·C_i − η_P·P_i | Per-agent E/E balance |
| Selection | greedy among {x_i, Y_A, Y_C, candidate} | Best of 4 candidates survives |

## Comparison Algorithms (1995–2024)

| Algorithm | Year | Type | Reference |
|-----------|------|------|-----------|
| PSO | 1995 | Classic swarm intelligence | Kennedy & Eberhart |
| DE | 1997 | Classic differential evolution | Storn & Price |
| GWO | 2014 | Grey wolf social hierarchy | Mirjalili et al. |
| WOA | 2016 | Whale bubble-net hunting | Mirjalili & Lewis |
| SHADE | 2013 | Success-history adaptive DE | Tanabe & Fukunaga |
| L-SHADE | 2014 | SHADE + population reduction | Tanabe & Fukunaga, CEC winner |
| jSO | 2017 | Weighted mutation DE | Brest et al., CEC 2017 winner |
| NL-SHADE-LBC | 2022 | Non-linear SHADE + bias change | Stanovov, **CEC 2022 winner** |
| L-SRTDE | 2024 | Success rate-based adaptive DE | Stanovov et al., **CEC 2024** |

## Ablation Study

Each variant removes exactly ONE component to measure its contribution:

| Variant | What is removed | Motion equation change |
|---------|-----------------|----------------------|
| NCRO (Full) | Nothing — complete algorithm | α(1+M_R)·E + β(1−M_R)·H + γ·C·D_C |
| NCRO-NoRegret | Regret signal (M_R forced to 0) | α·E + β·H + γ·C·D_C |
| NCRO-NoCF | Counterfactual candidate + direction | α·E + β·H (no γ·C·D_C) |
| NCRO-NoAdaptEE | Adaptive q (fixed schedule only) | Forces unchanged, q = q₀(t) |
| NCRO-NoMemory | EMA memory (uses instant values) | Uses instant R, instant C |

## Project Structure

```
neurojico-cognitive-regret-optimizer/
├── main.py                          # Full experiment pipeline (Part 1: benchmark, Part 2: ablation)
├── requirements.txt                 # Dependencies (numpy, matplotlib, scipy)
├── pyproject.toml                   # Project config
├── README.md
├── notebook/
│   ├── literature_review.md         # 50 references (1982–2026)
│   └── theoretical_foundation.md    # Mathematical novelty & theory
├── results/                         # JSON + 300 DPI PNG (auto-created)
└── ncro/
    ├── __init__.py
    ├── optimizer.py                 # NCRO core algorithm
    ├── benchmarks.py               # 10 benchmark functions
    ├── experiment.py               # Runner + plotting + ablation plots + JSON
    ├── ablation.py                 # 4 ablation variants (V2-consistent)
    ├── statistics.py               # Wilcoxon signed-rank + Friedman tests
    └── comparisons/
        ├── base.py                 # Shared OptResult dataclass
        ├── pso.py                  # PSO (1995)
        ├── de.py                   # DE/rand/1/bin (1997)
        ├── gwo.py                  # GWO (2014)
        ├── woa.py                  # WOA (2016)
        ├── shade.py                # SHADE (2013)
        ├── lshade.py               # L-SHADE (2014)
        ├── jso.py                  # jSO (CEC 2017)
        ├── nlshade_lbc.py          # NL-SHADE-LBC (CEC 2022)
        └── lsrtde.py               # L-SRTDE (CEC 2024)
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

Outputs to `results/` folder:
- JSON files: per-algorithm results + final_summary.json + statistical_analysis.json
- PNG plots: convergence curves + comparison plots + ablation plots (300 DPI)
- Console: performance tables + Wilcoxon signed-rank / Friedman statistical tests

## Fair Comparison

NCRO uses **3N function evaluations per iteration** (Y_A + Y_C + candidate motion). All comparison algorithms receive **3× iterations** to match the total FE budget:

- NCRO: T=500, N=30 → 45,000 FEs
- Others: T=1500, N=30 → 45,000 FEs

## Performance Metrics (Section 12)

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Best Fitness | F_best(t) = min_i F(x_i(t)) | Solution quality |
| Accuracy | Acc = \|F_best − F*\| | Distance from optimum |
| Exploration Ratio | ER = N_exploration / N_total | Fraction exploring |
| Exploitation Ratio | XR = N_exploitation / N_total | Fraction exploiting |
| CF Success Rate | CSR = N(F(Y_C) < F(Y_A)) / N | Counterfactual advantage |
| Average Regret | R̄ = (1/N) Σ R_i | Population regret level |
