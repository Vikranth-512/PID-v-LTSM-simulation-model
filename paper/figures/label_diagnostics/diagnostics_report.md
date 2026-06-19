# Label Degeneracy Diagnostics Report

## Part 1 — Root-Cause Ranking (Evidence-Based)

### 1. Single-impulse open-loop rollout (DOMINANT)
- Terminal EC spread across candidates: mean **nan**
- States with spread < 0.01: **0.0%**
- Rollout structure test (D8): see `rollout_structure_comparison_baseline.png`

### 2. Safety penalty flat across candidates (HIGH)
- Safety mean cost share: **72.5%**
- Safety std across candidates (mean per state): **41891.7302**

### 3. Monotonic dose preference below target (HIGH)
- Strictly decreasing objective below target: **1/6** EC levels

### 4. Near-tie argmin landscape (HIGH)
- Margin median: **1.168%**, p90: **5429304.121%**
- Decisions with margin < 0.5%: **30.7%**

### 5. Dose equivalence classes (MEDIUM)
- Unique doses: **33**
- Within-dose variance: **938066234.297571** vs between: **31010491.8419**

### 6. Nutrient/action terms too weak (MEDIUM)
- Nutrient share: **0.65%**, action: **0.05%**

### 7. Horizon extends collapse (LOW-MEDIUM)
- See `optimal_action_distribution_vs_horizon_baseline.png`

## Part 2 — Diagnostic Metrics Summary

### D1 Cost Breakdown
- Safety: 72.5% | Nutrient: 0.65% | Action: 0.05%

### D2 Terminal Collapse
- Mean EC_end spread: nan

### D3 Margins
- Mean 1454120.482%, median 1.168%

### D6 Safety Ablation
- **A_full**: unique_doses=8, pct_max=54.0%, entropy=0.93
- **B_no_safety**: unique_doses=10, pct_max=53.3%, entropy=0.96
- **C_no_terminal**: unique_doses=2, pct_max=59.3%, entropy=0.68
- **D_no_collapse**: unique_doses=8, pct_max=54.0%, entropy=0.93

### D8 Rollout Structure
- **impulse**: unique_doses=8, pct_max=54.0%, margin_med=1.168%
- **repeat**: unique_doses=13, pct_max=22.0%, margin_med=4.209%

### Current Label Distribution
- pct flowrate=0: 49.1%, pct flowrate=5: 48.8%
- dose entropy: 0.81 nats
- top pairs: [((np.float64(5.0), np.float64(30.0)), 28815), ((np.float64(0.0), np.float64(0.0)), 21983), ((np.float64(0.0), np.float64(5.0)), 7013)]

## Part 3 — Recommended Fixes (Ranked)

1. **Repeated-dosing rollout** during horizon (structural) — evidence D8
2. **Remove collapse penalty under open-loop coast** or shorten evaluation horizon (structural/objective)
3. **Remove under-target nutrient relief** (objective) — evidence D4 monotonicity
4. **Min-dose tie-break** among ε-optimal candidates (conservative)
5. Weight tuning alone — lowest priority

## Figures

All plots in `paper/figures/label_diagnostics/`.

---

## Before vs After Fix

| Metric | Baseline | After |
|--------|----------|-------|
| Margin median (%) | 1.168 | 4.209 |
| EC_end spread | nan | nan |
| D8 repeat unique doses | 13 | N/A |
| Label dose entropy | 0.81 | 2.0331366738014323 |
| pct max dose (labels) | 48.8 | 23.333333333333332 |