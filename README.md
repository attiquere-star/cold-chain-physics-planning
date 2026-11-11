# Cold-chain planning with CFL-consistent transport: code and results

[![DOI](https://zenodo.org/badge/DOI/[ZENODO-DOI].svg)](https://doi.org/[ZENODO-DOI])

This repository contains all the code and raw results behind Sections 5 and 6 of the manuscript "When Does Physical Resolution Pay Off? Forecast-Driven Cold-Chain Planning with CFL-Consistent Transport and Age-Structured Inventory".

Every number in the paper can be regenerated from these files. No result in the paper comes from any other source.

## Requirements

Python 3.10 or later with numpy, scipy (version 1.9 or later, for `scipy.optimize.milp` with HiGHS), pandas, scikit-learn, torch and matplotlib. All are open source; no commercial solver is needed.

## Layout

| Path | What it is |
|---|---|
| `cc/config.py` | All model and world parameters (Table 2 of the paper) and the Q10 loss-rate function |
| `cc/generator.py` | Synthetic world: weather, temperatures, route speeds, demand, promotions, and the stress scenarios |
| `cc/simulator.py` | Lot-based "true system" used to score every policy under realized conditions |
| `cc/planner.py` | Planning MILPs: `agg` (aggregated benchmark), `phy` (upwind finite volume on a CFL grid, with an option to violate CFL), `char` (characteristic transport); all with age-structured DC stock except `agg` |
| `cc/lp.py` | Small sparse model builder on top of `scipy.optimize.milp` (HiGHS) |
| `cc/forecast.py` | LSTM forecaster with known future covariates, ridge (ARIMAX-type) and seasonal-naive baselines |
| `cc/runner.py` | Forecast preparation, rolling-horizon evaluation and the decision-focused recalibration (Nelder–Mead on realized cost) |
| `run_main.py` | Main benchmark for one seed: forecast accuracy, DFL tuning and the nine methods over 42 test days |
| `run_scen.py` | Seven stress scenarios for one seed (21 test days) |
| `run_sens.py` | Shelf-life and route-length sensitivity for one seed (21 test days) |
| `run_scale.py` | Problem size and solve time for 4/8/16 customers and 5/7/10-day horizons |
| `run_rf.py` | Closed-loop check of relax-and-fix against the full MILP (perfect information) |
| `run_full.py` | Short robustness check: key methods with the full MILP instead of relax-and-fix (14 days, 4 seeds) |
| `run_fullmain.py` | Main benchmark rerun with the exact MILP (all integers kept, 30 s limit, 42 days) for one seed and method (Table 9) |
| `analyze_full.py` | Contrasts and Holm correction for the exact-MILP rerun |
| `run_bc.py` | Bias-corrected forecast baseline (BC-AGG, BC-CHAR) for one seed (Table 8) |
| `analyze_bc.py` | Exploratory contrasts for the bias-correction baseline |
| `cfl_demo.py` | Numerical demonstration of upwind diffusion and CFL instability |
| `fig_network.py` | Network diagram (Figure 1) |
| `analyze.py` | Statistics (paired Wilcoxon, Holm, bootstrap CIs, Cohen's dz), tables and Figures 2–6 |
| `out/` | Raw results (one JSON per seed and experiment) and logs |
| `out/analysis/` | Tables (CSV), `summary.txt` and figures (PNG at 300 dpi, and vector EPS) |
| `out/fc_*.pkl` | Trained forecasts and recalibration parameters per seed, reused by the scenario runs |

## Reproducing the results

```bash
# main benchmark, 12 seeds (about 23 minutes per seed on one core)
for s in $(seq 1 12); do python3 run_main.py $s out; done
python3 cfl_demo.py out
# scenario, sensitivity, scalability and solver checks (need the main results first)
for s in $(seq 1 8); do python3 run_scen.py $s out; python3 run_sens.py $s out; done
python3 run_scale.py out
python3 run_rf.py 1 out; python3 run_rf.py 2 out
for s in 1 2 3 4; do for m in FTO-AGG DFL-AGG FTO-CHAR DFL-CHAR ORACLE-AGG ORACLE-CHAR; do python3 run_full.py $s $m out; done; done
# exact-MILP rerun of the main benchmark (about 6 hours on four cores)
for s in $(seq 1 12); do for m in FTO-AGG DFL-AGG FTO-CHAR DFL-CHAR ORACLE-AGG ORACLE-CHAR; do python3 run_fullmain.py $s $m out; done; done
# bias-correction baseline (needs the main results; about 4 minutes per seed)
for s in $(seq 1 12); do python3 run_bc.py $s out; done
# tables and figures
python3 analyze.py out
python3 analyze_full.py out
python3 analyze_bc.py out
python3 fig_network.py
```

`jobs.txt`, `jobs_full.txt`, `jobs_fullmain.txt` and `queue.sh` run the same jobs four at a time. Setting `FAST=1` before `run_main.py` runs a three-day smoke test.

## Notes

- Random seeds fix the world, the forecaster initialization and everything else, so reruns give the same worlds. HiGHS solve times, and therefore which solves hit the time limit, can differ slightly between machines.
- All parameter values are illustrative and literature-informed, not calibrated to a company.

## Installing

```bash
pip install -r requirements.txt
```

## Citing

If you use this code, please cite the paper and the archived software (see `CITATION.cff`):

> Yatoo, A. H., Feng, Y., Rehman, A. U., Ghafoor, S., Zamani Gandomani, S., Muaad, A. Y., & Bin Heyat, M. B. ([YEAR]). When does physical resolution pay off? Forecast-driven cold-chain planning with CFL-consistent transport and age-structured inventory. *[JOURNAL, once accepted]*.

> Yatoo, A. H., Feng, Y., Rehman, A. U., Ghafoor, S., Zamani Gandomani, S., Muaad, A. Y., & Bin Heyat, M. B. ([YEAR]). Code and results for "When does physical resolution pay off?" (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/[ZENODO-DOI]

## Licence

MIT, see `LICENSE`.
