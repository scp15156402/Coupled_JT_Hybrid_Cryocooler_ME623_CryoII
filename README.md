# Joule-Thomson cooler model (He-3 and other fluids)

A cycle model for precooled Joule-Thomson (JT) coolers built from counterflow heat
exchangers (CHEX) and precooling heat exchangers (PHEX). It reproduces
**Zhang & Dang, Cryogenics 116 (2021) 103282** (He-3, 1 K) and extends it to other
coolers, other fluids and your own designs. Everything runs from one command-line
script, `jt_run.py`. Full theory, code map and validation are in `jt_theory_and_code.pdf`.

## What it does

Given the cold-end temperature `T_C`, upstream pressure `P_U`, the gas temperature leaving
each precooler `T_pre`, and the effectiveness `eta` of each CHEX, it computes

- every state point (T, P, h, phase) around the cycle,
- the specific refrigeration `q_C` (kJ/kg) and, with a mass flow, `Q_C` and the load on each PHEX (mW),
- the best `P_U` and last-stage precooler temperature `T_pre,last` for `q_C` and `Q_C`,
- sensible upstream precooler temperatures when you do not know them,
- whether a design is physically possible, and why not if it is not.

Fluids: He-3 (your CSV tables) and any CoolProp fluid (He-4, Neon, Hydrogen, Parahydrogen,
Nitrogen, Argon, Oxygen, Methane, ...).

## Install

```
pip install numpy scipy pandas matplotlib CoolProp
```
Run everything from this folder. Put the He-3 tables in `data/he3/`
(`SinglePhase_He_3.csv`, `Saturation_He_3.csv`; see `data/he3/README.md`).

## Quick start

```
python run_all.py --quick      # everything: He-3 check, paper 1 scorecard, then the other papers (Windows: run_all.bat)
```
`run_all.py` runs the He-3 check, reproduces and scores Zhang & Dang 2021, then solves each other-paper case and compares it with the paper's numbers (PASS inside a tolerance for idealised paper values, INFO for measured values; summary in `results/validation_all/`). Drop `--quick` for the full-grid paper-1 run; `--skip-paper1` runs only the other papers.

Step by step:

```
python jt_run.py check                 # are the He-3 tables present and sane?
python jt_run.py validate --quick      # reproduce the paper, pass/fail scorecard
python jt_run.py run cases/zhang_dang_2021_ideal.json     # solve one case, save everything
```

To run everything in one go (data check, unit tests, validation, every case file; logs in `results/run_all/`):
```
python run_all.py --quick        # about 80 s;  add --optimize for the blank-input optimiser demo (+90 s)
```
On Windows, double-click `run_all.bat` (it installs missing packages first).

## Commands

| Command | What it does |
|---|---|
| `check` | He-3 tables present, readable, consistent |
| `validate [--quick \| --dense]` | **Workflow 1.** Data check, unit tests, paper reproduction, scorecard in `results/validation/`. `--dense` uses the fine grids of the report (241-point curves, 61-point optimisation grid, 40 precooling temperatures, 46 flux points, eta step 0.0025); individual grids: `--n-curve`, `--n-opt-grid`, `--n-tpre`, `--n-flux`, `--eta-step` |
| `new --fluid ... --T_C ... --eta ...` | Create a case file named by the standard scheme, with a data reference, and check feasibility |
| `feasible cases/x.json` | Is the design physically possible? Verdict and reasons |
| `run cases/x.json` | Solve (all inputs given) or **optimise** (blank inputs, or `--optimize`). Prints state points and PHEX loads, saves plots and CSVs |
| `runs` | List saved runs and the configuration each belongs to |
| `solve cases/*.json` | Plain solve with `sweep` and `reported`-value comparison (workflows 2 and 3) |
| `optimize ...`, `phex ...` | Original last-stage optimiser and PHEX load map (`--help` for options) |

### The three workflows

1. **Validate the paper:** `python jt_run.py validate` (`--quick` for coarse grids, `--dense` for the grids used in the report and presentation).
2. **Another paper's cooler:** write a case file from the paper (inputs, plus `reported` values and a `reference` block), then `python jt_run.py solve cases/<name>.json`. Cases from Liu et al. 2017 and Ma et al. 2019 are included (`liu2017_*`, `ma2019_*`); only their ideal-model values are used for comparison.
3. **Your own design:** `new`, then `feasible`, then `run`.

### Creating a case

```
python jt_run.py new --fluid he3 --T_C 1.0 --eta 1,1,1,1 --T_pre 80,25,8 --P_U 0.36 --d_um 20 \
    --citation "my design" --kind "Own design"
```
- `--eta` has one value per CHEX (warm to cold); N values means N CHEXs and N-1 precoolers.
- `--T_pre` has N-1 values; write `-` for any you want optimised (e.g. `80,-` or `-,-,-`).
- `--P_U -` (default) leaves the upstream pressure to the optimiser.
- Give either `--d_um` (orifice diameter, flow from the choked-flow model) or `--mdot_mg_s` (measured flow), not both. With neither you get specific duties only.
- The file is saved as `cases/{Fluid}_{T_C}K_{d}um_{N}_stage[_eta{x}].json`, for example `Helium_3_1K_20um_4_stage.json`. With a measured flow, `{d}um` becomes `{mdot}mgs`. If the name exists with different content a `_v2`, `_v3` suffix is added.

### Running

```
python jt_run.py feasible cases/x.json
python jt_run.py run cases/x.json                 # solve, or optimise if anything is blank
python jt_run.py run cases/x.json --optimize --drive Q_C --pu-max 2 --save-final
```
Options of `run`: `--drive {Q_C,Q_C_practical,q_C}`, `--flow-model {hem,paper_fit}`,
`--pu-min/--pu-max` (MPa), `--tpre-min/--tpre-max` (K), `--hem-grid`, `--save-final`
(writes the optimised values into a new filled-in case file).

What `run` does when something is blank:
1. if `T_pre,last` is blank, scans `P_U` x `T_pre,last` for the maximum `q_C` and for the maximum `Q_C` (the `Q_C` scan is the slow step, 20 s to 2 min);
2. if `T_pre,last` is given, finds the best `P_U` for `q_C` and `Q_C` at that `T_pre,last`;
3. chooses the upstream precooler temperatures by minimising the Carnot-weighted precooling load `sum_j Q_pre,j (T_H/T_j - 1)` (the upstream temperatures cannot change `q_C`, the mass flow or `Q_C`, only how the load is shared between precoolers), and saves a sensitivity map per upstream precooler;
4. solves the cycle at the chosen values.

### Extra figures

```
python make_extra_figures.py --n-pu 200 --outdir results/extra
```
Writes `QC_vs_PU_Tpre.png/.csv` (Q_C vs P_U for several last-stage precooling temperatures, T_C = 1 K, d = 20 um) and `cp_He3.png` (He-3 c_p at P_U and P_D). Options: `--n-pu`, `--n-t`, `--tpre 4 5 8 10 12`, `--hem-grid`, `--outdir`. A progress bar with elapsed time and an estimate of the time left is printed.

Optima in all plots are marked with open circles.

### Where results go

Every `run` writes `results/runs/<config name>/<timestamp>_<solve|optimize>/` containing
`config.json` (copy of the configuration), `meta.json` (config hash, data reference),
`result.json`, `states.csv`, `ph_cycle.png`, and for optimisation `map_qC.png`, `map_QC.png`,
`stage_curves.png`, `upstream_map.png/csv`. A run is tied to its configuration by a hash
of the inputs (the data reference is excluded, so you can add or correct it later).
`python jt_run.py runs` lists them. `solve` writes to `results/cases/<name>/`; `validate`
to `results/validation/`. `results/` is generated output and safe to delete.

### Case file keys

`name, fluid, T_C [K], P_U [MPa], T_H [K, default 300], T_pre [K, N-1 values], eta [N values],
d_um or mdot_mg_s, reference {citation, doi, location, kind, note}, reported {...}, sweep {...},
optimize {...}, source, notes`. Keys starting with `_` are ignored. See `cases/_template.json`.
`reported` names: `q_C_kJ_kg, mdot_mg_s, Q_C_mW, Q_pre1_mW, Q_pre2_mW, ..., T_JT_inlet_K, P_D_kPa`.
`sweep` can scan `T_C, P_U, T_H, d_um, eta_last, T_pre_last` (all combinations).

## Reading the feasibility verdict

`feasible` / `feasible with warnings` / `infeasible`. It checks, in order: the fluid's
supported temperature window (He-3: 0.5-3.2 K; CoolProp fluids: triple point to 0.995 T_crit);
`P_U > P_D = P_sat(T_C)`; `T_pre` strictly decreasing from `T_H` and above `T_C`;
`q_L = h_g(T_C) - h(P_U,T_C) > 0` (otherwise no refrigeration at any `T_pre`; the message gives
`P_U,max`); `dh_T = h(P_D,T) - h(P_U,T) > 0` at `T_pre,last` (otherwise above the inversion
temperature); a warning if `T_pre,last` is warmer than `T_pre,opt` (then `q_C` is limited by `dh_T`);
then a trial solve with flags classified into errors (precooler would have to heat the gas,
energy balance fails, no net refrigeration) and warnings (internal pinch, unchoked flow).

## What the solver assumes (read before comparing with other papers)

- Low-side pressure is `P_sat(T_C)`; no pressure drop anywhere.
- No heat leaks, no axial conduction, discharge coefficient 1.
- Mass flow comes from a homogeneous-equilibrium choked orifice or is given (`mdot_mg_s`); it is never solved from the compressor.
- Precooling is imposed: `T_pre` is the gas temperature leaving each precooler.
- Each CHEX has effectiveness `eta = Q/Q_max`, `Q_max` the smaller of the warm-end and cold-end limits.
- For non-helium fluids use `P_U` above the fluid's critical pressure (Ne 2.66, H2 1.30, N2 3.40 MPa); below it the hot stream condenses inside the exchangers and the property lookup fails.
- Real coolers will deliver less than the model predicts (heat leaks, return-line pressure drop). See the report (`report/jt_report.pdf`).
- In `new`, write placeholders with `=`, e.g. `--T_pre=-,-,8 --P_U=-` (a value starting with `-` is otherwise read as an option).

## Validation status

Dense grids (`validate --dense`), ideal model, identical inputs to each source.

- **Zhang & Dang 2021** (He-3 tables): 27 of 27 checks pass. `P_U,opt` 0.358 MPa (paper 0.36), `q_C,opt` 7.98 kJ/kg (8.0), `P_U,max` 1.131 MPa (1.13), `m_dot` 2.339 mg/s (2.34), `Q_C` 18.62 mW (18.7); with `eta_4 = 0.97`, 14.74 mW (14.7), a 20.8 % drop (paper about 21 %).
  - Optimum vs Eq. 36: 1.39 % mean, 3.72 % worst; `q_C,opt` vs Eqs. 35-36: 3.12 % mean.
  - `T_pre3,opt` vs Eq. 36 at 0.3/0.5/0.7/0.9 MPa: 0.01/0.23/0.34/0.78 K.
  - HEM flux vs Table 2 at T_C = 1/2/2.6/3 K: mean 0.25/0.41/0.51/0.69 %, worst 0.71/0.72/2.28/1.49 %.
- **Liu et al. 2017** (He-4): `Q_C` 53.1 mW vs 52.5 mW calculated in the paper (+1.1 %), with the last-CHEX effectiveness (0.9857) derived from the paper's temperatures.
- **Ma et al. 2019, design point** (He-4): 90.2-108.8 mW for eta 0.97-1 vs the 100 mW design value (eta about 0.985); 15 K precooler load 278 mW vs 260 mW.
- Measured performance from other papers is not used for validation: it includes heat leak and return-line pressure drop, which the ideal model leaves out by construction. The remaining measured case files are informative only.
- 29 unit tests: `python -m unittest discover -s tests -t .`
- The scorecard tolerances (`TOL` at the top of `jt_run.py`) are looser than the agreement obtained.

## Folders

```
jt_run.py                       the command-line front door
run_all.py / run_all.bat        run check, tests, validation and every case in one go
reproduce_zhang_dang_2021.py    paper reproduction (called by `validate`)
make_extra_figures.py           Q_C vs P_U curves and He-3 c_p
jt/
  cycle_core.py                 states, CHEX solver, solve_cycle, choked-flow model, CoolProp adapter
  properties.py                 Helium3Fluid: CSV-backed interpolation
  design_optimize.py            last-stage optimiser, Eq. 49, file naming
  phex_load_map.py              PHEX load map vs warmer precooler temperatures
  workflow.py                   configs, feasibility check, run storage, optimiser pipeline
data/he3/                       SinglePhase_He_3.csv, Saturation_He_3.csv, README.md
cases/                          case files (template, Zhang & Dang, other-paper cases, your own)
tests/                          unit tests
results/                        generated output
```

## Troubleshooting

- `cannot import name 'workflow' from 'jt'`: `workflow.py` must be inside the `jt/` folder.
- `No module named CoolProp` (or numpy/scipy/pandas/matplotlib): run the `pip install` line above.
- `missing file ... SinglePhase_He_3.csv`: put the tables in `data/he3/` and run `python jt_run.py check`.
- `OutOfRange` for He-3: the request is outside the tables (T_C above 3.2 K, a very low pressure, or the thin near-critical gap).
- Infeasible with "Saturation pressure ... within 1e-4 %" (non-helium): `P_U` is below the critical pressure; raise it.
