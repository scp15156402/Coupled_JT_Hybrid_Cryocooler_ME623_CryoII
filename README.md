# Joule-Thomson cooler model (He-3, after Zhang & Dang 2021)

Run everything from this folder with `python jt_run.py <command>`.

| Command | What it does |
|---|---|
| `python jt_run.py check` | Are the He-3 tables present and sane? |
| `python jt_run.py validate` | **Workflow 1.** Data check, self-tests, reproduce the paper, scorecard |
| `python jt_run.py solve cases/xyz.json` | **Workflows 2 and 3.** Solve a cycle from a case file |
| `python jt_run.py optimize ...` | Last-stage optimizer (best P_U and T_pre) |
| `python jt_run.py phex ...` | PHEX load map for the warmer precooling temperatures |

## Workflow 1: validate the paper (once the He-3 tables exist)

1. Put `SinglePhase_He_3.csv` and `Saturation_He_3.csv` in `data/he3/` and fill in `data/he3/README.md`.
2. `python jt_run.py validate`  (add `--quick` for a fast first look)

It stops at the first failing step: data check, then unit tests, then the five reproduction
sections (Eq. 35/36 fits, Fig. 5, Fig. 7, mass flux, Figs. 15/16), then a pass/fail scorecard
against the paper's numbers. Plots, CSVs and `scorecard.json` land in `results/validation/`.
The tolerances (`TOL` at the top of `jt_run.py`) are provisional; adjust them after the first real run.
Misses in model-dependent checks (HEM flow versus the paper's flux method) are only warnings.

## Workflow 2: coolers from other papers

1. Copy `cases/_template.json` to `cases/<author_year>.json` and fill it in from the paper.
2. Put the paper's reported values under `"reported"` (for example `"Q_C_mW": 10.0`).
3. `python jt_run.py solve cases/<author_year>.json`  (several files or `cases/*.json` also work)

Output per case, in `results/cases/<name>/`: `states.csv` (P, T, h at every state), `result.json`
(inputs, metrics, flags, comparison with the reported values), `case.json` (a copy of the input),
and `sweep.csv` if the case has a `"sweep"` block.

## Workflow 3: your own inputs

Same as workflow 2 without the `"reported"` block. To scan a variable, add for example
`"sweep": {"P_U": [0.3, 0.4, 0.5], "eta_last": [1.0, 0.97]}` (all combinations are solved).
For the best P_U and T_pre of the last stage use `python jt_run.py optimize --help`.

## What the solver assumes (read before comparing with other papers)

- Low-side pressure is P_sat(T_C); no pressure drop anywhere.
- Mass flow comes from a choked orifice (HEM model, discharge coefficient 1) and cannot be set.
  Without `d_um` only specific duties (kJ/kg) are returned.
- Precooling is imposed: `T_pre` is the gas temperature leaving each precooling HX.
- Each counterflow HX has effectiveness eta = Q / Q_max (Q_max is the smaller of the warm-end and cold-end limits).
- No heat leaks, no axial conduction.

## Folders

```
jt/                code (cycle_core, properties, design_optimize, phex_load_map)
tests/             unit tests (python -m unittest discover -s tests -t .)
data/he3/          He-3 tables and their provenance
cases/             input files for workflows 2 and 3
results/           everything generated (git-ignored, safe to delete)
Papers/            reference papers
jt_run.py          the front door
reproduce_zhang_dang_2021.py   paper reproduction (called by `validate`)
```
