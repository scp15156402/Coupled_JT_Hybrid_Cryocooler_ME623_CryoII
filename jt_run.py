"""Front door for the Joule-Thomson cooler project.

    python jt_run.py check                      He-3 property tables: present and sane?
    python jt_run.py validate                   Workflow 1: reproduce the paper and score it
    python jt_run.py solve cases/xyz.json ...   Workflows 2 and 3: solve cycles from case files
    python jt_run.py optimize <args>            last-stage optimizer (python -m jt.design_optimize)
    python jt_run.py phex <args>                PHEX load map       (python -m jt.phex_load_map)
    python jt_run.py help

Run it from the project root. Results go to results/ (never into the code folders).
See README.md for the three workflows and the case-file format.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
DATA_HE3 = ROOT / "data" / "he3"
HE3_NAMES = {"he3", "helium3", "helium-3", "he-3"}

# ---------------------------------------------------------------------------
# Scorecard tolerances for `validate` (PROVISIONAL - tune after the first real He-3 run).
# mode "abs": |ours - paper| <= tol ; "pct": relative difference in % <= tol ;
# "max": ours <= tol ; "zero": ours must be 0.
# kind "anchor"/"consistency": a miss is a FAIL. kind "model": expected to differ
# between our HEM flow and the paper's real-gas flux method, a miss is only a WARN.
TOL = {
    "P_U_opt_MPa": 0.03,           # fig5: vs 0.36 MPa
    "q_C_opt_kJ_kg": 0.4,          # fig5: vs 8.0 kJ/kg
    "P_U_max_MPa": 0.05,           # fig5: vs 1.13 MPa
    "qC_vs_min_rel": 1e-6,         # fig5: q_C vs min(dh_T, q_L)       (implementation consistency)
    "gap_dhT_qL_rel": 0.01,        # fits: dh_T = q_L at the optimum    (Eq. 34 consistency)
    "eq36_mape_pct": 5.0,          # fits: P_U,opt vs Eq. 36, mean relative error
    "eq36_max_pct": 10.0,          # fits: P_U,opt vs Eq. 36, worst relative error
    "eq35_mape_pct": 5.0,          # fits: q_C,opt vs Eq. 35+36 chain, mean relative error
    "tpre_opt_K": 1.5,             # fig7: T_pre3,opt vs inverted Eq. 36
    "flux_vs_paper_mape_pct": 15.0,  # flux: HEM vs Table 2 curve (model difference expected)
    "flux_refit_max_pct": 2.0,     # flux: Eq. 49 form describes the HEM data
    "Q_C_mW": 1.0,                 # fig15: vs 18.7 and 14.7 mW        (flow-model dependent)
    "mdot_mg_s": 0.15,             # fig15: vs 2.34 mg/s               (flow-model dependent)
    "drop_pct_points": 3.0,        # fig15: Q_C drop at eta = 0.97 vs 21.4 %
}


# ---------------------------------------------------------------------------
# helpers

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "case"


def _num(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not math.isfinite(v) else f"{v:.{nd}g}"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _is_he3(name: str) -> bool:
    return str(name).strip().lower() in HE3_NAMES


# ---------------------------------------------------------------------------
# check : He-3 data files

def check_he3(singlephase: Optional[str] = None, saturation: Optional[str] = None) -> List[Tuple[str, str]]:
    """Return [(status, message)] with status PASS / FAIL / INFO."""
    out: List[Tuple[str, str]] = []
    sp = Path(singlephase) if singlephase else DATA_HE3 / "SinglePhase_He_3.csv"
    sat = Path(saturation) if saturation else DATA_HE3 / "Saturation_He_3.csv"
    missing = [p for p in (sp, sat) if not p.is_file()]
    if missing:
        for p in missing:
            out.append(("FAIL", f"missing file: {p}"))
        out.append(("INFO", f"put the TA's two CSV files in {DATA_HE3} (see data/he3/README.md)"))
        return out
    out.append(("PASS", f"found {sp.name} and {sat.name} (sha256 {_sha(sp)} / {_sha(sat)})"))
    try:
        from jt.properties import Helium3Fluid
        f = Helium3Fluid(str(sp), str(sat))
    except Exception as exc:  # loader raises DataError with a descriptive message
        out.append(("FAIL", f"loader rejected the files: {type(exc).__name__}: {exc}"))
        return out
    out.append(("PASS", "loader accepted both files (columns, units, monotonicity, saturation split)"))

    t0, t1 = float(f.T[0]), float(f.T[-1])
    p0, p1 = float(f.P[0]), float(f.P[-1])
    out.append(("PASS" if t0 <= 1.0 and t1 >= 30.0 else "FAIL",
                f"single-phase T range {t0:g}-{t1:g} K (needs <= 1 K and >= 30 K)"))
    out.append(("PASS" if p0 <= 1.16e3 and p1 >= 2.0e6 else "FAIL",
                f"single-phase P range {p0:g}-{p1:g} Pa (needs <= 1.16 kPa and >= 2 MPa)"))
    tmin, tmax = float(getattr(f, "Tmin_sat", math.nan)), float(getattr(f, "Tsat_max", math.nan))
    out.append(("PASS" if tmin <= 1.0 and tmax >= 3.0 else "FAIL",
                f"saturation table spans {tmin:g}-{tmax:g} K (needs <= 1 K and >= 3 K)"))
    try:
        ps = float(f.Psat(1.0))
        dev = 100.0 * abs(ps - 1160.0) / 1160.0
        out.append(("PASS" if dev <= 3.0 else "FAIL",
                    f"P_sat(1.0 K) = {ps:.1f} Pa; paper quotes about 1.16 kPa ({dev:.1f} % apart)"))
        h = f.h(0.36e6, 8.0)
        t_back = f.T_from_Ph(0.36e6, h)
        out.append(("PASS" if abs(t_back - 8.0) <= 2e-3 else "FAIL",
                    f"h(P,T) -> T(P,h) round trip at 0.36 MPa, 8 K returns {t_back:.5f} K"))
    except Exception as exc:
        out.append(("FAIL", f"property lookups failed: {type(exc).__name__}: {exc}"))
    out.append(("INFO", "the reference state of h and s is not checked; cycles only use differences"))
    return out


def _print_checks(items: Sequence[Tuple[str, str]]) -> bool:
    for status, msg in items:
        print(f"  {status:<4}  {msg}")
    return not any(s == "FAIL" for s, _ in items)


def cmd_check(rest: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="jt_run.py check", description="Check the He-3 property tables.")
    ap.add_argument("--singlephase", help="path to SinglePhase CSV (default data/he3/SinglePhase_He_3.csv)")
    ap.add_argument("--saturation", help="path to Saturation CSV (default data/he3/Saturation_He_3.csv)")
    a = ap.parse_args(rest)
    print("He-3 property tables")
    return 0 if _print_checks(check_he3(a.singlephase, a.saturation)) else 1


# ---------------------------------------------------------------------------
# validate : workflow 1

def _check(rows: List[Dict[str, Any]], section: str, name: str, ours: Any, ref: Any, tol: float,
           mode: str, kind: str = "anchor", unit: str = "") -> None:
    status = "N/A"
    try:
        o = float(ours) if ours is not None and not isinstance(ours, bool) else ours
        if isinstance(o, float) and math.isfinite(o):
            if mode == "abs":
                status = "PASS" if abs(o - float(ref)) <= tol else "FAIL"
            elif mode == "pct":
                status = "PASS" if 100.0 * abs(o - float(ref)) / abs(float(ref)) <= tol else "FAIL"
            elif mode == "max":
                status = "PASS" if o <= tol else "FAIL"
            elif mode == "zero":
                status = "PASS" if o == 0 else "FAIL"
        elif isinstance(ours, bool) and mode == "false":
            status = "PASS" if ours is False else "FAIL"
    except (TypeError, ValueError, ZeroDivisionError):
        status = "N/A"
    if status == "FAIL" and kind == "model":
        status = "WARN"
    rows.append({"section": section, "check": name, "ours": ours, "paper": ref, "tol": tol,
                 "mode": mode, "kind": kind, "unit": unit, "status": status})


def build_scorecard(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Turn reproduction_report.json into pass/fail rows against the paper's numbers."""
    rows: List[Dict[str, Any]] = []
    secs = report.get("sections", {})

    def sec(name: str) -> Optional[Dict[str, Any]]:
        s = secs.get(name)
        if s is None:
            return None
        if "error" in s and len(s) == 1:
            rows.append({"section": name, "check": "section ran", "ours": s["error"], "paper": None,
                         "tol": None, "mode": "", "kind": "anchor", "unit": "", "status": "FAIL"})
            return None
        return s

    s = sec("fig5")
    if s:
        _check(rows, "fig5", "P_U,opt at T_pre3 = 8 K", s.get("P_U_opt_MPa"), 0.36, TOL["P_U_opt_MPa"], "abs", unit="MPa")
        _check(rows, "fig5", "q_C,opt", s.get("q_C_opt_kJ_kg"), 8.0, TOL["q_C_opt_kJ_kg"], "abs", unit="kJ/kg")
        _check(rows, "fig5", "P_U max (q_L = 0)", s.get("P_U_max_MPa"), 1.13, TOL["P_U_max_MPa"], "abs", unit="MPa")
        _check(rows, "fig5", "q_C = min(dh_T, q_L) to within", s.get("max_rel_dev_qC_vs_min_deltahT_qL"), None,
               TOL["qC_vs_min_rel"], "max", kind="consistency")
        _check(rows, "fig5", "optimum is inside the search range", s.get("optimum_at_search_bound"), None, 0, "false",
               kind="consistency")
    s = sec("fits")
    if s:
        _check(rows, "fits", "dh_T = q_L at optimum (Eq. 34), max rel gap", s.get("max_rel_gap_deltahT_vs_qL_at_optimum"),
               None, TOL["gap_dhT_qL_rel"], "max", kind="consistency")
        _check(rows, "fits", "P_U,opt vs Eq. 36, mean error", (s.get("P_U_opt_vs_Eq36") or {}).get("mape_percent"),
               None, TOL["eq36_mape_pct"], "max", unit="%")
        _check(rows, "fits", "P_U,opt vs Eq. 36, worst error", (s.get("P_U_opt_vs_Eq36") or {}).get("max_rel_percent"),
               None, TOL["eq36_max_pct"], "max", unit="%")
        _check(rows, "fits", "q_C,opt vs Eq. 35+36, mean error", (s.get("q_C_opt_vs_Eq35_36_chain") or {}).get("mape_percent"),
               None, TOL["eq35_mape_pct"], "max", unit="%")
        _check(rows, "fits", "optima at the search bound (count)", len(s.get("at_search_bound") or []), None, 0, "zero",
               kind="consistency")
    s = sec("fig7")
    if s:
        for e in s.get("per_P_U", []):
            _check(rows, "fig7", f"T_pre3,opt at P_U = {e.get('P_U_MPa'):g} MPa vs Eq. 36", e.get("difference_K"), 0.0,
                   TOL["tpre_opt_K"], "abs", unit="K")
    s = sec("flux")
    if s:
        for e in s.get("per_T_C", []):
            tc = e.get("T_C_K")
            if "HEM_vs_paper_curve" in e:
                _check(rows, "flux", f"HEM G vs Table 2 at T_C = {tc:g} K, mean error",
                       e["HEM_vs_paper_curve"].get("mape_percent"), None, TOL["flux_vs_paper_mape_pct"], "max",
                       kind="model", unit="%")
            _check(rows, "flux", f"Eq. 49 form fits HEM data at T_C = {tc:g} K, worst error",
                   (e.get("HEM_refit") or {}).get("fit_vs_HEM_points", {}).get("max_rel_percent"), None,
                   TOL["flux_refit_max_pct"], "max", kind="consistency", unit="%")
    s = sec("fig15")
    if s:
        p = s.get("paper", {})
        _check(rows, "fig15", "q_C at eta = 1", s.get("q_C_eta1_kJ_kg"), p.get("q_C_opt_kJ_kg", 8.0), TOL["q_C_opt_kJ_kg"],
               "abs", unit="kJ/kg")
        _check(rows, "fig15", "Q_C at eta = 1", s.get("Q_C_eta1_mW"), p.get("Q_C_ideal_mW", 18.7), TOL["Q_C_mW"], "abs",
               kind="model", unit="mW")
        _check(rows, "fig15", "mdot at eta = 1", s.get("mdot_eta1_mg_s"), p.get("mdot_mg_s", 2.34), TOL["mdot_mg_s"], "abs",
               kind="model", unit="mg/s")
        _check(rows, "fig15", "Q_C at eta = 0.97", s.get("Q_C_eta097_mW"), p.get("Q_C_eta097_mW", 14.7), TOL["Q_C_mW"], "abs",
               kind="model", unit="mW")
        _check(rows, "fig15", "Q_C drop at eta = 0.97", s.get("Q_C_drop_097_percent"), p.get("Q_C_drop_097_percent", 21.4),
               TOL["drop_pct_points"], "abs", unit="% points")
    return rows


def _print_scorecard(rows: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int]:
    print(f"  {'':4} {'check':<58} {'ours':>10} {'paper':>9}  tolerance")
    for r in rows:
        if r["mode"] in ("abs", "pct"):
            tol = f"+/- {_num(r['tol'])} {r['unit']}"
        elif r["mode"] == "max":
            tol = f"<= {_num(r['tol'])} {r['unit']}"
        elif r["mode"] == "zero":
            tol = "must be 0"
        elif r["mode"] == "false":
            tol = "must be no"
        else:
            tol = ""
        tag = " *" if r["kind"] == "model" else ("  " if r["kind"] == "anchor" else " c")
        print(f"  {r['status']:<4} {str(r['check'])[:58]:<58} {_num(r['ours']):>10} {_num(r['paper']):>9}  {tol}{tag}")
    n_fail = sum(r["status"] == "FAIL" for r in rows)
    n_warn = sum(r["status"] == "WARN" for r in rows)
    n_pass = sum(r["status"] == "PASS" for r in rows)
    print("  (c = consistency check of our own implementation;  * = model-dependent, a miss is only a WARN)")
    return n_pass, n_warn, n_fail


def _run(cmd: List[str], label: str) -> bool:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
    print(f"  {'PASS' if proc.returncode == 0 else 'FAIL'}  {label}  ({tail[0].strip()})")
    if proc.returncode != 0:
        print((proc.stdout + proc.stderr)[-1500:])
    return proc.returncode == 0


def cmd_validate(rest: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="jt_run.py validate",
        description="Workflow 1: check data, run tests, reproduce the paper, print a scorecard. "
                    "Unrecognised options are passed on to reproduce_zhang_dang_2021.py.")
    ap.add_argument("--fluid", default="he3", help="'he3' (default) or a CoolProp name for a machinery test")
    ap.add_argument("--outdir", default="results/validation")
    ap.add_argument("--quick", action="store_true", help="coarser grids (minutes -> seconds)")
    ap.add_argument("--skip-tests", action="store_true")
    a, extra = ap.parse_known_args(rest)
    he3 = _is_he3(a.fluid)

    print("[1/4] property data")
    if he3:
        if not _print_checks(check_he3()):
            print("Stopped: fix the data files first.")
            return 1
    else:
        print("  INFO  skipped (not He-3)")
    print("[2/4] self-tests")
    if a.skip_tests:
        print("  INFO  skipped")
    else:
        ok = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], "unit tests (tests/)")
        ok = _run([sys.executable, str(ROOT / "reproduce_zhang_dang_2021.py"), "--self-test"], "fit helpers") and ok
        if not ok:
            print("Stopped: a self-test failed.")
            return 1
    print("[3/4] reproduction")
    import reproduce_zhang_dang_2021 as rz
    args = ["--fluid", a.fluid, "--outdir", a.outdir] + (["--quick"] if a.quick else []) + extra
    rz.main(args)
    report_path = Path(a.outdir) / "reproduction_report.json"   # same base folder as the reproduction script
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("[4/4] scorecard")
    if not report.get("paper_comparison_meaningful"):
        print(f"  INFO  {report.get('note')}  No scorecard.")
        return 0
    rows = build_scorecard(report)
    n_pass, n_warn, n_fail = _print_scorecard(rows)
    (report_path.parent / "scorecard.json").write_text(json.dumps(_clean(rows), indent=2), encoding="utf-8")
    print(f"  {n_pass} pass, {n_warn} warn, {n_fail} fail   (tolerances are provisional: see TOL in jt_run.py)")
    print(f"  details: {report_path.parent}")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# solve : workflows 2 and 3

CASE_KEYS = {"name", "fluid", "singlephase_csv", "saturation_csv", "T_C", "P_U", "T_H", "T_pre", "eta",
             "d_um", "sweep", "reported", "source", "notes"}
SWEEP_KEYS = {"T_C", "P_U", "T_H", "d_um", "eta_last", "T_pre_last"}


def load_case(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("a case file must contain a JSON object")
    case = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = set(case) - CASE_KEYS
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)}; allowed: {sorted(CASE_KEYS)}")
    for k in ("fluid", "T_C", "P_U", "T_pre", "eta"):
        if k not in case:
            raise ValueError(f"missing required key '{k}'")
    case.setdefault("name", path.stem)
    case.setdefault("T_H", 300.0)
    case.setdefault("d_um", None)
    if not isinstance(case["eta"], list) or not case["eta"]:
        raise ValueError("'eta' must be a non-empty list, one effectiveness per CHEX (warm to cold)")
    if not isinstance(case["T_pre"], list) or len(case["T_pre"]) != len(case["eta"]) - 1:
        raise ValueError(f"'T_pre' must list {len(case['eta']) - 1} temperatures (one per PHEX) "
                         f"for {len(case['eta'])} CHEXs; got {case['T_pre']!r}")
    for k in ("T_C", "P_U", "T_H"):
        if not isinstance(case[k], (int, float)) or case[k] <= 0:
            raise ValueError(f"'{k}' must be a positive number")
    sweep = {k: v for k, v in (case.get("sweep") or {}).items() if not k.startswith("_")}
    case["sweep"] = sweep
    case["reported"] = {k: v for k, v in (case.get("reported") or {}).items() if not k.startswith("_")}
    bad = set(sweep) - SWEEP_KEYS
    if bad:
        raise ValueError(f"cannot sweep {sorted(bad)}; sweepable: {sorted(SWEEP_KEYS)}")
    for k, v in sweep.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"sweep['{k}'] must be a non-empty list")
    return case


def _apply(base: Mapping[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    p = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
    for k, v in over.items():
        if k == "eta_last":
            p["eta"][-1] = v
        elif k == "T_pre_last":
            if not p["T_pre"]:
                raise ValueError("T_pre_last needs at least one PHEX")
            p["T_pre"][-1] = v
        else:
            p[k] = v
    return p


def make_case_fluid(case: Mapping[str, Any]):
    from jt.cycle_core import make_fluid
    kw = {}
    if case.get("singlephase_csv"):
        kw["singlephase_csv"] = str(case["singlephase_csv"])
    if case.get("saturation_csv"):
        kw["saturation_csv"] = str(case["saturation_csv"])
    return make_fluid(str(case["fluid"]), **kw)


def solve_point(fluid, p: Mapping[str, Any], pinch: bool = True):
    from jt.cycle_core import solve_cycle
    d = None if p.get("d_um") is None else float(p["d_um"]) * 1e-6
    return solve_cycle(fluid, Tc=float(p["T_C"]), Pu=float(p["P_U"]) * 1e6, Th=float(p["T_H"]),
                       Tpre=[float(t) for t in p["T_pre"]], eta=[float(e) for e in p["eta"]], d=d,
                       pinch_points=20 if pinch else 0)


def cycle_metrics(res, p: Mapping[str, Any]) -> Dict[str, Any]:
    n = len(p["eta"])
    m: Dict[str, Any] = {
        "q_C_kJ_kg": res.q_C / 1e3,
        "T_JT_inlet_K": res.states[2 * n].T,
        "P_D_kPa": res.P_D / 1e3,
        "energy_residual": res.energy_residual,
        "chex_limiting_end": ",".join(res.chex_binding),
    }
    for i, q in enumerate(res.q_pre, 1):
        m[f"q_pre{i}_kJ_kg"] = q / 1e3
    if res.mdot is not None:
        m["mdot_mg_s"] = res.mdot * 1e6
        m["Q_C_mW"] = res.Q_C * 1e3
        for i, Q in enumerate(res.Q_pre or [], 1):
            m[f"Q_pre{i}_mW"] = Q * 1e3
        if res.throat:
            m["G_kg_m2_s"] = res.throat.get("G")
            m["throat_choked"] = res.throat.get("choked")
    if res.x_l_JT_outlet is not None:
        m["liquid_fraction_after_JT"] = res.x_l_JT_outlet
    return m


def state_table(res, n: int) -> List[Dict[str, Any]]:
    from jt.cycle_core import state_labels
    lab = state_labels(n)
    role: Dict[int, List[str]] = {}
    for i in range(n):
        role.setdefault(lab["A"][i], []).append(f"CHEX{i + 1} hot in")
        role.setdefault(lab["B"][i], []).append(f"CHEX{i + 1} hot out")
        role.setdefault(lab["C"][i], []).append(f"CHEX{i + 1} cold in")
        role.setdefault(lab["D"][i], []).append(f"CHEX{i + 1} cold out")
    role.setdefault(lab["JT_out"], []).append("JT valve outlet")
    role.setdefault(lab["evap_out"], []).append("evaporator outlet")
    rows = []
    for k in sorted(res.states):
        s = res.states[k]
        rows.append({"state": k, "role": "; ".join(role.get(k, [])), "P_Pa": s.P, "T_K": s.T,
                     "h_J_kg": s.h, "x_vapour": s.x_v, "phase": s.phase})
    return rows


def compare_reported(metrics: Mapping[str, Any], reported: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for k, ref in reported.items():
        if k.startswith("_"):
            continue
        ours = metrics.get(k)
        row: Dict[str, Any] = {"metric": k, "ours": ours, "reported": ref}
        if isinstance(ours, (int, float)) and not isinstance(ours, bool) and isinstance(ref, (int, float)):
            row["difference"] = ours - ref
            row["difference_percent"] = 100.0 * (ours - ref) / ref if ref else None
        else:
            row["note"] = "not a metric this solver produces" if ours is None else ""
        rows.append(row)
    return rows


def _provenance(case: Mapping[str, Any]) -> Dict[str, Any]:
    prov: Dict[str, Any] = {"date": dt.datetime.now().isoformat(timespec="seconds"), "fluid": case["fluid"]}
    if _is_he3(case["fluid"]):
        sp = Path(case.get("singlephase_csv") or DATA_HE3 / "SinglePhase_He_3.csv")
        sat = Path(case.get("saturation_csv") or DATA_HE3 / "Saturation_He_3.csv")
        prov["he3_tables"] = {str(p): (_sha(p) if p.is_file() else "missing") for p in (sp, sat)}
    return prov


def run_case(path: Path, outroot: Path, pinch: bool) -> bool:
    case = load_case(path)
    fluid = make_case_fluid(case)
    base = {k: case[k] for k in ("T_C", "P_U", "T_H", "T_pre", "eta", "d_um")}
    out = outroot / _slug(case["name"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.json").write_text(path.read_text(encoding="utf-8-sig"), encoding="utf-8")
    print(f"\nCase: {case['name']}   [{path.name}]   fluid: {case['fluid']}")
    if case.get("source"):
        print(f"  source: {case['source']}")
    result: Dict[str, Any] = {"case": case, "provenance": _provenance(case)}
    ok = True

    # base point
    try:
        res = solve_point(fluid, base, pinch)
        met = cycle_metrics(res, base)
        _write_csv(out / "states.csv", state_table(res, len(base["eta"])))
        result.update({"metrics": met, "flags": res.flags})
        n = len(base["eta"])
        d = base["d_um"]
        print(f"  inputs : T_C={base['T_C']:g} K  P_U={base['P_U']:g} MPa  T_H={base['T_H']:g} K  "
              f"T_pre={base['T_pre']}  eta={base['eta']}  d={'-' if d is None else f'{d:g} um'}")
        print(f"  result : q_C={_num(met['q_C_kJ_kg'])} kJ/kg   T before JT valve={_num(met['T_JT_inlet_K'])} K   "
              f"P_D={_num(met['P_D_kPa'])} kPa")
        if "Q_C_mW" in met:
            print(f"           mdot={_num(met['mdot_mg_s'])} mg/s   Q_C={_num(met['Q_C_mW'])} mW   "
                  f"PHEX loads (warm->cold) [mW]: {[round(met.get(f'Q_pre{i}_mW', float('nan')), 3) for i in range(1, n)]}")
        else:
            print("           (no orifice diameter given: specific duties only; set d_um for mdot and Q_C)")
        print(f"           limiting end of each CHEX (warm->cold): {met['chex_limiting_end']}")
        print(f"  flags  : {'; '.join(res.flags) if res.flags else 'none'}")
        if case.get("reported"):
            cmp_rows = compare_reported(met, case["reported"])
            result["comparison"] = cmp_rows
            print("  vs reported:")
            for r in cmp_rows:
                if "difference" in r:
                    pct = "" if r["difference_percent"] is None else f" ({r['difference_percent']:+.1f} %)"
                    print(f"    {r['metric']:<18} ours {_num(r['ours']):>9}   reported {_num(r['reported']):>9}   "
                          f"diff {r['difference']:+.4g}{pct}")
                else:
                    print(f"    {r['metric']:<18} {r.get('note') or 'no value'}")
    except Exception as exc:
        ok = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED : {result['error']}")

    # optional sweep
    sweep = case.get("sweep") or {}
    if sweep:
        keys = list(sweep)
        rows = []
        for combo in itertools.product(*(sweep[k] for k in keys)):
            over = dict(zip(keys, combo))
            row: Dict[str, Any] = dict(over)
            try:
                p = _apply(base, over)
                r = solve_point(fluid, p, pinch)
                row.update(cycle_metrics(r, p))
                row["n_flags"] = len(r.flags)
                row["flags"] = "; ".join(r.flags)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        _write_csv(out / "sweep.csv", rows)
        n_err = sum("error" in r for r in rows)
        print(f"  sweep  : {len(rows)} points over {keys}  ({n_err} failed)  -> sweep.csv")
    (out / "result.json").write_text(json.dumps(_clean(result), indent=2, default=str), encoding="utf-8")
    print(f"  saved  : {out}")
    return ok


def cmd_solve(rest: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="jt_run.py solve", description="Solve one or more case files.")
    ap.add_argument("cases", nargs="+", help="case JSON file(s); wildcards such as cases/*.json are expanded here")
    ap.add_argument("--outdir", default="results/cases")
    ap.add_argument("--no-pinch", action="store_true", help="skip the internal-pinch profile check")
    a = ap.parse_args(rest)
    paths: List[Path] = []
    for pat in a.cases:
        hits = sorted(glob.glob(pat))
        paths += [Path(h) for h in hits] if hits else [Path(pat)]
    paths = [p for p in paths if not p.name.startswith("_")]   # skip _template.json
    if not paths:
        print("No case files found.")
        return 1
    n_bad = 0
    for p in paths:
        try:
            if not run_case(p, Path(a.outdir), pinch=not a.no_pinch):
                n_bad += 1
        except Exception as exc:
            n_bad += 1
            hint = "  (run: python jt_run.py check)" if isinstance(exc, FileNotFoundError) else ""
            print(f"\nCase file {p}: {type(exc).__name__}: {exc}{hint}")
    print(f"\n{len(paths) - n_bad} of {len(paths)} case(s) solved.")
    return 1 if n_bad else 0


# ---------------------------------------------------------------------------
# passthroughs and dispatch

def cmd_optimize(rest: List[str]) -> int:
    from jt import design_optimize
    return int(design_optimize.main(rest) or 0)


def cmd_phex(rest: List[str]) -> int:
    from jt import phex_load_map
    return int(phex_load_map.main(rest) or 0)


COMMANDS = {"check": cmd_check, "validate": cmd_validate, "solve": cmd_solve,
            "optimize": cmd_optimize, "phex": cmd_phex}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'.\n{__doc__}")
        return 2
    return COMMANDS[cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())
