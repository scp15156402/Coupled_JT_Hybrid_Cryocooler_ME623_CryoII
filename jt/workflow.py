"""Design workflow behind `jt_run.py new / feasible / run / runs`.

  configs     normalise / name / save a configuration  (cases/<id>.json)
  fluids      He-3 (CSV) and CoolProp cryogens, with sensible default ranges
  feasibility check_config(): can this configuration run, and is it sensible?
  solve       run_solve(): state points (T, P, h, phase), PHEX loads, P-h plot
  optimize    run_optimize(): last-stage optimum for q_C and Q_C, T_pre,opt, then the upstream
              precooler temperatures (criterion b: minimise the Carnot-weighted load
              sum_j Q_pre,j (T_H/T_j - 1)), plus a sensitivity map per upstream PHEX
  reference   per-configuration reference (citation, link, location in source, notes), kept with every run
  runs        every run is written to results/runs/<config id>/<stamp>_<kind>/ with a copy of the
              configuration and its hash, so a run can always be traced to its configuration

Only uses the existing solver (jt.cycle_core) and last-stage optimizer (jt.design_optimize).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from jt import design_optimize as do
from jt.cycle_core import make_fluid, solve_cycle, state_labels

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cases"
RUNS = ROOT / "results" / "runs"

# label, backend name for make_fluid
FLUIDS = [
    ("He-3 (CSV tables)", "he3"), ("He-4", "Helium"), ("Neon", "Neon"), ("Hydrogen", "Hydrogen"),
    ("Parahydrogen", "ParaHydrogen"), ("Nitrogen", "Nitrogen"), ("Argon", "Argon"),
    ("Oxygen", "Oxygen"), ("Methane", "Methane"),
]
# typical starting values only; used for optimiser ranges
DEFAULTS = {
    "he3": dict(T_C=1.0, P_U=0.36, T_pre=[80, 25, 8], pu=(0.2, 1.2), tpre=(1.2, 30.0)),
    "Helium": dict(T_C=4.2, P_U=0.8, T_pre=[80, 25, 12], pu=(0.2, 2.0), tpre=(5.0, 40.0)),
    "Neon": dict(T_C=27.0, P_U=2.0, T_pre=[150, 60], pu=(0.5, 6.0), tpre=(35.0, 150.0)),
    "Hydrogen": dict(T_C=20.0, P_U=2.0, T_pre=[100, 40], pu=(0.5, 8.0), tpre=(25.0, 120.0)),
    "ParaHydrogen": dict(T_C=20.0, P_U=2.0, T_pre=[100, 40], pu=(0.5, 8.0), tpre=(25.0, 120.0)),
    "Nitrogen": dict(T_C=80.0, P_U=10.0, T_pre=[200, 130], pu=(1.0, 20.0), tpre=(100.0, 250.0)),
    "Argon": dict(T_C=90.0, P_U=10.0, T_pre=[200, 140], pu=(1.0, 20.0), tpre=(110.0, 260.0)),
    "Oxygen": dict(T_C=90.0, P_U=10.0, T_pre=[200, 140], pu=(1.0, 20.0), tpre=(110.0, 260.0)),
    "Methane": dict(T_C=115.0, P_U=10.0, T_pre=[220, 160], pu=(1.0, 20.0), tpre=(130.0, 270.0)),
}
_cache: Dict[str, Any] = {}


# ------------------------------------------------------------------ fluids and limits
def get_fluid(name: str):
    key = name.strip()
    if key not in _cache:
        _cache[key] = make_fluid(key)
    return _cache[key]


def is_he3(name: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", name.lower()) in {"he3", "helium3"}


def fluid_limits(name: str, fluid) -> Dict[str, Any]:
    """Cold-end temperature window the property backend supports."""
    if is_he3(name):
        from jt.properties import MAX_COLD_END_T
        return dict(T_min=float(fluid.Tmin_sat), T_max=float(MAX_COLD_END_T), T_crit=float(fluid.Tcrit))
    from CoolProp.CoolProp import PropsSI
    out: Dict[str, Any] = {}
    for key, fn in (("T_crit", "Tcrit"), ("T_min", "Ttriple")):
        try:
            out[key] = float(PropsSI(fn, name))
        except Exception:
            out[key] = None
    if out["T_min"] is None:
        try:
            out["T_min"] = float(PropsSI("Tmin", name))
        except Exception:
            out["T_min"] = 0.0
    out["T_max"] = out["T_crit"] * 0.995 if out["T_crit"] else None
    return out


def default_ranges(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    key = "he3" if is_he3(name) else name.strip()
    d = DEFAULTS.get(key) or DEFAULTS["Nitrogen"]
    tc = float(cfg.get("T_C") or d["T_C"])
    return dict(pu_min=d["pu"][0], pu_max=d["pu"][1],
                tpre_min=max(d["tpre"][0], round(1.2 * tc, 2)), tpre_max=d["tpre"][1])


# ------------------------------------------------------------------ configuration
def norm_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the shape of a configuration. T_pre entries may be None (= to be optimised)."""
    def num(v, what, allow_none=False):
        if v is None or v == "":
            if allow_none:
                return None
            raise ValueError(f"{what} is required")
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{what} must be a number, got {v!r}")
        if not math.isfinite(x):
            raise ValueError(f"{what} must be finite")
        return x

    fluid = str(raw.get("fluid", "")).strip()
    if not fluid:
        raise ValueError("fluid is required")
    eta = [num(e, "eta") for e in (raw.get("eta") or [])]
    if not eta:
        raise ValueError("give at least one CHEX effectiveness (one per CHEX, warm to cold)")
    if any(not 0.0 <= e <= 1.0 for e in eta):
        raise ValueError("every eta must lie between 0 and 1")
    t_pre = [num(t, "T_pre", True) for t in (raw.get("T_pre") or [])]
    if len(t_pre) != len(eta) - 1:
        raise ValueError(f"{len(eta)} CHEXs need {len(eta) - 1} precooling temperatures (T_pre); "
                         f"got {len(t_pre)}. Leave a blank to have it optimised.")
    cfg = dict(name=str(raw.get("name") or "").strip(), fluid=fluid,
               T_C=num(raw.get("T_C"), "T_C"), P_U=num(raw.get("P_U"), "P_U", True),
               T_H=num(raw.get("T_H", 300.0), "T_H"), T_pre=t_pre, eta=eta,
               d_um=num(raw.get("d_um"), "d_um", True), mdot_mg_s=num(raw.get("mdot_mg_s"), "mdot_mg_s", True))
    for k in ("T_C", "T_H"):
        if cfg[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if cfg["P_U"] is not None and cfg["P_U"] <= 0:
        raise ValueError("P_U must be positive")
    if cfg["d_um"] is not None and cfg["d_um"] <= 0:
        raise ValueError("d_um must be positive")
    if cfg["mdot_mg_s"] is not None:
        if cfg["mdot_mg_s"] <= 0:
            raise ValueError("mdot_mg_s must be positive")
        if cfg["d_um"] is not None:
            raise ValueError("give either d_um (flow from the orifice model) or mdot_mg_s (measured flow), not both")
    else:
        del cfg["mdot_mg_s"]
    for k in ("source", "notes"):
        if raw.get(k):
            cfg[k] = str(raw[k])
    # Where these inputs come from (paper, table, measurement, own design). Not part of the configuration hash,
    # so adding or correcting a reference later never breaks the link between a run and its configuration.
    ref = {k: str(v).strip() for k, v in (raw.get("reference") or {}).items()
           if not str(k).startswith("_") and str(v).strip()}
    if ref:
        cfg["reference"] = ref
        if ref.get("citation") and not cfg.get("source"):
            cfg["source"] = ref["citation"]
    opt = {k: v for k, v in (raw.get("optimize") or {}).items() if not str(k).startswith("_")}
    if opt:
        cfg["optimize"] = opt
    return cfg


def config_id(cfg: Dict[str, Any]) -> str:
    """{Fluid}_{T_C}K_{d}um_{N}_stage  (N = number of CHEXs), + _eta<x> if the last CHEX is not ideal."""
    # CoolProp's "Helium" is He-4: label it as such so it can never be mistaken for the He-3 runs.
    fl = "Helium_4" if cfg["fluid"].strip().lower() == "helium" else do._fluid_filename_label(cfg["fluid"], None)
    if cfg.get("d_um") is not None:
        d = do._format_filename_number(cfg["d_um"]) + "um"
    elif cfg.get("mdot_mg_s") is not None:
        d = do._format_filename_number(cfg["mdot_mg_s"]) + "mgs"
    else:
        d = "nod"
    stem = f"{fl}_{do._format_filename_number(cfg['T_C'])}K_{d}_{len(cfg['eta'])}_stage"
    if abs(cfg["eta"][-1] - 1.0) > 1e-12:
        stem += f"_eta{do._format_filename_number(cfg['eta'][-1])}"
    return stem


def config_hash(cfg: Dict[str, Any]) -> str:
    keys = ("fluid", "T_C", "P_U", "T_H", "T_pre", "eta", "d_um", "mdot_mg_s", "optimize")
    blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _file_hash(path: Path) -> Optional[str]:
    try:
        return config_hash(norm_cfg(json.loads(path.read_text(encoding="utf-8-sig"))))
    except Exception:
        return None


def save_config(cfg: Dict[str, Any], overwrite_id: Optional[str] = None) -> Dict[str, Any]:
    """Write cases/<id>.json. Same content -> same file. Different content under the same id -> _v2, _v3 ..."""
    CASES.mkdir(exist_ok=True)
    base = overwrite_id or config_id(cfg)
    sha = config_hash(cfg)
    cid, k = base, 1
    while (CASES / f"{cid}.json").exists() and _file_hash(CASES / f"{cid}.json") != sha and not overwrite_id:
        k += 1
        cid = f"{base}_v{k}"
    out = dict(cfg)
    out["name"] = cfg.get("name") or cid
    (CASES / f"{cid}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return {"id": cid, "file": f"{cid}.json", "sha": sha}


# ------------------------------------------------------------------ run storage
def new_run(cfg: Dict[str, Any], kind: str) -> Path:
    cid = config_id(cfg)
    run = RUNS / cid / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{kind}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return run


def finish_run(run: Path, cfg: Dict[str, Any], kind: str, result: Dict[str, Any], headline: str) -> None:
    files = sorted(p.name for p in run.iterdir() if p.suffix in (".png", ".csv"))
    meta = dict(config_id=config_id(cfg), config_sha=config_hash(cfg), kind=kind,
                time=dt.datetime.now().isoformat(timespec="seconds"), headline=headline, files=files,
                reference=cfg.get("reference") or ({"citation": cfg["source"]} if cfg.get("source") else {}))
    (run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    result = dict(result, reference=meta["reference"])
    (run / "result.json").write_text(json.dumps(_clean(result), indent=2), encoding="utf-8")


def list_runs() -> List[Dict[str, Any]]:
    rows = []
    for meta in sorted(RUNS.glob("*/*/meta.json"), reverse=True):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        cf = CASES / f"{m['config_id']}.json"
        linked = [p.name for p in CASES.glob(f"{m['config_id']}*.json") if _file_hash(p) == m["config_sha"]]
        m.update(run=f"{meta.parent.parent.name}/{meta.parent.name}", config_files=linked,
                 config_saved=bool(linked))
        rows.append(m)
    return sorted(rows, key=lambda r: r["time"], reverse=True)


def delete_run(run: str) -> None:
    p = (RUNS / run).resolve()
    if RUNS.resolve() not in p.parents or not p.is_dir():
        raise ValueError("not a run folder")
    shutil.rmtree(p)
    if not any(p.parent.iterdir()):
        p.parent.rmdir()


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ------------------------------------------------------------------ feasibility
def _classify(res, n: int) -> Dict[str, List[str]]:
    """Split solver flags into errors (result is not physical) and warnings (result is usable, look at it)."""
    errors: List[str] = []
    warns: List[str] = []
    for j, q in enumerate(res.q_pre, 1):
        if q < -1e-3 * max(abs(res.q_C), 1.0):
            errors.append(f"PHEX {j} would have to HEAT the stream (load {q / 1e3:.3f} kJ/kg < 0): "
                          f"T_pre,{j} is warmer than the stream leaving CHEX {j}. Lower T_pre,{j}.")
    for f in res.flags:
        head = f.split(":")[0]
        if head == "NONPOSITIVE_REFRIGERATION":
            errors.append("no net refrigeration (q_C <= 0)")
        elif head == "GLOBAL_ENERGY_BALANCE":
            errors.append("global energy balance does not close: " + f)
        elif head == "NEGATIVE_PHEX_LOAD":
            continue                                    # handled above with a tolerance
        elif head == "UNCHOKED_RESTRICTION_FLOW":
            warns.append("orifice flow is not choked (HEM maximum at the back pressure); "
                         "typical for liquid-like inlets, mass flux is still the HEM maximum")
        else:
            warns.append(f)
    return {"errors": errors, "warnings": warns}


def check_config(cfg: Dict[str, Any], fluid=None) -> Dict[str, Any]:
    """Can this configuration run, and does it make physical sense?  Never raises."""
    errors: List[str] = []
    warns: List[str] = []
    info: Dict[str, Any] = {}
    try:
        fluid = fluid or get_fluid(cfg["fluid"])
    except Exception as exc:
        return dict(verdict="infeasible", errors=[f"cannot load fluid '{cfg['fluid']}': {exc}"], warnings=[], info={})
    n, tc, th = len(cfg["eta"]), cfg["T_C"], cfg["T_H"]
    lim = fluid_limits(cfg["fluid"], fluid)
    info["limits"] = lim
    if lim.get("T_min") is not None and tc < lim["T_min"]:
        errors.append(f"T_C = {tc:g} K is below the lowest supported temperature {lim['T_min']:g} K")
    if lim.get("T_max") is not None and tc > lim["T_max"]:
        errors.append(f"T_C = {tc:g} K is above the highest supported cold-end temperature {lim['T_max']:g} K "
                      f"(critical temperature {lim.get('T_crit') or float('nan'):g} K)")
    if errors:
        return dict(verdict="infeasible", errors=errors, warnings=warns, info=info)
    try:
        pd_pa = float(fluid.Psat(tc))
        info["P_D_kPa"] = pd_pa / 1e3
    except Exception as exc:
        return dict(verdict="infeasible", errors=[f"cannot get the saturation pressure at T_C: {exc}"], warnings=[], info=info)

    known = [t for t in cfg["T_pre"] if t is not None]
    chain = [th] + [t for t in cfg["T_pre"] if t is not None]
    if any(b >= a for a, b in zip(chain, chain[1:])):
        errors.append("precooling temperatures must decrease from T_H to the last PHEX: "
                      f"T_H = {th:g} > T_pre,1 > ... > T_pre,{n - 1}")
    if known and min(known) <= tc:
        errors.append("every T_pre must be above T_C")
    pu = cfg.get("P_U")
    if pu is None:
        warns.append("P_U is blank: only the optimizer can run")
    else:
        pu_pa = pu * 1e6
        if pu_pa <= pd_pa:
            errors.append(f"P_U = {pu:g} MPa must be above the evaporator pressure P_D = {pd_pa / 1e6:.4g} MPa")
        else:
            try:
                q_l = float(fluid.h_sat_vapor(tc)) - float(fluid.h(pu_pa, tc))
                info["q_L_kJ_kg"] = q_l / 1e3
                if q_l <= 0:
                    pmax = do.find_pu_max(fluid, tc_k=tc, pu_hi_mpa=max(3.0, 3 * pu))
                    errors.append(f"q_L <= 0 at P_U = {pu:g} MPa: the stream cannot be cooled to saturated vapour "
                                  f"at T_C" + (f" (P_U must be below about {pmax:.3g} MPa)" if pmax else ""))
                else:
                    pm = do.find_pu_max(fluid, tc_k=tc, pu_hi_mpa=max(3.0, 3 * pu))
                    if pm:
                        info["P_U_max_MPa"] = pm
            except Exception as exc:
                errors.append(f"property lookup failed at (P_U, T_C): {type(exc).__name__}: {exc}")
        if not errors and cfg["T_pre"] and cfg["T_pre"][-1] is not None:
            tl = cfg["T_pre"][-1]
            try:
                dht = float(fluid.h(pd_pa, tl)) - float(fluid.h(pu_pa, tl))
                info["delta_h_T_kJ_kg"] = dht / 1e3
                if dht <= 0:
                    errors.append(f"delta h_T <= 0 at T_pre,last = {tl:g} K: the gas is above its inversion "
                                  f"temperature at this pressure, throttling does not cool it")
                t_opt = do.find_tpre_opt(fluid, tc_k=tc, pu_mpa=pu, tpre_max_k=min(th, max(40.0, 4 * tl)))
                if t_opt:
                    info["T_pre_opt_K"] = t_opt
                    if tl > t_opt * 1.001:
                        warns.append(f"T_pre,last = {tl:g} K is warmer than T_pre,opt = {t_opt:.3g} K for this P_U: "
                                     f"q_C is below its maximum (limited by delta h_T)")
            except Exception as exc:
                errors.append(f"property lookup failed at T_pre,last: {type(exc).__name__}: {exc}")
    if not errors and pu is not None and all(t is not None for t in cfg["T_pre"]):
        try:
            res = solve_cycle(fluid, Tc=tc, Pu=pu * 1e6, Th=th, Tpre=[float(t) for t in cfg["T_pre"]],
                              eta=cfg["eta"], d=None if cfg.get("d_um") is None else cfg["d_um"] * 1e-6,
                         mdot_given=None if cfg.get("mdot_mg_s") is None else cfg["mdot_mg_s"] * 1e-6)
            cl = _classify(res, n)
            errors += cl["errors"]
            warns += cl["warnings"]
            info["q_C_kJ_kg"] = res.q_C / 1e3
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    verdict = "infeasible" if errors else ("warnings" if warns else "feasible")
    return dict(verdict=verdict, errors=errors, warnings=warns, info=info)


# ------------------------------------------------------------------ solve
def _state_rows(res, n: int) -> List[Dict[str, Any]]:
    lab = state_labels(n)
    role: Dict[int, List[str]] = {}
    for i in range(n):
        for key, text in (("A", "hot in"), ("B", "hot out"), ("C", "cold in"), ("D", "cold out")):
            role.setdefault(lab[key][i], []).append(f"CHEX{i + 1} {text}")
    role.setdefault(lab["JT_out"], []).append("JT valve outlet")
    role.setdefault(lab["evap_out"], []).append("evaporator outlet")
    return [dict(state=k, role="; ".join(role.get(k, [])), T_K=s.T, P_kPa=s.P / 1e3, h_kJ_kg=s.h / 1e3,
                 phase=s.phase, x_vapour=s.x_v) for k, s in sorted(res.states.items())]


def _load_rows(res, n: int) -> List[Dict[str, Any]]:
    lab = state_labels(n)
    rows = []
    for j in range(n - 1):
        b, a = res.states[lab["B"][j]], res.states[lab["A"][j + 1]]
        q = b.h - a.h                                    # h_in - h_out
        rows.append(dict(phex=j + 1, T_in_K=b.T, T_out_K=a.T, q_kJ_kg=q / 1e3,
                         Q_mW=None if res.mdot is None else res.mdot * q * 1e3))
    return rows


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_ph(res, n: int, path: Path, title: str) -> None:
    plt = _plt()
    lab = state_labels(n)
    S = res.states
    hot = [S[lab["A"][0]]]
    for i in range(n):
        hot.append(S[lab["B"][i]])
        if i < n - 1:
            hot.append(S[lab["A"][i + 1]])
    cold = [S[lab["JT_out"]], S[lab["evap_out"]]] + [S[lab["D"][i]] for i in range(n - 1, -1, -1)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.plot([s.h / 1e3 for s in hot], [s.P / 1e3 for s in hot], "o-", color="#c0392b", label="high-pressure stream")
    ax.plot([hot[-1].h / 1e3, cold[0].h / 1e3], [hot[-1].P / 1e3, cold[0].P / 1e3], "--", color="gray", label="JT valve")
    ax.plot([s.h / 1e3 for s in cold], [s.P / 1e3 for s in cold], "s-", color="#1f5fbf", label="low-pressure stream")
    for k, s in S.items():
        ax.annotate(str(k), (s.h / 1e3, s.P / 1e3), textcoords="offset points", xytext=(4, 5), fontsize=8)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("specific enthalpy h  [kJ/kg]  (log scale)")
    ax.set_ylabel("pressure  [kPa]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _solve_point(fluid, cfg, T_pre=None, P_U=None, d_um="cfg"):
    d = cfg.get("d_um") if d_um == "cfg" else d_um
    return solve_cycle(fluid, Tc=cfg["T_C"], Pu=(P_U if P_U is not None else cfg["P_U"]) * 1e6, Th=cfg["T_H"],
                       Tpre=[float(t) for t in (T_pre if T_pre is not None else cfg["T_pre"])], eta=cfg["eta"],
                       d=None if d is None else d * 1e-6,
                       mdot_given=None if cfg.get("mdot_mg_s") is None else cfg["mdot_mg_s"] * 1e-6)


def run_solve(cfg: Dict[str, Any], log: Callable[[str], None] = print, run: Optional[Path] = None) -> Dict[str, Any]:
    fluid = get_fluid(cfg["fluid"])
    chk = check_config(cfg, fluid)
    out: Dict[str, Any] = dict(kind="solve", config=cfg, config_id=config_id(cfg), check=chk)
    if chk["verdict"] == "infeasible":
        out["headline"] = "infeasible: " + chk["errors"][0]
        return out
    n = len(cfg["eta"])
    res = _solve_point(fluid, cfg)
    out["metrics"] = dict(q_C_kJ_kg=res.q_C / 1e3, P_D_kPa=res.P_D / 1e3, T_JT_inlet_K=res.states[2 * n].T,
                          mdot_mg_s=None if res.mdot is None else res.mdot * 1e6,
                          Q_C_mW=None if res.Q_C is None else res.Q_C * 1e3,
                          G_kg_m2_s=None if not res.throat else res.throat.get("G"),
                          chex_limiting_end=list(res.chex_binding), flags=list(res.flags))
    out["states"] = _state_rows(res, n)
    out["loads"] = _load_rows(res, n)
    if run is not None:
        _plot_ph(res, n, run / "ph_cycle.png", f"{config_id(cfg)}: cycle on the P-h plane (numbers = state labels)")
        import csv
        with (run / "states.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out["states"][0]))
            w.writeheader()
            w.writerows(out["states"])
    m = out["metrics"]
    out["headline"] = f"q_C = {m['q_C_kJ_kg']:.3f} kJ/kg" + (f", Q_C = {m['Q_C_mW']:.2f} mW" if m["Q_C_mW"] else "")
    return out


# ------------------------------------------------------------------ optimizer pipeline
def _carnot_cost(q_pre: List[float], T_pre: List[float], T_H: float) -> float:
    return float(sum(q * (T_H / t - 1.0) for q, t in zip(q_pre, T_pre)))


def _upstream_eval(fluid, cfg, T_pre: List[float], P_U: float):
    """Specific PHEX loads for a trial T_pre vector, or None if it is infeasible. q_C, mdot are not affected."""
    try:
        res = solve_cycle(fluid, Tc=cfg["T_C"], Pu=P_U * 1e6, Th=cfg["T_H"], Tpre=[float(t) for t in T_pre],
                          eta=cfg["eta"], d=None, pinch_points=0)
    except Exception:
        return None
    if any(q < -1e-3 * max(abs(res.q_C), 1.0) for q in res.q_pre):
        return None
    return res


def optimise_upstream(fluid, cfg, P_U: float, T_last: float, log=print, seed: int = 1) -> Dict[str, Any]:
    """Criterion (b): choose the blank upstream T_pre minimising sum_j q_pre,j (T_H/T_j - 1).

    The last stage does not depend on them (its inlet states are fixed), so q_C, mdot and Q_C are unchanged;
    only how the fixed total load is shared between the precooler stages moves.
    """
    from scipy.optimize import differential_evolution
    n = len(cfg["eta"])
    th = cfg["T_H"]
    base = list(cfg["T_pre"])
    base[-1] = T_last
    free = [j for j in range(n - 2) if base[j] is None]
    if not free:
        return dict(T_pre=base, free=[], note="no blank upstream temperatures")
    fixed_pen = 1e9

    def assemble(x):
        T = list(base)
        for j, v in zip(free, x):
            T[j] = float(v)
        return T

    def cost(x):
        T = assemble(x)
        if any(T[i] is None for i in range(n - 1)):
            return fixed_pen
        if any(b >= a - 0.05 for a, b in zip([th] + T, T)):
            return fixed_pen
        res = _upstream_eval(fluid, cfg, T, P_U)
        if res is None:
            return fixed_pen
        return _carnot_cost(res.q_pre, T, th) / max(abs(res.q_C), 1.0)

    lo = T_last + 0.5
    r = differential_evolution(cost, [(lo, th - 0.5)] * len(free), seed=seed, popsize=14, maxiter=60, tol=1e-6, polish=False)
    log(f"upstream search: {r.nfev} evaluations, objective {r.fun:.4g}")
    T = assemble(r.x)
    return dict(T_pre=T, free=free, objective=float(r.fun), feasible=bool(r.fun < fixed_pen / 10))


def upstream_map(fluid, cfg, P_U: float, T_pre: List[float], free: List[int], n_pts: int = 41) -> List[Dict[str, Any]]:
    """For each upstream PHEX j: sweep T_pre,j between its neighbours, others fixed. One row per point."""
    th = cfg["T_H"]
    rows = []
    for j in free:
        hi = (T_pre[j - 1] if j > 0 else th) - 0.05
        lo = T_pre[j + 1] + 0.05
        for t in np.linspace(lo, hi, n_pts):
            T = list(T_pre)
            T[j] = float(t)
            res = _upstream_eval(fluid, cfg, T, P_U)
            row = dict(phex=j + 1, T_pre_K=float(t), feasible=res is not None)
            if res is not None:
                row["total_q_kJ_kg"] = sum(res.q_pre) / 1e3
                row["carnot_cost"] = _carnot_cost(res.q_pre, T, th) / 1e3
                for k, q in enumerate(res.q_pre, 1):
                    row[f"q{k}_kJ_kg"] = q / 1e3
                row["cold_limited_chex"] = sum(b != "warm_end" for b in res.chex_binding)
            rows.append(row)
    return rows


def _plot_map(rows, T_pre, free, n, path: Path, mdot: Optional[float]) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(free), figsize=(4.6 * len(free), 4.0), constrained_layout=True, squeeze=False)
    unit = "mW" if mdot else "kJ/kg"
    scale = (mdot * 1e6) if mdot else 1.0            # kJ/kg x kg/s = kW; x1e6 -> mW
    for ax, j in zip(axes[0], free):
        pts = [r for r in rows if r["phex"] == j + 1 and r["feasible"]]
        x = [r["T_pre_K"] for r in pts]
        for s in range(1, n):
            ax.plot(x, [r[f"q{s}_kJ_kg"] * scale for r in pts], lw=2 if s == j + 1 else 1,
                    label=f"PHEX {s} load" + (" (varied)" if s == j + 1 else ""))
        ax.plot(x, [r["total_q_kJ_kg"] * scale for r in pts], "k--", lw=1, label="total")
        ax.axvline(T_pre[j], color="gray", ls=":", label="chosen")
        ax.set_xlabel(f"T_pre,{j + 1}  [K]")
        ax.set_ylabel(f"load  [{unit}]")
        ax.set_title(f"vary PHEX {j + 1}, others fixed", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stage_curves(fluid, cfg, tl, p_marks, tpre_hi, path: Path, pu_rng) -> None:
    """Fig. 5 style (vs P_U at T_pre,last) and Fig. 7 style (vs T_pre at several P_U)."""
    plt = _plt()
    tc = cfg["T_C"]
    pus = np.linspace(pu_rng[0], pu_rng[1], 241)
    rows = do.stage_sweep(fluid, tc_k=tc, pu_values_mpa=pus, tpre_values_k=[tl], eta_last=cfg["eta"][-1], flow_model="none")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    f = lambda key: np.array([r.get(key, np.nan) if isinstance(r.get(key), (int, float)) else np.nan for r in rows], float)
    a1.plot(pus, f("delta_h_T_kj_kg"), "--", label=r"$\Delta h_T$")
    a1.plot(pus, f("q_L_kj_kg"), ":", label=r"$q_L$")
    a1.plot(pus, f("q_c_kj_kg"), "k-", lw=2, label=r"$q_C$")
    for name, p in p_marks.items():
        if p:
            a1.axvline(p, ls="-.", lw=0.9, label=f"{name}: {p:.3g} MPa")
    a1.set_xlabel(r"$P_U$  [MPa]")
    a1.set_ylabel("kJ/kg")
    a1.set_title(f"last stage at T_pre,last = {tl:.3g} K", fontsize=10)
    a1.set_ylim(bottom=0)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    ts = np.linspace(max(1.1 * tc, tc + 0.2), tpre_hi, 241)
    for name, p in p_marks.items():
        if not p:
            continue
        rr = do.stage_sweep(fluid, tc_k=tc, pu_values_mpa=[p], tpre_values_k=ts, eta_last=cfg["eta"][-1], flow_model="none")
        a2.plot(ts, [r["delta_h_T_kj_kg"] for r in rr], "--", lw=0.9)
        a2.plot(ts, [r["q_c_kj_kg"] for r in rr], "-", lw=1.8, label=f"{name}: {p:.3g} MPa")
    a2.set_xlabel("T_pre,last  [K]")
    a2.set_ylabel("kJ/kg   (dashed: $\\Delta h_T$,  solid: $q_C$)")
    a2.set_ylim(bottom=0)
    a2.set_title("effect of the precooling temperature", fontsize=10)
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_heat(rows, key, label, best, path: Path, title: str) -> None:
    plt = _plt()
    good = [r for r in rows if not r.get("error") and isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    if len(good) < 6:
        return
    x = np.array([r["tpre_last_k"] for r in good])
    y = np.array([r["pu_mpa"] for r in good])
    z = np.array([r[key] for r in good])
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    c = ax.tricontourf(x, y, z, levels=18)
    fig.colorbar(c, ax=ax, label=label)
    ax.plot(best["tpre_last_k"], best["pu_mpa"], "ko", mfc="none", mew=2, ms=11, label="maximum")
    ax.set_xlabel("T_pre,last  [K]")
    ax.set_ylabel(r"$P_U$  [MPa]")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _row_kv(r: Dict[str, Any]) -> Dict[str, Any]:
    return dict(P_U_MPa=r["pu_mpa"], T_pre_last_K=r["tpre_last_k"], q_C_kJ_kg=r["q_c_j_kg"] / 1e3,
                T_8_K=r["t8_k"], binding=r.get("binding_wall"),
                mdot_mg_s=None if not math.isfinite(r.get("mdot_kg_s", math.nan)) else r["mdot_kg_s"] * 1e6,
                Q_C_mW=None if not math.isfinite(r.get("q_c_w", math.nan)) else r["q_c_w"] * 1e3)


def run_optimize(cfg: Dict[str, Any], opts: Dict[str, Any], log: Callable[[str], None] = print,
                 run: Optional[Path] = None) -> Dict[str, Any]:
    """Optimise the last stage (P_U and, if blank, T_pre,last), then the blank upstream T_pre, then solve."""
    fluid = get_fluid(cfg["fluid"])
    n, tc = len(cfg["eta"]), cfg["T_C"]
    rng = default_ranges(cfg["fluid"], cfg)
    for k in rng:
        if opts.get(k) not in (None, ""):
            rng[k] = float(opts[k])
    he3 = is_he3(cfg["fluid"])
    flow = opts.get("flow_model") or "hem"
    if flow == "paper_fit" and not he3:
        flow = "hem"
    d_um = cfg.get("d_um")
    if d_um is None:
        d_um = 20.0
        log("no orifice diameter given: using 20 um for Q_C (q_C is independent of it)")
    d_m = d_um * 1e-6
    eta_last = cfg["eta"][-1]
    hem_grid = int(opts.get("hem_grid") or 120)
    drive = opts.get("drive") or "Q_C"
    out: Dict[str, Any] = dict(kind="optimize", config=cfg, config_id=config_id(cfg), ranges=rng, flow_model=flow,
                               drive=drive, d_um=d_um, notes=[])
    chk = check_config({**cfg, "P_U": cfg.get("P_U") or rng["pu_min"], "T_pre": [None if t is None else t for t in cfg["T_pre"]]}, fluid)
    if any("T_C" in e or "saturation" in e or "fluid" in e for e in chk["errors"]):
        out["check"] = chk
        out["headline"] = "infeasible: " + chk["errors"][0]
        return out

    tl_given = (cfg["T_pre"][-1] if n >= 2 else cfg["T_H"])
    if n == 1:
        out["notes"].append("one CHEX: the hot inlet is T_H, so only P_U is optimised")
    kw = dict(tc_k=tc, eta_last=eta_last, hem_grid=hem_grid, pu_min_mpa=rng["pu_min"], pu_max_mpa=rng["pu_max"])
    pu_max_theory = do.find_pu_max(fluid, tc_k=tc, pu_hi_mpa=max(rng["pu_max"] * 2, 3.0))
    out["P_U_max_MPa"] = pu_max_theory

    if tl_given is not None:
        # ---- T_pre,last set by the user: optimum P_U for q_C and for Q_C
        log(f"T_pre,last = {tl_given:g} K given: searching P_U")
        o_q = do.find_optimum_pu(fluid, tpre_last_k=tl_given, objective="q_c", flow_model="none", d_m=d_m, **kw)
        log("  max q_C done")
        o_Q = do.find_optimum_pu(fluid, tpre_last_k=tl_given, objective="Q_C", flow_model=flow, d_m=d_m, **kw)
        log("  max Q_C done")
        opt_q, opt_Q = o_q["row"], o_Q["row"]
        if not math.isfinite(opt_q.get("mdot_kg_s", math.nan)):
            opt_q = do._safe_evaluate(fluid, tc_k=tc, pu_mpa=o_q["pu_opt_mpa"], tpre_last_k=tl_given, d_m=d_m,
                                      eta_last=eta_last, flow_model=flow, hem_grid=hem_grid)
        out["optima"] = {"max_q_C": _row_kv(opt_q), "max_Q_C": _row_kv(opt_Q)}
        for nm, o in (("max_q_C", o_q), ("max_Q_C", o_Q)):
            if o["at_search_bound"]:
                out["notes"].append(f"{nm}: optimum sits at the P_U search bound - widen the range")
        chosen = opt_Q if drive == "Q_C" else opt_q
        P_U, T_L = chosen["pu_mpa"], tl_given
        marks = {"max q_C": opt_q["pu_mpa"], "max Q_C": opt_Q["pu_mpa"]}
        grid_rows = None
    else:
        # ---- nothing about T_pre,last: 2-D search
        log("T_pre,last is blank: scanning P_U x T_pre,last")
        npu, nt = int(opts.get("n_pu") or 31), int(opts.get("n_tpre") or 31)
        pus = np.linspace(rng["pu_min"], rng["pu_max"], npu)
        tps = np.linspace(rng["tpre_min"], rng["tpre_max"], nt)
        qrows = do.stage_sweep(fluid, tc_k=tc, pu_values_mpa=pus, tpre_values_k=tps, eta_last=eta_last, flow_model="none")
        good = [r for r in qrows if not r.get("error") and r["q_c_j_kg"] > 0]
        if not good:
            raise RuntimeError("no point in the search window gives q_C > 0; widen the P_U / T_pre ranges")
        mx = max(r["q_c_j_kg"] for r in good)
        top = [r for r in good if r["q_c_j_kg"] >= mx * (1 - 1e-6)]
        best_q = max(top, key=lambda r: r["tpre_last_k"])          # q_C is flat on a plateau: take the warmest T_pre
        out["notes"].append("q_C is flat along a plateau in T_pre; the warmest T_pre on it is reported (cheapest precooling)")
        log("q_C map done; now Q_C (HEM) - this is the slow step")
        summary, qc_rows = do.optimize_last_stage(fluid, tc_k=tc, d_m=d_m, eta_last=eta_last, flow_model=flow,
                                                  pu_min_mpa=rng["pu_min"], pu_max_mpa=rng["pu_max"],
                                                  tpre_min_k=rng["tpre_min"], tpre_max_k=rng["tpre_max"],
                                                  n_pu=int(opts.get("n_pu_Q") or 15), n_tpre=int(opts.get("n_tpre_Q") or 15),
                                                  hem_grid=hem_grid, refine_rounds=2, refine_points=9)
        bq = summary["optimum"]
        bestQ = max((r for r in qc_rows if not r.get("error") and r.get("q_c_w", math.nan) > 0), key=lambda r: r["q_c_w"])
        okQ = [r for r in qc_rows if not r.get("error") and r.get("q_c_w", math.nan) > 0]
        near = [r for r in okQ if r["q_c_w"] >= 0.995 * bestQ["q_c_w"]]
        practical = max(near, key=lambda r: r["tpre_last_k"])
        out["optima"] = {"max_q_C": _row_kv(best_q) | dict(mdot_mg_s=None, Q_C_mW=None), "max_Q_C": _row_kv(bestQ),
                         "Q_C_practical": _row_kv(practical)}
        out["notes"] += summary["boundary_warnings"]
        out["notes"].append("Q_C_practical = warmest T_pre,last whose Q_C is within 0.5 % of the maximum "
                            "(Q_C is nearly flat along the ridge delta_h_T = q_L, so this is the cheaper precooling choice)")
        at_lo = best_q["pu_mpa"] <= rng["pu_min"] * (1 + 1e-9) or best_q["tpre_last_k"] <= rng["tpre_min"] * (1 + 1e-9)
        if at_lo:
            out["notes"].append("max q_C sits at the lower P_U / T_pre bound: q_C alone has no interior optimum when "
                                "T_pre,last is free (it keeps rising toward low P_U and low T_pre). Use Q_C, or fix T_pre,last")
        chosen = {"Q_C": bestQ, "q_C": best_q, "Q_C_practical": practical}.get(drive, bestQ)
        P_U, T_L = chosen["pu_mpa"], chosen["tpre_last_k"]
        marks = {"max q_C": best_q["pu_mpa"], "max Q_C": bestQ["pu_mpa"]}
        grid_rows = (qrows, qc_rows, best_q, bestQ)
        if run is not None:
            _plot_heat(qrows, "q_c_kj_kg", "q_C [kJ/kg]", best_q, run / "map_qC.png", "specific cooling q_C")
            _plot_heat(qc_rows, "q_c_mw", "Q_C [mW]", bestQ, run / "map_QC.png", "gross cooling Q_C (HEM flow)")
    out["P_U_MPa"], out["T_pre_last_K"] = float(P_U), float(T_L)
    # T_pre,opt at the chosen P_U (paper Eq. 34: delta_h_T = q_L)
    t_opt = do.find_tpre_opt(fluid, tc_k=tc, pu_mpa=P_U, tpre_max_k=min(cfg["T_H"], max(40.0, 4 * T_L)))
    out["T_pre_opt_at_P_U_K"] = t_opt
    if cfg.get("P_U"):
        out["T_pre_opt_at_config_P_U_K"] = do.find_tpre_opt(fluid, tc_k=tc, pu_mpa=cfg["P_U"], tpre_max_k=min(cfg["T_H"], max(40.0, 4 * T_L)))
    if run is not None:
        _plot_stage_curves(fluid, cfg, T_L, marks | ({"set P_U": cfg["P_U"]} if cfg.get("P_U") else {}),
                           min(cfg["T_H"], max(rng["tpre_max"], 2 * T_L)), run / "stage_curves.png",
                           (rng["pu_min"], min(rng["pu_max"], (pu_max_theory or rng["pu_max"]) * 1.02)))

    # ---- upstream precoolers (criterion b) and the map
    T_pre = list(cfg["T_pre"])
    if n >= 2:
        T_pre[-1] = float(T_L)
    free = [j for j in range(n - 2) if T_pre[j] is None]
    if free:
        log(f"choosing {len(free)} upstream precooler temperature(s) (minimum Carnot-weighted load)")
        up = optimise_upstream(fluid, cfg, P_U, float(T_L), log)
        T_pre = up["T_pre"]
        out["upstream"] = dict(free=[j + 1 for j in free], objective=up.get("objective"), feasible=up.get("feasible"),
                               criterion="minimise sum_j Q_pre,j (T_H/T_j - 1)")
        if not up.get("feasible"):
            out["notes"].append("no feasible upstream temperature set found")
    # ---- full solve with the chosen values
    final = {**cfg, "P_U": float(P_U), "T_pre": [float(t) for t in T_pre]}
    sol = run_solve(final, log, run)
    out["final_config"] = final
    out["solve"] = sol
    if free and sol.get("metrics") is not None:
        mdot = (sol["metrics"]["mdot_mg_s"] or 0) * 1e-6 or None
        rows = upstream_map(fluid, final, P_U, final["T_pre"], free)
        out["upstream_map"] = rows
        if run is not None:
            import csv
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with (run / "upstream_map.csv").open("w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            _plot_map(rows, final["T_pre"], free, n, run / "upstream_map.png", mdot)
    m = sol.get("metrics") or {}
    out["headline"] = (f"P_U = {P_U:.4g} MPa, T_pre,last = {T_L:.4g} K"
                       + (f", Q_C = {m['Q_C_mW']:.2f} mW" if m.get("Q_C_mW") else ""))
    return out
