"""Optimize the isolated JT stage for a specified fluid, target temperature and orifice.

This is the first of two design-analysis scripts. It uses the last-stage equations
from Zhang & Dang (2021), rather than solving the upstream CHEXs/PHEXs, to search
for the upstream pressure and last PHEX temperature that maximize gross cooling
capacity Q_C.

The existing jt/cycle_core.py remains the shared thermodynamic backend. The
paper's mass-flux fit can be used for comparison; HEM is available for other
property backends/fluids.

Example (run from the project directory):
    python -m jt.design_optimize \
        --fluid he3 \
        --tc 2.1 --diameter-um 20 --stirling-stages 4

By default, all outputs are saved in results/optimization, with the
configuration embedded in filenames, e.g. Helium_3_2.1K_20um_4_stage.json.
Then run the load mapping stage using that exact JSON path:
    python -m jt.phex_load_map \
        --fluid he3 \
        --optimum-json results/optimization/Helium_3_2.1K_20um_4_stage.json \
        --warmer-ranges 30:200:16 8.5:80:16

Units:
    pressure: Pa internally / MPa in CLI and CSV
    enthalpy and specific duties: J/kg (also exports kJ/kg for q_C)
    mdot: kg/s (also mg/s)
    power: W (also mW)
    diameter: micrometres in CLI, metres internally

Important:
* ``paper_fit`` uses the He-3 correlations published in Zhang & Dang's Table 2
  / Eq. (49). Use it only when that model is appropriate. It is limited to the
  fit's pressure and stagnation-temperature range.
* ``hem`` uses jt.cycle_core.critical_mass_flux_HEM and therefore requires the
  property backend to support h(P,s) and rho(P,s) on the expansion path.
* The numerical optimum is only as reliable as the fluid property data, chosen
  search bounds, and search resolution. Inspect the CSV and JSON metadata.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from jt.cycle_core import CoolPropFluid, FluidTable, critical_mass_flux_HEM, solve_cycle


@dataclass(frozen=True)
class LastStagePoint:
    pu_mpa: float
    tpre_last_k: float
    tc_k: float
    pd_pa: float
    eta_last: float
    q_c_j_kg: float
    t8_k: float
    g_kg_m2_s: float
    mdot_kg_s: float
    q_c_w: float
    flow_model: str
    choked: Optional[bool] = None
    throat_pressure_pa: Optional[float] = None
    binding_wall: str = ""
    delta_h_T_j_kg: Optional[float] = None
    q_L_j_kg: Optional[float] = None
    error: str = ""


def _load_fluid(fluid_name: Optional[str], factory_spec: Optional[str]) -> Tuple[FluidTable, str, bool]:
    """Create a built-in CoolProp fluid, the He-3 table backend, or custom backend."""
    if factory_spec:
        if ":" not in factory_spec:
            raise ValueError("Use --fluid-factory package.module:ClassName")
        module_name, class_name = factory_spec.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, class_name)
        fluid = factory()
        is_he3 = any(token in factory_spec.lower().replace("-", "") for token in ("he3", "helium3"))
        return fluid, factory_spec, is_he3
    if not fluid_name:
        raise ValueError("Specify either --fluid or --fluid-factory")
    normalized = fluid_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized in {"he3", "helium3"}:
        module = importlib.import_module("jt.properties")
        fluid = getattr(module, "Helium3Fluid")()
        return fluid, "jt.properties:Helium3Fluid", True
    return CoolPropFluid(fluid_name), f"CoolProp:{fluid_name}", False


# Published Eq. (49) / Table 2 with the table's scalings applied
# (P_U in MPa, G in kg m^-2 s^-1), tabulated at T_C = 1, 2, 2.6, 3 K.
PAPER_EQ49 = {1.0: (1.26e4, 0.44, -0.58e3), 2.0: (1.40e4, 0.40, -2.15e3),
              2.6: (1.59e4, 0.36, -4.36e3), 3.0: (1.82e4, 0.32, -6.91e3)}


def paper_mass_flux_at_temperature(T0_K: float, P_U_MPa: Any) -> Any:
    """Eq. (49) with the stagnation temperature approximated by T_C.

    Between tabulated T_C values the two adjacent fitted curves are linearly
    interpolated (a convenience, not a published correlation). Extrapolation is
    rejected rather than done silently.
    """
    temps = sorted(PAPER_EQ49)
    if not (temps[0] <= float(T0_K) <= temps[-1]):
        raise ValueError(f"Paper fit supports T0 in [{temps[0]}, {temps[-1]}] K; got {T0_K}")
    p = np.asarray(P_U_MPa, dtype=float)
    if np.any((p < 0.2) | (p > 2.0)):
        raise ValueError("The paper's Eq. (49) fit is only used for 0.2 <= P_U <= 2.0 MPa")
    lo = max(t for t in temps if t <= T0_K)
    hi = min(t for t in temps if t >= T0_K)

    def one(t: float) -> Any:
        a, b, c = PAPER_EQ49[t]
        return a * p**b + c

    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
        return one(lo)
    frac = (T0_K - lo) / (hi - lo)
    return (1.0 - frac) * one(lo) + frac * one(hi)


def _paper_mass_flux(T0_K: float, pu_mpa: float) -> float:
    """Paper's fitted choked mass flux G [kg m^-2 s^-1] at (T0, P_U)."""
    return float(paper_mass_flux_at_temperature(float(T0_K), float(pu_mpa)))


def evaluate_last_stage(
    fluid: FluidTable,
    *,
    tc_k: float,
    pu_mpa: float,
    tpre_last_k: float,
    d_m: float,
    eta_last: float = 1.0,
    flow_model: str = "paper_fit",
    hem_grid: int = 180,
) -> Dict[str, Any]:
    """Evaluate one isolated last-stage operating point.

    The thermodynamics is delegated to jt.cycle_core.solve_cycle with a single
    CHEX, so this script and the full-cycle solver share one implementation.
    With n = 1 there are no PHEXs (Tpre=[]) and the only hot-side inlet is
    state A_1, whose temperature is the solver's `Th`. Setting Th = T_pre,last
    makes A_1 the paper's state 7 at (P_U, T_pre,last); nothing upstream is
    solved. The evaporator exit (state 10) is saturated vapour at T_C and
    P_D = P_sat(T_C).

    The last-CHEX effectiveness is eta = Q / Q_max with
    Q_max = min(h_7D - h_10, h_7 - h(P_U, T_C)). The warm-end limit gives
    q_C = Delta h_T and the cold-end limit q_C = q_L, so eta = 1 yields
    q_C = min(Delta h_T, q_L) and T_8 >= T_C always.

    Mass flux is evaluated here, at the stagnation state (P_U, T_8), because
    the choice between the paper's fitted flux and HEM (and the HEM grid) is an
    option of this script rather than of the core solver.
    """
    if not (math.isfinite(tc_k) and tc_k > 0):
        raise ValueError("tc_k must be finite and positive")
    if not (math.isfinite(pu_mpa) and pu_mpa > 0):
        raise ValueError("pu_mpa must be finite and positive")
    if not (math.isfinite(tpre_last_k) and tpre_last_k > tc_k):
        raise ValueError("tpre_last_k must be finite and greater than tc_k")
    if not (math.isfinite(d_m) and d_m > 0):
        raise ValueError("d_m must be finite and positive")
    if not (math.isfinite(eta_last) and 0 <= eta_last <= 1):
        raise ValueError("eta_last must lie in [0, 1]")
    if flow_model not in {"paper_fit", "hem", "none"}:
        raise ValueError("flow_model must be 'paper_fit', 'hem' or 'none'")

    pu_pa = pu_mpa * 1.0e6
    res = solve_cycle(
        fluid,
        Tc=float(tc_k),
        Pu=pu_pa,
        Th=float(tpre_last_k),     # hot inlet of the single CHEX = state 7
        Tpre=[],                   # one CHEX, no PHEX
        eta=[float(eta_last)],
        d=None,                    # flow is evaluated below
        pinch_points=0,            # keep the grid search fast
    )
    pd_pa = res.P_D
    q_c = res.q_C
    # flow_model="none" returns specific quantities only, so a non-positive q_C
    # (no net cooling) is reported rather than rejected.
    if not math.isfinite(q_c) or (flow_model != "none" and q_c <= 0):
        raise ValueError(f"Non-positive or invalid specific cooling capacity q_C={q_c!r} J/kg")

    t8_k = float(res.states[2].T)           # hot stream leaving the CHEX = JT inlet
    if not (math.isfinite(t8_k) and t8_k > 0):
        raise ValueError(f"Property backend returned invalid T_8={t8_k!r} K")

    # Diagnostics: the two limits on q_C.
    h10 = float(fluid.h_sat_vapor(tc_k))
    h7 = float(fluid.h(pu_pa, tpre_last_k))
    delta_h_T = float(fluid.h(pd_pa, tpre_last_k)) - h7
    q_L = h10 - float(fluid.h(pu_pa, tc_k))
    binding = res.chex_binding[0]

    if flow_model == "none":
        g = math.nan
        choked = None
        throat_pressure = None
    elif flow_model == "paper_fit":
        g = _paper_mass_flux(t8_k, pu_mpa)
        if not (math.isfinite(g) and g > 0):
            raise ValueError(f"Paper mass-flux fit returned non-positive G={g!r}")
        choked = None
        throat_pressure = None
    else:
        throat = critical_mass_flux_HEM(pu_pa, t8_k, pd_pa, fluid, n_grid=hem_grid)
        g = float(throat["G"])
        choked = bool(throat["choked"])
        throat_pressure = float(throat["P_star"])
        if not (math.isfinite(g) and g > 0):
            raise ValueError(f"HEM returned non-positive mass flux G={g!r}")

    area = math.pi * d_m**2 / 4.0
    mdot = area * g          # NaN when flow_model == "none"
    q_c_w = mdot * q_c
    return asdict(LastStagePoint(
        pu_mpa=float(pu_mpa),
        tpre_last_k=float(tpre_last_k),
        tc_k=float(tc_k),
        pd_pa=pd_pa,
        eta_last=float(eta_last),
        q_c_j_kg=q_c,
        t8_k=t8_k,
        g_kg_m2_s=g,
        mdot_kg_s=mdot,
        q_c_w=q_c_w,
        flow_model=flow_model,
        choked=choked,
        throat_pressure_pa=throat_pressure,
        binding_wall=binding,
        delta_h_T_j_kg=delta_h_T,
        q_L_j_kg=q_L,
    ))


def _safe_evaluate(fluid: FluidTable, **kwargs: Any) -> Dict[str, Any]:
    try:
        return evaluate_last_stage(fluid, **kwargs)
    except Exception as exc:
        return {
            "pu_mpa": kwargs.get("pu_mpa"),
            "tpre_last_k": kwargs.get("tpre_last_k"),
            "tc_k": kwargs.get("tc_k"),
            "q_c_j_kg": math.nan,
            "q_c_w": math.nan,
            "mdot_kg_s": math.nan,
            "g_kg_m2_s": math.nan,
            "t8_k": math.nan,
            "pd_pa": math.nan,
            "flow_model": kwargs.get("flow_model", ""),
            "error": f"{type(exc).__name__}: {exc}",
        }


def stage_limits(fluid: FluidTable, *, tc_k: float, pu_mpa: float, tpre_last_k: float) -> Dict[str, float]:
    """The two limits on q_C for the isolated last stage, from property lookups only.

    delta_h_T = h(P_D, T_pre) - h(P_U, T_pre)   (isothermal JT enthalpy difference)
    q_L       = h_10 - h(P_U, T_C)              (latent heat of a fully liquefied stream)
    with P_D = P_sat(T_C) and h_10 the saturated-vapour enthalpy at T_C.
    Returns NaNs if the backend cannot evaluate the state.
    """
    try:
        pd_pa = float(fluid.Psat(tc_k))
        pu_pa = pu_mpa * 1.0e6
        h10 = float(fluid.h_sat_vapor(tc_k))
        delta_h_T = float(fluid.h(pd_pa, tpre_last_k)) - float(fluid.h(pu_pa, tpre_last_k))
        q_L = h10 - float(fluid.h(pu_pa, tc_k))
        return {"delta_h_T_j_kg": delta_h_T, "q_L_j_kg": q_L}
    except Exception:
        return {"delta_h_T_j_kg": math.nan, "q_L_j_kg": math.nan}


def stage_sweep(
    fluid: FluidTable,
    *,
    tc_k: float,
    pu_values_mpa: Iterable[float],
    tpre_values_k: Iterable[float],
    eta_last: float = 1.0,
    d_m: float = 20e-6,
    flow_model: str = "none",
    hem_grid: int = 180,
) -> List[Dict[str, Any]]:
    """Evaluate the isolated last stage over a P_U x T_pre grid.

    Each row holds evaluate_last_stage output plus the two limits (delta_h_T and
    q_L) computed independently of the effectiveness model, and kJ/kg, mW and
    mg/s convenience columns. Failed points carry an error string and NaNs.
    """
    rows: List[Dict[str, Any]] = []
    for tpre in tpre_values_k:
        for pu in pu_values_mpa:
            row = _safe_evaluate(
                fluid, tc_k=tc_k, pu_mpa=float(pu), tpre_last_k=float(tpre),
                d_m=d_m, eta_last=eta_last, flow_model=flow_model, hem_grid=hem_grid,
            )
            row.update(stage_limits(fluid, tc_k=tc_k, pu_mpa=float(pu), tpre_last_k=float(tpre)))
            row["q_c_kj_kg"] = _as_float(row.get("q_c_j_kg")) / 1000.0
            row["delta_h_T_kj_kg"] = _as_float(row.get("delta_h_T_j_kg")) / 1000.0
            row["q_L_kj_kg"] = _as_float(row.get("q_L_j_kg")) / 1000.0
            row["q_c_mw"] = _as_float(row.get("q_c_w")) * 1000.0
            row["mdot_mg_s"] = _as_float(row.get("mdot_kg_s")) * 1e6
            rows.append(row)
    return rows


def find_optimum_pu(
    fluid: FluidTable,
    *,
    tc_k: float,
    tpre_last_k: float,
    eta_last: float = 1.0,
    objective: str = "q_c",
    d_m: float = 20e-6,
    flow_model: str = "none",
    hem_grid: int = 180,
    pu_min_mpa: float = 0.02,
    pu_max_mpa: float = 1.2,
    n_grid: int = 41,
    golden_iterations: int = 30,
) -> Dict[str, Any]:
    """Best P_U at a fixed T_pre for the isolated last stage.

    objective "q_c" maximizes the specific cooling capacity (the quantity behind
    the paper's Eqs. 35-36); "Q_C" maximizes gross capacity and needs a flow model.
    A coarse grid locates the peak, and a golden-section search refines it between
    the neighbouring grid points. The objective is evaluated through
    evaluate_last_stage, so the optimum shares the cycle solver's q_C formula.
    """
    if objective not in {"q_c", "Q_C"}:
        raise ValueError("objective must be 'q_c' or 'Q_C'")
    if objective == "Q_C" and flow_model == "none":
        raise ValueError("objective 'Q_C' needs flow_model 'hem' or 'paper_fit'")
    if not (0 < pu_min_mpa < pu_max_mpa) or n_grid < 5:
        raise ValueError("Require 0 < pu_min_mpa < pu_max_mpa and n_grid >= 5")
    key = "q_c_j_kg" if objective == "q_c" else "q_c_w"

    def evaluate(pu: float) -> Dict[str, Any]:
        return _safe_evaluate(
            fluid, tc_k=tc_k, pu_mpa=float(pu), tpre_last_k=tpre_last_k, d_m=d_m,
            eta_last=eta_last, flow_model=flow_model, hem_grid=hem_grid,
        )

    def value(pu: float) -> float:
        v = _as_float(evaluate(pu).get(key))
        return v if math.isfinite(v) else -math.inf

    grid = np.linspace(pu_min_mpa, pu_max_mpa, n_grid)
    vals = np.array([value(float(p)) for p in grid])
    if not np.any(np.isfinite(vals)):
        raise RuntimeError(f"No valid points for T_pre={tpre_last_k} K while searching P_U")
    k = int(np.argmax(vals))
    lo, hi = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, n_grid - 1)])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = value(c), value(d)
    for _ in range(golden_iterations):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = value(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = value(d)
    candidates = [(float(grid[k]), float(vals[k])), (c, fc), (d, fd)]
    pu_best, v_best = max(candidates, key=lambda t: t[1])
    row = evaluate(pu_best)
    row.update(stage_limits(fluid, tc_k=tc_k, pu_mpa=pu_best, tpre_last_k=tpre_last_k))
    at_bound = k in (0, n_grid - 1) and math.isclose(
        pu_best, float(grid[k]), rel_tol=0.0, abs_tol=1e-6)
    return {"pu_opt_mpa": pu_best, "value": v_best, "objective": objective,
            "at_search_bound": bool(at_bound), "row": row}


def find_pu_max(
    fluid: FluidTable, *, tc_k: float, pu_lo_mpa: float = 0.02, pu_hi_mpa: float = 3.0,
) -> Optional[float]:
    """Upstream pressure at which q_L reaches zero (the paper's P_U^MAX, Eq. 32).

    Above it the stream cannot be liquefied at T_C for any precooling temperature.
    Returns None if q_L does not change sign in the bracket.
    """
    from scipy.optimize import brentq

    h10 = float(fluid.h_sat_vapor(tc_k))

    def q_l(p_mpa: float) -> float:
        return h10 - float(fluid.h(p_mpa * 1.0e6, tc_k))

    try:
        if q_l(pu_lo_mpa) <= 0.0 or q_l(pu_hi_mpa) >= 0.0:
            return None
        return float(brentq(q_l, pu_lo_mpa, pu_hi_mpa, xtol=1e-9))
    except Exception:
        return None


def find_tpre_opt(
    fluid: FluidTable, *, tc_k: float, pu_mpa: float,
    tpre_min_k: Optional[float] = None, tpre_max_k: float = 40.0,
) -> Optional[float]:
    """Precooling temperature where delta_h_T = q_L at fixed P_U (paper's Eq. 34).

    delta_h_T falls as T_pre rises while q_L does not depend on T_pre, so the
    crossing is the maximum of q_C = min(delta_h_T, q_L). Returns None if there
    is no sign change in [tpre_min_k, tpre_max_k].
    """
    from scipy.optimize import brentq

    lo = tpre_min_k if tpre_min_k is not None else max(1.2 * tc_k, tc_k + 0.2)

    def f(t: float) -> float:
        lim = stage_limits(fluid, tc_k=tc_k, pu_mpa=pu_mpa, tpre_last_k=t)
        return lim["delta_h_T_j_kg"] - lim["q_L_j_kg"]

    try:
        f_lo, f_hi = f(lo), f(tpre_max_k)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo <= 0.0 or f_hi >= 0.0:
            return None
        return float(brentq(f, lo, tpre_max_k, xtol=1e-7))
    except Exception:
        return None


def mass_flux_vs_pu(
    fluid: FluidTable,
    *,
    t0_k: float,
    pu_values_mpa: Iterable[float],
    pd_pa: Optional[float] = None,
    hem_grid: int = 180,
) -> List[Dict[str, Any]]:
    """HEM choked mass flux G at stagnation state (P_U, T0), back pressure P_D.

    This is the restriction calculation of the cycle with the inlet temperature
    prescribed, e.g. T0 = T_C as in the paper's Section 4 (T_8 ~ T_C). P_D defaults
    to P_sat(T0).
    """
    back = float(pd_pa) if pd_pa is not None else float(fluid.Psat(t0_k))
    rows: List[Dict[str, Any]] = []
    for pu in pu_values_mpa:
        row: Dict[str, Any] = {"t0_k": float(t0_k), "pu_mpa": float(pu), "pd_pa": back,
                               "g_kg_m2_s": math.nan, "choked": None,
                               "throat_pressure_pa": math.nan, "phase": "", "error": ""}
        try:
            throat = critical_mass_flux_HEM(float(pu) * 1.0e6, float(t0_k), back, fluid, n_grid=hem_grid)
            row.update({"g_kg_m2_s": float(throat["G"]), "choked": bool(throat["choked"]),
                        "throat_pressure_pa": float(throat["P_star"]),
                        "phase": str(throat.get("phase", ""))})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Make a stable header even for error-only sweeps.
    preferred = [
        "pu_mpa", "tpre_last_k", "tc_k", "pd_pa", "eta_last", "q_c_j_kg",
        "t8_k", "g_kg_m2_s", "mdot_kg_s", "q_c_w", "flow_model",
        "choked", "throat_pressure_pa", "error",
    ]
    keys = list(dict.fromkeys(preferred + [key for row in rows for key in row.keys()]))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_heatmap(path: Path, rows: Sequence[Mapping[str, Any]], best: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    good = [r for r in rows if math.isfinite(_as_float(r.get("q_c_w"))) and not r.get("error")]
    if len(good) < 4:
        return
    try:
        x = np.asarray([float(r["tpre_last_k"]) for r in good])
        y = np.asarray([float(r["pu_mpa"]) for r in good])
        z = np.asarray([float(r["q_c_w"]) * 1000.0 for r in good])
        fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        contour = ax.tricontourf(x, y, z, levels=18)
        fig.colorbar(contour, ax=ax, label=r"$Q_C$ (mW)")
        ax.plot(float(best["tpre_last_k"]), float(best["pu_mpa"]), marker="o", mfc="none", mew=2, ms=11,
                color="black", label="Maximum sampled $Q_C$")
        ax.set_xlabel(r"Last PHEX precooling temperature $T_{pre,last}$ (K)")
        ax.set_ylabel(r"Upstream pressure $P_U$ (MPa)")
        ax.set_title("Isolated JT-stage gross cooling capacity")
        ax.legend(loc="best")
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception:
        # A plot failure must not invalidate the calculated CSV/JSON results.
        return


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def optimize_last_stage(
    fluid: FluidTable,
    *,
    tc_k: float,
    d_m: float,
    eta_last: float,
    flow_model: str,
    pu_min_mpa: float,
    pu_max_mpa: float,
    tpre_min_k: float,
    tpre_max_k: float,
    n_pu: int = 101,
    n_tpre: int = 101,
    hem_grid: int = 180,
    refine_rounds: int = 2,
    refine_points: int = 15,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Grid search, then locally refine around the best Q_C point."""
    if not (0 < pu_min_mpa < pu_max_mpa):
        raise ValueError("Require 0 < pu_min_mpa < pu_max_mpa")
    if not (tc_k < tpre_min_k < tpre_max_k):
        raise ValueError("Require tc_k < tpre_min_k < tpre_max_k")
    if n_pu < 3 or n_tpre < 3:
        raise ValueError("n_pu and n_tpre must be at least 3")
    if refine_rounds < 0 or refine_points < 3:
        raise ValueError("refine_rounds must be >= 0 and refine_points >= 3")

    all_rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int]] = set()

    def sample(pu_values: Iterable[float], tpre_values: Iterable[float]) -> None:
        for tpre in tpre_values:
            for pu in pu_values:
                # Round only the cache key; preserve full precision in actual calls.
                key = (round(float(pu) * 1e9), round(float(tpre) * 1e9))
                if key in seen:
                    continue
                seen.add(key)
                row = _safe_evaluate(
                    fluid, tc_k=tc_k, pu_mpa=float(pu), tpre_last_k=float(tpre),
                    d_m=d_m, eta_last=eta_last, flow_model=flow_model, hem_grid=hem_grid,
                )
                row["q_c_kj_kg"] = _as_float(row.get("q_c_j_kg")) / 1000.0
                row["q_c_mw"] = _as_float(row.get("q_c_w")) * 1000.0
                row["mdot_mg_s"] = _as_float(row.get("mdot_kg_s")) * 1e6
                all_rows.append(row)

    pu_values = np.linspace(pu_min_mpa, pu_max_mpa, n_pu)
    tpre_values = np.linspace(tpre_min_k, tpre_max_k, n_tpre)
    sample(pu_values, tpre_values)

    def best_row() -> Dict[str, Any]:
        good = [r for r in all_rows if not r.get("error") and math.isfinite(_as_float(r.get("q_c_w")))
                and _as_float(r.get("q_c_w")) > 0]
        if not good:
            top_errors = [str(r.get("error", "")) for r in all_rows if r.get("error")][:5]
            raise RuntimeError(
                "No valid positive-Q_C operating points found. Check the property backend, "
                "flow model, and search bounds. Sample errors: " + " | ".join(top_errors)
            )
        return max(good, key=lambda r: _as_float(r["q_c_w"]))

    best = best_row()
    pu_step = float(pu_values[1] - pu_values[0])
    tpre_step = float(tpre_values[1] - tpre_values[0])
    for _ in range(refine_rounds):
        center_pu = float(best["pu_mpa"])
        center_tpre = float(best["tpre_last_k"])
        p_lo = max(pu_min_mpa, center_pu - pu_step)
        p_hi = min(pu_max_mpa, center_pu + pu_step)
        t_lo = max(tpre_min_k, center_tpre - tpre_step)
        t_hi = min(tpre_max_k, center_tpre + tpre_step)
        sample(np.linspace(p_lo, p_hi, refine_points), np.linspace(t_lo, t_hi, refine_points))
        best = best_row()
        pu_step = (p_hi - p_lo) / max(refine_points - 1, 1)
        tpre_step = (t_hi - t_lo) / max(refine_points - 1, 1)

    # Save the best point according to gross capacity and the best sampled q_C point
    # separately; these optima need not coincide for all Tpre/flow conditions.
    best = best_row()
    good = [r for r in all_rows if not r.get("error") and math.isfinite(_as_float(r.get("q_c_j_kg")))
            and _as_float(r.get("q_c_j_kg")) > 0]
    best_qc = max(good, key=lambda r: _as_float(r["q_c_j_kg"]))
    bound_warnings = []
    if math.isclose(float(best["pu_mpa"]), pu_min_mpa, rel_tol=0.0, abs_tol=1e-8) or math.isclose(float(best["pu_mpa"]), pu_max_mpa, rel_tol=0.0, abs_tol=1e-8):
        bound_warnings.append("The maximizing pressure is at a search bound; widen the P_U search range to confirm the optimum.")
    if math.isclose(float(best["tpre_last_k"]), tpre_min_k, rel_tol=0.0, abs_tol=1e-7) or math.isclose(float(best["tpre_last_k"]), tpre_max_k, rel_tol=0.0, abs_tol=1e-7):
        bound_warnings.append("The maximizing last-PHEX temperature is at a search bound; widen the temperature range to confirm the optimum.")
    summary = {
        "schema_version": 1,
        "method": "isolated_last_stage_grid_search_with_local_refinement",
        "fluid_backend": None,
        "T_C_K": float(tc_k),
        "diameter_m": float(d_m),
        "diameter_um": float(d_m * 1e6),
        "eta_last": float(eta_last),
        "flow_model": flow_model,
        "search_bounds": {
            "P_U_MPa": [float(pu_min_mpa), float(pu_max_mpa)],
            "T_pre_last_K": [float(tpre_min_k), float(tpre_max_k)],
        },
        "coarse_grid": {"n_P_U": int(n_pu), "n_T_pre_last": int(n_tpre)},
        "refinement": {"rounds": int(refine_rounds), "points_per_axis": int(refine_points)},
        "optimum": {
            "P_U_MPa": float(best["pu_mpa"]),
            "T_pre_last_K": float(best["tpre_last_k"]),
            "P_D_Pa": float(best["pd_pa"]),
            "q_C_J_kg": float(best["q_c_j_kg"]),
            "q_C_kJ_kg": float(best["q_c_j_kg"]) / 1000.0,
            "T_8_K": float(best["t8_k"]),
            "G_kg_m2_s": float(best["g_kg_m2_s"]),
            "mdot_kg_s": float(best["mdot_kg_s"]),
            "mdot_mg_s": float(best["mdot_kg_s"]) * 1e6,
            "Q_C_W": float(best["q_c_w"]),
            "Q_C_mW": float(best["q_c_w"]) * 1000.0,
            "choked": best.get("choked"),
            "throat_pressure_Pa": best.get("throat_pressure_pa"),
        },
        "maximum_sampled_specific_capacity": {
            "P_U_MPa": float(best_qc["pu_mpa"]),
            "T_pre_last_K": float(best_qc["tpre_last_k"]),
            "q_C_J_kg": float(best_qc["q_c_j_kg"]),
            "q_C_kJ_kg": float(best_qc["q_c_j_kg"]) / 1000.0,
            "mdot_kg_s": float(best_qc["mdot_kg_s"]),
            "Q_C_W_at_that_point": float(best_qc["q_c_w"]),
        },
        "valid_evaluations": sum(1 for r in all_rows if not r.get("error")),
        "failed_evaluations": sum(1 for r in all_rows if r.get("error")),
        "boundary_warnings": bound_warnings,
        "warning": (
            "Numerical result for the selected property and flow models; not a prediction "
            "of PHEX stage compatibility. Phase 2 calculates those loads."
        ),
    }
    return summary, all_rows


def _format_filename_number(value: float) -> str:
    """Compact numeric filename field without unnecessary trailing zeroes."""
    return format(float(value), ".12g")


def _fluid_filename_label(fluid_name: Optional[str], factory_spec: Optional[str]) -> str:
    """Map common fluid selectors to readable, stable labels for run filenames."""
    source = fluid_name or (factory_spec.split(":", 1)[-1] if factory_spec else "Fluid")
    normalized = re.sub(r"[^a-z0-9]+", "", source.lower())
    if normalized in {"he3", "helium3", "heliumthree"} or "helium3" in normalized or "he3" in normalized:
        return "Helium_3"
    if normalized in {"he4", "helium4", "heliumfour"} or "helium4" in normalized or "he4" in normalized:
        return "Helium_4"
    # Keep user-defined/CoolProp names readable while avoiding path separators.
    label = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_")
    return label or "Fluid"


def _build_run_stem(
    fluid_name: Optional[str], factory_spec: Optional[str], tc_k: float,
    diameter_um: float, stirling_stages: int,
) -> str:
    fluid_label = _fluid_filename_label(fluid_name, factory_spec)
    tc_label = _format_filename_number(tc_k)
    diameter_label = _format_filename_number(diameter_um)
    return f"{fluid_label}_{tc_label}K_{diameter_label}um_{int(stirling_stages)}_stage"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    fluid_group = parser.add_mutually_exclusive_group(required=True)
    fluid_group.add_argument("--fluid", help="Fluid name for CoolProp, or 'he3' for the He-3 CSV backend")
    fluid_group.add_argument("--fluid-factory", help="Custom no-argument backend as module:ClassName")
    parser.add_argument("--tc", type=float, required=True, help="Target JT cooling temperature [K]")
    parser.add_argument("--diameter-um", type=float, required=True, help="Orifice diameter [micrometres]")
    parser.add_argument("--stirling-stages", type=int, required=True,
                        help="Number of Stirling stages; recorded in the output filename and JSON")
    parser.add_argument("--eta-last", type=float, default=1.0, help="Last CHEX effectiveness [default: 1]")
    parser.add_argument("--flow-model", choices=("auto", "paper_fit", "hem"), default="auto",
                        help="auto uses paper_fit for He-3 and HEM otherwise; override with paper_fit or hem")
    parser.add_argument("--pu-min-mpa", type=float, default=0.20)
    parser.add_argument("--pu-max-mpa", type=float, default=1.20)
    parser.add_argument("--tpre-min-k", type=float, default=None,
                        help="Default max(4 K, Tc + 0.1 K)")
    parser.add_argument("--tpre-max-k", type=float, default=30.0)
    parser.add_argument("--n-pu", type=int, default=101, help="Coarse pressure-grid points")
    parser.add_argument("--n-tpre", type=int, default=101, help="Coarse last-PHEX temperature-grid points")
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--refine-points", type=int, default=15)
    parser.add_argument("--hem-grid", type=int, default=180,
                        help="Pressure samples used by each HEM evaluation")
    parser.add_argument("--outdir", default="results/optimization",
                        help="Directory for all result files [default: results/optimization]")
    args = parser.parse_args(argv)

    if args.diameter_um <= 0:
        parser.error("--diameter-um must be positive")
    if args.stirling_stages < 2:
        parser.error("--stirling-stages must be at least 2")
    tpre_min = max(4.0, args.tc + 0.1) if args.tpre_min_k is None else args.tpre_min_k
    try:
        fluid, backend_label, is_he3 = _load_fluid(args.fluid, args.fluid_factory)
        flow_model = args.flow_model
        if flow_model == "auto":
            flow_model = "paper_fit" if is_he3 else "hem"
        if flow_model == "paper_fit" and not is_he3:
            print("WARNING: paper_fit uses the published He-3 mass-flux fit. Confirm it is appropriate for this backend.")
        summary, rows = optimize_last_stage(
            fluid,
            tc_k=args.tc,
            d_m=args.diameter_um * 1e-6,
            eta_last=args.eta_last,
            flow_model=flow_model,
            pu_min_mpa=args.pu_min_mpa,
            pu_max_mpa=args.pu_max_mpa,
            tpre_min_k=tpre_min,
            tpre_max_k=args.tpre_max_k,
            n_pu=args.n_pu,
            n_tpre=args.n_tpre,
            hem_grid=args.hem_grid,
            refine_rounds=args.refine_rounds,
            refine_points=args.refine_points,
        )
        summary["fluid_backend"] = backend_label
        summary["stirling_stages"] = int(args.stirling_stages)
        summary["number_of_PHEXs"] = int(args.stirling_stages - 1)
        stem = _build_run_stem(args.fluid, args.fluid_factory, args.tc, args.diameter_um, args.stirling_stages)
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        json_path = outdir / f"{stem}.json"
        csv_path = outdir / f"{stem}_optimization_surface.csv"
        _write_csv(csv_path, rows)
        json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        plot_path = outdir / f"{stem}_QC_heatmap.png"
        _write_heatmap(plot_path, rows, summary["optimum"] | {
            "pu_mpa": summary["optimum"]["P_U_MPa"],
            "tpre_last_k": summary["optimum"]["T_pre_last_K"],
            "q_c_w": summary["optimum"]["Q_C_W"],
        })
        print("JT last-stage optimization complete.")
        print(f"  P_U*             = {summary['optimum']['P_U_MPa']:.8g} MPa")
        print(f"  T_pre,last*      = {summary['optimum']['T_pre_last_K']:.8g} K")
        print(f"  q_C              = {summary['optimum']['q_C_kJ_kg']:.8g} kJ/kg")
        print(f"  mdot             = {summary['optimum']['mdot_mg_s']:.8g} mg/s")
        print(f"  Q_C              = {summary['optimum']['Q_C_mW']:.8g} mW")
        for warning in summary.get("boundary_warnings", []):
            print(f"  WARNING: {warning}")
        print(f"  optimum JSON     : {json_path.resolve()}")
        print(f"  surface CSV      : {csv_path.resolve()}")
        if plot_path.exists():
            print(f"  capacity heatmap : {plot_path.resolve()}")
    except Exception as exc:
        parser.exit(2, f"ERROR: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
