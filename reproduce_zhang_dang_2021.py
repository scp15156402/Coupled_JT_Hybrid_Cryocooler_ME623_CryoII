"""Reproduction checks for Zhang & Dang (Cryogenics 116, 2021, 103282).

This script only orchestrates, compares and plots. The calculations live in
    jt/cycle_core.py       cycle solver (effectiveness model, HEM restriction flow)
    jt/design_optimize.py  last-stage sweeps, optimum search, mass-flux curves
The curve-fitting and error-metric helpers and the paper's published numbers
and fits are kept in this file.

Sections (each writes a CSV, one PNG and metrics into reproduction_report.json):
  fits   Eq. 35 / 36: optimum P_U and q_C versus T_pre3 (eta = 1), refitted
         and compared with the published curves.
  fig5   delta_h_T, q_L and q_C versus P_U at T_C = 1 K, T_pre3 = 8 K.
  fig7   delta_h_T and q_C versus T_pre3 for several P_U; T_pre3,opt versus Eq. 36.
  flux   HEM mass flux versus P_U for several T_C with stagnation T0 = T_C
         (the paper's T_8 ~ T_C), refitted to Eq. 49's form and compared with Table 2.
  fig15  q_C, mdot and Q_C versus last-CHEX effectiveness (Figs. 15 and 16).

The published numbers are for He-3 at T_C = 1 K, d = 20 um. With any other fluid
or T_C the paper comparisons are not meaningful (the report says so); the
sections then only exercise the machinery.

Usage:
    python reproduce_zhang_dang_2021.py --self-test
    python reproduce_zhang_dang_2021.py --fluid he3 --outdir results/validation
    python reproduce_zhang_dang_2021.py --fluid Helium --tc 2.6 --flux-temps 2.3 2.6 3 --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from jt import design_optimize as jdo
from jt.design_optimize import PAPER_EQ49 as EQ49

# ---------------------------------------------------------------------------
# Curve fitting and error metrics (no fluid model needed)


def error_metrics(reference: Sequence[float], predicted: Sequence[float]) -> Dict[str, float]:
    """Error metrics of `predicted` against `reference` (paired, finite points only).

    Errors are predicted - reference. max_rel_percent and mape_percent are
    relative to the reference values (points with reference ~ 0 are excluded).
    """
    y_ref = np.asarray(reference, dtype=float).ravel()
    y_pred = np.asarray(predicted, dtype=float).ravel()
    if y_ref.shape != y_pred.shape:
        raise ValueError("reference and predicted must have the same length")
    mask = np.isfinite(y_ref) & np.isfinite(y_pred)
    y_ref, y_pred = y_ref[mask], y_pred[mask]
    keys = ("n", "rmse", "mae", "max_abs_error", "max_rel_percent", "mape_percent", "bias", "r_squared")
    if y_ref.size == 0:
        return {k: (0 if k == "n" else math.nan) for k in keys}
    err = y_pred - y_ref
    nz = np.abs(y_ref) > np.finfo(float).eps
    rel = np.abs(err[nz] / y_ref[nz]) * 100.0 if np.any(nz) else np.array([math.nan])
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    return {
        "n": int(y_ref.size),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "max_rel_percent": float(np.max(rel)),
        "mape_percent": float(np.mean(rel)),
        "bias": float(np.mean(err)),
        "r_squared": float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else math.nan,
    }


def _finite_pairs(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    xa, ya = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def fit_polynomial(x: Sequence[float], y: Sequence[float], degree: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """Least-squares polynomial. Returns coefficients in ASCENDING powers (a0, a1, ...)
    and the fit-versus-data metrics."""
    xa, ya = _finite_pairs(x, y)
    if xa.size <= degree:
        raise ValueError(f"Need more than {degree} finite points for a degree-{degree} fit")
    desc = np.polyfit(xa, ya, degree)
    return desc[::-1], error_metrics(ya, np.polyval(desc, xa))


def polyval_ascending(coefficients: Sequence[float], x: Sequence[float]) -> np.ndarray:
    return np.polynomial.polynomial.polyval(np.asarray(x, dtype=float), np.asarray(coefficients, dtype=float))


def fit_plane(a: Sequence[float], b: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Least-squares y = c0 + c1*a + c2*b (the form of the paper's Eq. 35, with
    a = P_U,opt and b = T_pre3). Returns (c0, c1, c2) and fit metrics."""
    aa, bb, yy = (np.asarray(v, dtype=float).ravel() for v in (a, b, y))
    mask = np.isfinite(aa) & np.isfinite(bb) & np.isfinite(yy)
    aa, bb, yy = aa[mask], bb[mask], yy[mask]
    if yy.size < 3:
        raise ValueError("Need at least three finite points for a plane fit")
    design = np.column_stack([np.ones_like(aa), aa, bb])
    coef, *_ = np.linalg.lstsq(design, yy, rcond=None)
    return coef, error_metrics(yy, design @ coef)


def power_law(p: Sequence[float], alpha: float, beta: float, gamma: float) -> np.ndarray:
    """G(P) = alpha * P**beta + gamma (the form of the paper's Eq. 49)."""
    return alpha * np.asarray(p, dtype=float) ** beta + gamma


def fit_power_law(p: Sequence[float], g: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Nonlinear least-squares fit of G = alpha * P**beta + gamma."""
    from scipy.optimize import curve_fit

    x, y = _finite_pairs(p, g)
    keep = x > 0
    x, y = x[keep], y[keep]
    if x.size < 4:
        raise ValueError("Need at least four valid points for the three-parameter fit")
    p0 = (max(float(np.max(y)), 1.0), 0.4, 0.0)
    params, _ = curve_fit(power_law, x, y, p0=p0, maxfev=200000)
    return ({"alpha": float(params[0]), "beta": float(params[1]), "gamma": float(params[2])},
            error_metrics(y, power_law(x, *params)))


# ---------------------------------------------------------------------------
# Published numbers and fits (data only)

PAPER = {
    "T_C_K": 1.0, "d_um": 20.0, "T_pre3_K": 8.0,
    "P_U_opt_MPa": 0.36, "q_C_opt_kJ_kg": 8.0, "P_U_max_MPa": 1.13,
    "mdot_mg_s": 2.34, "Q_C_ideal_mW": 18.7, "Q_C_eta097_mW": 14.7,
    "eta_real": 0.97, "drop_per_percent_eta": 7.0,
}
PAPER["Q_C_drop_097_percent"] = 100.0 * (1.0 - PAPER["Q_C_eta097_mW"] / PAPER["Q_C_ideal_mW"])

# Eq. (36), P_U,opt [MPa] versus T_pre3 [K]; Table 1, ascending powers.
EQ36_ASC = [-0.102, 0.074, -2.37e-3, 4.88e-5, -4.30e-7]
# Eq. (35): q_C,opt [kJ/kg] = b0 + b1*P_U,opt[MPa] + b2*T_pre3[K].
EQ35 = (11.9, -11.5, 0.027)


def paper_pu_opt(t_pre3: Any) -> np.ndarray:
    return polyval_ascending(EQ36_ASC, t_pre3)


def paper_qc_opt(pu_opt: Any, t_pre3: Any) -> np.ndarray:
    return EQ35[0] + EQ35[1] * np.asarray(pu_opt, float) + EQ35[2] * np.asarray(t_pre3, float)


def paper_tpre_opt(pu_mpa: float) -> float:
    """Invert Eq. (36): T_pre3 at which P_U,opt equals pu_mpa (NaN outside 4-35 K)."""
    t = np.linspace(4.0, 35.0, 3101)
    p = paper_pu_opt(t)
    if not (p[0] <= pu_mpa <= p[-1]) or np.any(np.diff(p) <= 0):
        return math.nan
    return float(np.interp(pu_mpa, p, t))


# ---------------------------------------------------------------------------
# Output helpers

def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _arr(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.array([jdo._as_float(r.get(key)) for r in rows], dtype=float)


def _clean(obj: Any) -> Any:
    """Make nested metrics JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Sections

def section_fits(fluid, cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """Eq. 35 / 36: optimum P_U and q_C versus T_pre3 at eta = 1."""
    tc = cfg["tc"]
    t_grid = np.asarray(cfg["tpre_grid"], dtype=float)
    rows: List[Dict[str, Any]] = []
    for t in t_grid:
        try:
            res = jdo.find_optimum_pu(fluid, tc_k=tc, tpre_last_k=float(t), eta_last=1.0,
                                      objective="q_c", pu_min_mpa=cfg["pu_min"], pu_max_mpa=cfg["pu_max"],
                                      n_grid=cfg["n_opt_grid"])
            r = res["row"]
            rows.append({"T_pre3_K": float(t), "P_U_opt_MPa": res["pu_opt_mpa"],
                         "q_C_opt_kJ_kg": r["q_c_j_kg"] / 1000.0,
                         "delta_h_T_kJ_kg": r["delta_h_T_j_kg"] / 1000.0, "q_L_kJ_kg": r["q_L_j_kg"] / 1000.0,
                         "binding_wall": r.get("binding_wall", ""), "at_search_bound": res["at_search_bound"],
                         "error": ""})
        except Exception as exc:
            rows.append({"T_pre3_K": float(t), "P_U_opt_MPa": math.nan, "q_C_opt_kJ_kg": math.nan,
                         "error": f"{type(exc).__name__}: {exc}"})
    good = [r for r in rows if math.isfinite(r["P_U_opt_MPa"]) and not r.get("at_search_bound")]
    t_ok = np.array([r["T_pre3_K"] for r in good])
    pu_ok = np.array([r["P_U_opt_MPa"] for r in good])
    qc_ok = np.array([r["q_C_opt_kJ_kg"] for r in good])
    for r in rows:
        r["P_U_opt_Eq36_MPa"] = float(paper_pu_opt(r["T_pre3_K"]))
        r["q_C_opt_Eq35_36_kJ_kg"] = float(paper_qc_opt(r["P_U_opt_Eq36_MPa"], r["T_pre3_K"]))
    _write_table(out / "fits_optimum_vs_tpre3.csv", rows)

    summary: Dict[str, Any] = {"n_points": len(rows), "n_valid": len(good),
                               "failed": [r["T_pre3_K"] for r in rows if r.get("error")],
                               "at_search_bound": [r["T_pre3_K"] for r in rows if r.get("at_search_bound")]}
    if len(good) >= 5:
        # Self-consistency with Eq. 34: at the optimum delta_h_T = q_L.
        dh, ql = _arr(good, "delta_h_T_kJ_kg"), _arr(good, "q_L_kJ_kg")
        summary["max_rel_gap_deltahT_vs_qL_at_optimum"] = float(np.max(np.abs(dh - ql) / np.abs(ql)))
        coef36, m36_internal = fit_polynomial(t_ok, pu_ok, 4)
        coef35, m35_internal = fit_plane(pu_ok, t_ok, qc_ok)
        summary["eq36_refit"] = {"ours_a0_a4": list(coef36), "paper_a0_a4": EQ36_ASC, "fit_vs_our_points": m36_internal}
        summary["eq35_refit"] = {"ours_b0_b1_b2": list(coef35), "paper_b0_b1_b2": list(EQ35), "fit_vs_our_points": m35_internal}
        summary["P_U_opt_vs_Eq36"] = error_metrics(paper_pu_opt(t_ok), pu_ok)
        summary["q_C_opt_vs_Eq35_36_chain"] = error_metrics(paper_qc_opt(paper_pu_opt(t_ok), t_ok), qc_ok)
        summary["q_C_opt_vs_Eq35_at_our_PU"] = error_metrics(paper_qc_opt(pu_ok, t_ok), qc_ok)
        summary["P_U_opt_refit_vs_Eq36"] = error_metrics(paper_pu_opt(t_ok), polyval_ascending(coef36, t_ok))

        plt = _plt()
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        tt = np.linspace(t_ok.min(), t_ok.max(), 200)
        ax[0].plot(tt, paper_pu_opt(tt), "k-", label="Paper Eq. 36")
        ax[0].plot(t_ok, pu_ok, "o", ms=4, label="This work (optimum)")
        ax[0].plot(tt, polyval_ascending(coef36, tt), "--", label="This work, refit (deg 4)")
        ax[0].set(xlabel="T_pre3 [K]", ylabel="P_U,opt [MPa]", title="Optimum upstream pressure")
        ax[1].plot(tt, paper_qc_opt(paper_pu_opt(tt), tt), "k-", label="Paper Eq. 35 + 36")
        ax[1].plot(t_ok, qc_ok, "o", ms=4, label="This work (optimum)")
        ax[1].plot(tt, coef35[0] + coef35[1] * np.interp(tt, t_ok, pu_ok) + coef35[2] * tt, "--",
                   label="This work, Eq. 35 form refit")
        ax[1].set(xlabel="T_pre3 [K]", ylabel="q_C,opt [kJ/kg]", title="Optimum specific cooling capacity")
        ax[2].plot(t_ok, 100 * (pu_ok - paper_pu_opt(t_ok)) / paper_pu_opt(t_ok), "o-", ms=4, label="P_U,opt")
        ax[2].plot(t_ok, 100 * (qc_ok - paper_qc_opt(paper_pu_opt(t_ok), t_ok)) / paper_qc_opt(paper_pu_opt(t_ok), t_ok),
                   "s-", ms=4, label="q_C,opt")
        ax[2].axhline(0, color="k", lw=0.8)
        ax[2].set(xlabel="T_pre3 [K]", ylabel="This work vs paper [%]", title="Relative difference")
        for a in ax:
            a.grid(alpha=0.3); a.legend(fontsize=8)
        fig.suptitle(f"Eqs. 35/36 versus independent optimum search (T_C = {tc:g} K, eta = 1)")
        fig.savefig(out / "fits_eq35_36.png", dpi=170); plt.close(fig)
    return summary


def section_fig5(fluid, cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """delta_h_T, q_L and q_C versus P_U at fixed T_pre3, eta = 1 (paper Fig. 5)."""
    tc, tp = cfg["tc"], cfg["fig5_tpre"]
    pu = np.linspace(cfg["pu_min"], cfg["pu_max"], cfg["n_curve"])
    rows = jdo.stage_sweep(fluid, tc_k=tc, pu_values_mpa=pu, tpre_values_k=[tp], eta_last=1.0, flow_model="none")
    _write_table(out / "fig5_qc_deltahT_qL_vs_pu.csv", rows)
    x, qc = _arr(rows, "pu_mpa"), _arr(rows, "q_c_kj_kg")
    dh, ql = _arr(rows, "delta_h_T_kj_kg"), _arr(rows, "q_L_kj_kg")
    ok = np.isfinite(qc) & np.isfinite(dh) & np.isfinite(ql) & (qc > 0)
    summary: Dict[str, Any] = {"T_C_K": tc, "T_pre3_K": tp, "n_valid": int(ok.sum()), "n": len(rows)}
    if ok.sum() >= 3:
        # q_C from the cycle solver against min(delta_h_T, q_L) from direct property lookups.
        summary["max_rel_dev_qC_vs_min_deltahT_qL"] = float(np.max(np.abs(qc[ok] - np.minimum(dh[ok], ql[ok])) / qc[ok]))
    opt = jdo.find_optimum_pu(fluid, tc_k=tc, tpre_last_k=tp, eta_last=1.0, objective="q_c",
                              pu_min_mpa=cfg["pu_min"], pu_max_mpa=cfg["pu_max"], n_grid=cfg["n_opt_grid"])
    pu_max = jdo.find_pu_max(fluid, tc_k=tc)
    summary.update({"P_U_opt_MPa": opt["pu_opt_mpa"], "q_C_opt_kJ_kg": opt["value"] / 1000.0,
                    "optimum_at_search_bound": opt["at_search_bound"], "P_U_max_MPa": pu_max,
                    "paper": {"P_U_opt_MPa": PAPER["P_U_opt_MPa"], "q_C_opt_kJ_kg": PAPER["q_C_opt_kJ_kg"],
                              "P_U_max_MPa": PAPER["P_U_max_MPa"]}})
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    ax.plot(x, dh, "--", color="tab:blue", label="Δh_T")
    ax.plot(x, ql, "--", color="tab:red", label="q_L")
    ax.plot(x, np.where(ok, qc, np.nan), "-", color="k", lw=2, label="q_C (eta = 1)")
    ax.plot(opt["pu_opt_mpa"], opt["value"] / 1000.0, "ko", ms=7, label="this work: optimum")
    if tc == PAPER["T_C_K"] and tp == PAPER["T_pre3_K"]:
        ax.plot(PAPER["P_U_opt_MPa"], PAPER["q_C_opt_kJ_kg"], "g*", ms=14, label="paper: optimum (text)")
        ax.axvline(PAPER["P_U_max_MPa"], color="g", ls=":", label="paper: P_U max (text)")
    ax.set(xlabel="P_U [MPa]", ylabel="specific capacity [kJ/kg]", title=f"Fig. 5 check: T_C = {tc:g} K, T_pre3 = {tp:g} K")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend()
    fig.savefig(out / "fig5.png", dpi=170); plt.close(fig)
    return summary


def section_fig7(fluid, cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """delta_h_T and q_C versus T_pre3 for several P_U (paper Fig. 7)."""
    tc = cfg["tc"]
    t_grid = np.linspace(cfg["fig7_tmin"], cfg["fig7_tmax"], cfg["n_curve"])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.8, 5.4), constrained_layout=True)
    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"T_C_K": tc, "per_P_U": []}
    for color, pu in zip(("tab:orange", "tab:blue", "tab:green", "tab:red"), cfg["fig7_pu"]):
        rows = jdo.stage_sweep(fluid, tc_k=tc, pu_values_mpa=[pu], tpre_values_k=t_grid, eta_last=1.0, flow_model="none")
        all_rows += rows
        t, qc = _arr(rows, "tpre_last_k"), _arr(rows, "q_c_kj_kg")
        dh, ql = _arr(rows, "delta_h_T_kj_kg"), _arr(rows, "q_L_kj_kg")
        ax.plot(t, dh, "--", color=color, lw=1.2)
        ax.plot(t, ql, ":", color=color, lw=1.2)
        ax.plot(t, np.where(qc > 0, qc, np.nan), "-", color=color, lw=2, label=f"P_U = {pu:g} MPa")
        t_opt = jdo.find_tpre_opt(fluid, tc_k=tc, pu_mpa=pu, tpre_max_k=cfg["fig7_tmax"] + 10)
        t_paper = paper_tpre_opt(pu)
        if t_opt is not None:
            lim = jdo.stage_limits(fluid, tc_k=tc, pu_mpa=pu, tpre_last_k=t_opt)
            ax.plot(t_opt, lim["q_L_j_kg"] / 1000.0, "o", color=color, ms=6)
        summary["per_P_U"].append({"P_U_MPa": pu, "T_pre3_opt_this_work_K": t_opt,
                                   "T_pre3_opt_from_Eq36_K": t_paper,
                                   "difference_K": (t_opt - t_paper) if t_opt is not None and math.isfinite(t_paper) else None})
    _write_table(out / "fig7_qc_deltahT_vs_tpre3.csv", all_rows)
    ax.plot([], [], "k--", label="Δh_T"); ax.plot([], [], "k:", label="q_L"); ax.plot([], [], "k-", lw=2, label="q_C (eta = 1)")
    ax.set(xlabel="T_pre3 [K]", ylabel="specific capacity [kJ/kg]", title=f"Fig. 7 check: T_C = {tc:g} K (dots: Δh_T = q_L)")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(out / "fig7.png", dpi=170); plt.close(fig)
    return summary


def section_flux(fluid, cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """HEM mass flux at stagnation T0 = T_C, refitted to Eq. 49's form (paper Fig. 11)."""
    pu = np.linspace(cfg["flux_pu_min"], cfg["flux_pu_max"], cfg["n_flux"])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.8, 5.4), constrained_layout=True)
    summary: Dict[str, Any] = {"stagnation_temperature": "T0 = T_C", "per_T_C": []}
    all_rows: List[Dict[str, Any]] = []
    for color, t0 in zip(("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"), cfg["flux_temps"]):
        try:
            rows = jdo.mass_flux_vs_pu(fluid, t0_k=t0, pu_values_mpa=pu, hem_grid=cfg["hem_grid"])
        except Exception as exc:
            summary["per_T_C"].append({"T_C_K": t0, "error": f"{type(exc).__name__}: {exc}"})
            continue
        all_rows += rows
        x, g = _arr(rows, "pu_mpa"), _arr(rows, "g_kg_m2_s")
        entry: Dict[str, Any] = {"T_C_K": t0, "n_valid": int(np.isfinite(g).sum()), "n": len(rows)}
        ax.plot(x, g, "o", ms=4, color=color, label=f"HEM, T_C = {t0:g} K")
        try:
            coef, fit_m = fit_power_law(x, g)
            entry["HEM_refit"] = {**coef, "fit_vs_HEM_points": fit_m}
            ax.plot(x, power_law(x, **coef), "--", color=color, lw=1)
        except Exception as exc:
            entry["HEM_refit_error"] = f"{type(exc).__name__}: {exc}"
        key = min(EQ49, key=lambda k: abs(k - t0))
        if math.isclose(key, t0):
            a, b, c = EQ49[key]
            gp = power_law(x, a, b, c)
            entry["paper_Table2"] = {"alpha": a, "beta": b, "gamma": c}
            entry["HEM_vs_paper_curve"] = error_metrics(gp, g)
            ax.plot(x, gp, "-", color=color, lw=2)
        summary["per_T_C"].append(entry)
    _write_table(out / "flux_hem_vs_pu.csv", all_rows)
    ax.plot([], [], "k-", lw=2, label="paper Eq. 49 (Table 2)"); ax.plot([], [], "k--", label="HEM, refit to Eq. 49 form")
    ax.set(xlabel="P_U [MPa]", ylabel="G [kg m$^{-2}$ s$^{-1}$]", title="Fig. 11 check: choked mass flux, T0 = T_C")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(out / "flux.png", dpi=170); plt.close(fig)
    return summary


def section_fig15(fluid, cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """Effect of the last-CHEX effectiveness at the paper's operating point (Figs. 15 and 16)."""
    tc, tp, pu = cfg["tc"], cfg["fig15_tpre"], cfg["fig15_pu"]
    etas = np.asarray(cfg["eta_grid"], dtype=float)
    rows: List[Dict[str, Any]] = []
    for eta in etas:
        r = jdo._safe_evaluate(fluid, tc_k=tc, pu_mpa=pu, tpre_last_k=tp, d_m=cfg["d_m"], eta_last=float(eta),
                               flow_model="hem", hem_grid=cfg["hem_grid"])
        rows.append({"eta4": float(eta), "q_C_kJ_kg": jdo._as_float(r.get("q_c_j_kg")) / 1000.0,
                     "T8_K": jdo._as_float(r.get("t8_k")), "mdot_mg_s": jdo._as_float(r.get("mdot_kg_s")) * 1e6,
                     "Q_C_mW": jdo._as_float(r.get("q_c_w")) * 1000.0, "binding_wall": r.get("binding_wall", ""),
                     "error": r.get("error", "")})
    q1 = next((r["Q_C_mW"] for r in rows if math.isclose(r["eta4"], 1.0) and math.isfinite(r["Q_C_mW"])), math.nan)
    for r in rows:
        r["Q_C_reduction_percent"] = 100.0 * (1.0 - r["Q_C_mW"] / q1) if math.isfinite(q1) else math.nan
    _write_table(out / "fig15_16_eta_sweep.csv", rows)
    e, q = _arr(rows, "eta4"), _arr(rows, "Q_C_mW")
    summary: Dict[str, Any] = {"T_C_K": tc, "T_pre3_K": tp, "P_U_MPa": pu, "d_um": cfg["d_m"] * 1e6,
                               "Q_C_eta1_mW": q1, "mdot_eta1_mg_s": next((r["mdot_mg_s"] for r in rows if math.isclose(r["eta4"], 1.0)), math.nan),
                               "q_C_eta1_kJ_kg": next((r["q_C_kJ_kg"] for r in rows if math.isclose(r["eta4"], 1.0)), math.nan),
                               "paper": {k: PAPER[k] for k in ("Q_C_ideal_mW", "Q_C_eta097_mW", "Q_C_drop_097_percent",
                                                               "drop_per_percent_eta", "mdot_mg_s", "q_C_opt_kJ_kg")}}
    r97 = next((r for r in rows if math.isclose(r["eta4"], PAPER["eta_real"])), None)
    if r97 is not None and math.isfinite(q1):
        summary["Q_C_eta097_mW"] = r97["Q_C_mW"]
        summary["Q_C_drop_097_percent"] = r97["Q_C_reduction_percent"]
        summary["drop_per_percent_eta_(1_to_0.97)"] = r97["Q_C_reduction_percent"] / ((1.0 - PAPER["eta_real"]) * 100.0)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax[0].plot(e, _arr(rows, "q_C_kJ_kg"), "o-", ms=4, color="tab:orange", label="q_C")
    ax[0].set(xlabel="eta_4", ylabel="q_C [kJ/kg]", title="Fig. 15 check")
    ax2 = ax[0].twinx()
    ax2.plot(e, _arr(rows, "mdot_mg_s"), "s-", ms=4, color="tab:blue", label="mdot (HEM)")
    ax2.set_ylabel("mdot [mg/s]")
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(e, _arr(rows, "Q_C_reduction_percent"), "o-", ms=4, color="k", label="this work")
    if tc == PAPER["T_C_K"]:
        ax[1].plot(PAPER["eta_real"], PAPER["Q_C_drop_097_percent"], "g*", ms=14, label="paper: eta = 0.97")
    ax[1].set(xlabel="eta_4", ylabel="Q_C reduction vs eta_4 = 1 [%]", title="Fig. 16 check (relative)")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle(f"Last-CHEX effectiveness: T_C = {tc:g} K, T_pre3 = {tp:g} K, P_U = {pu:g} MPa, d = {cfg['d_m']*1e6:g} um")
    fig.savefig(out / "fig15_16.png", dpi=170); plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Self-tests for the fit helpers (python reproduce_zhang_dang_2021.py --self-test)


class TestFits(unittest.TestCase):
    def test_polynomial_recovers_coefficients_ascending(self):
        t = np.linspace(4, 30, 20)
        true = [-0.102, 0.074, -2.37e-3, 4.88e-5, -4.30e-7]
        coef, m = fit_polynomial(t, polyval_ascending(true, t), 4)
        np.testing.assert_allclose(coef, true, rtol=1e-6, atol=1e-9)
        self.assertLess(m["max_abs_error"], 1e-9)

    def test_plane_fit(self):
        rng = np.random.default_rng(0)
        a, b = rng.uniform(0.1, 1, 30), rng.uniform(4, 30, 30)
        coef, _ = fit_plane(a, b, 11.9 - 11.5 * a + 0.027 * b)
        np.testing.assert_allclose(coef, [11.9, -11.5, 0.027], rtol=1e-8)

    def test_power_law_fit_recovers_paper_form(self):
        p = np.linspace(0.2, 2.0, 25)
        coef, m = fit_power_law(p, power_law(p, 1.26e4, 0.44, -580.0))
        self.assertAlmostEqual(coef["beta"], 0.44, places=4)
        self.assertLess(m["max_rel_percent"], 1e-3)

    def test_metrics_ignore_nonfinite_and_report_count(self):
        m = error_metrics([1.0, 2.0, math.nan, 4.0], [1.1, 2.0, 3.0, 3.6])
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["max_rel_percent"], 10.0)


SECTIONS = {"fits": section_fits, "fig5": section_fig5, "fig7": section_fig7,
            "flux": section_flux, "fig15": section_fig15}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fluid", help="'he3' for the He-3 CSV backend, or a CoolProp fluid name")
    g.add_argument("--fluid-factory", help="no-argument backend as module:ClassName")
    ap.add_argument("--self-test", action="store_true", help="run the built-in unit tests and exit")
    ap.add_argument("--tc", type=float, default=1.0, help="Cooling temperature T_C [K] (default 1)")
    ap.add_argument("--diameter-um", type=float, default=20.0)
    ap.add_argument("--outdir", default="results/validation")
    ap.add_argument("--sections", nargs="+", choices=list(SECTIONS), default=list(SECTIONS))
    ap.add_argument("--flux-temps", nargs="+", type=float, default=[1.0, 2.0, 2.6, 3.0],
                    help="T_C values for the mass-flux section (default: the paper's four)")
    ap.add_argument("--pu-min", type=float, default=0.02, help="lower P_U search bound [MPa]")
    ap.add_argument("--pu-max", type=float, default=1.2, help="upper P_U search bound [MPa]")
    ap.add_argument("--hem-grid", type=int, default=120, help="pressure samples per HEM evaluation")
    ap.add_argument("--quick", action="store_true", help="coarser grids (for testing)")
    a = ap.parse_args(argv)
    if a.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFits)
        return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1
    if not (a.fluid or a.fluid_factory):
        ap.error("one of --fluid or --fluid-factory is required (or use --self-test)")

    fluid, label, is_he3 = jdo._load_fluid(a.fluid, a.fluid_factory)
    q = a.quick
    cfg: Dict[str, Any] = {
        "tc": a.tc, "d_m": a.diameter_um * 1e-6, "hem_grid": a.hem_grid,
        "pu_min": a.pu_min, "pu_max": a.pu_max, "n_curve": 25 if q else 61, "n_opt_grid": 21 if q else 41,
        "tpre_grid": np.linspace(4.0, 30.0, 8 if q else 14),
        "fig5_tpre": 8.0, "fig7_pu": [0.3, 0.5, 0.7, 0.9], "fig7_tmin": 4.0, "fig7_tmax": 32.0,
        "flux_temps": a.flux_temps, "flux_pu_min": 0.2, "flux_pu_max": 2.0, "n_flux": 8 if q else 19,
        "fig15_tpre": 8.0, "fig15_pu": 0.36,
        "eta_grid": np.round(np.arange(0.85, 1.0001, 0.03 if q else 0.01), 4) if not q else [0.85, 0.91, 0.97, 1.0],
    }
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    comparable = bool(is_he3 and math.isclose(a.tc, PAPER["T_C_K"]) and math.isclose(a.diameter_um, PAPER["d_um"]))
    report: Dict[str, Any] = {"fluid_backend": label, "T_C_K": a.tc, "d_um": a.diameter_um,
                              "paper_comparison_meaningful": comparable,
                              "note": ("Paper numbers are for He-3, T_C = 1 K, d = 20 um." if comparable else
                                       "NOT He-3 at T_C = 1 K, d = 20 um: paper comparisons are not meaningful; machinery test only."),
                              "sections": {}}
    for name in a.sections:
        try:
            report["sections"][name] = SECTIONS[name](fluid, cfg, out)
            print(f"[ok] {name}")
        except Exception as exc:
            report["sections"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[failed] {name}: {type(exc).__name__}: {exc}")
    (out / "reproduction_report.json").write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    print(f"Report: {out / 'reproduction_report.json'}")
    if not comparable:
        print(report["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
