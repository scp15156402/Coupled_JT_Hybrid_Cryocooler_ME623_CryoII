"""Paper-reproduction harness for Zhang & Dang (Cryogenics 116, 2021, 103282).

This file treats ``jt_cycle_core.py`` as a black-box thermodynamic engine. It
contains paper reference correlations, parameter sweeps, plots, and comparison
metrics. It is intentionally separate from the core solver.

IMPORTANT
---------
* The reference-only mode needs NumPy and Matplotlib, but no fluid-property
  model. It recreates the published fitted curves from the paper's coefficients.
* Full thermodynamic reproduction needs an instantiated fluid backend satisfying
  the ``FluidTable`` interface in jt_cycle_core.py. The backend must represent
  the actual fluid and support the required single/two-phase states.
* The cycle core currently uses HEM for restriction flow. The paper's fitted
  mass-flux correlation is included separately, so HEM and paper-flow results
  can be compared instead of being silently treated as equivalent.
* This script does not invent helium-3 results when no valid backend is supplied.

Published correlations used
---------------------------
Eq. (35), with P_U,opt in MPa, T_pre3 in K and q_C,opt in kJ/kg:
    q_C,opt = 11.9 - 11.5 P_U,opt + 0.027 T_pre3

Eq. (36): fourth-order polynomial P_U,opt(T_pre3), in MPa.
Table 1 coefficients are stored in ascending powers of T_pre3.

Eq. (49): G_i(P_U) = alpha_i * P_U**beta_i + gamma_i, with P_U in MPa and
G in kg/(m^2 s). The tabulated scalings are interpreted as alpha = table_value
* 1e4 and gamma = table_value * 1e3; this reproduces the paper's reference
mass flow (2.34 mg/s at d=20 um, P_U=0.36 MPa, T0=1 K).

Typical use after a fluid backend is available:
    from helium3_properties import Helium3Fluid
    from reproduce_zhang_dang_2021 import run_full_reproduction

    fluid = Helium3Fluid()
    run_full_reproduction(
        fluid,
        output_dir="paper_reproduction",
        tpre12=(T_pre1, T_pre2),   # K; values for the first two PHEXs
    )

Reference-only mode:
    python reproduce_zhang_dang_2021.py --reference-only --outdir paper_reference
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from jt_cycle_core import CycleResult, FluidTable, critical_mass_flux_HEM, solve_cycle, state_labels


# ---------------------------------------------------------------------------
# Published reference data and correlations

@dataclass(frozen=True)
class PaperReference:
    T_C_K: float = 1.0
    P_D_Pa: float = 1.16e3
    T_pre3_opt_K: float = 8.0
    P_U_opt_MPa: float = 0.36
    q_C_opt_kJ_kg: float = 8.0
    d_um: float = 20.0
    mdot_mg_s: float = 2.34
    Q_C_ideal_mW: float = 18.7
    eta4_real: float = 0.97
    Q_C_eta4_097_mW: float = 14.7
    P_U_max_MPa: float = 1.13


PAPER = PaperReference()

# Eq. (36): P_U,opt [MPa] = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4.
PAPER_PU_OPT_POLY_ASC = np.array(
    [-0.102, 0.074, -2.37e-3, 4.88e-5, -4.30e-7], dtype=float
)

# Table 2 / Eq. (49). The coefficients are stored after applying the table's
# power-of-ten scalings, with P_U in MPa and G in kg/(m^2 s).
# The tabulated temperature corresponds to T_C in the paper's formulation;
# Section 4 approximates restriction inlet T0 by T_C for those curves.
PAPER_MASS_FLUX_FITS: Dict[float, Dict[str, float]] = {
    1.0: {"alpha": 1.26e4, "beta": 0.44, "gamma": -0.58e3},
    2.0: {"alpha": 1.40e4, "beta": 0.40, "gamma": -2.15e3},
    2.6: {"alpha": 1.59e4, "beta": 0.36, "gamma": -4.36e3},
    3.0: {"alpha": 1.82e4, "beta": 0.32, "gamma": -6.91e3},
}

PAPER_FIGURE_TARGETS = {
    "fig01": "Simplified system schematic; figure 1",
    "fig02_03": "P-h cycle diagrams; figures 2-3",
    "fig04": "Isobaric specific heat versus temperature; figure 4",
    "fig05": "Specific refrigeration versus P_U at T_pre3=8 K; figure 5",
    "fig06": "Last-stage cycle states versus T_pre3; figure 6",
    "fig07": "Specific refrigeration versus T_pre3 for several P_U; figure 7",
    "fig08": "Published and independently recalculated optimum fits; figure 8",
    "fig09": "Restriction isentrope / HEM critical point; schematic counterpart to figure 9",
    "fig10": "Mass-flux deviation factor; figure 10 (needs gamma and R_specific)",
    "fig11": "Published mass-flux correlations and HEM comparison; figure 11",
    "fig12": "Gross cooling capacity versus P_U at T_pre3=8 K; figure 12",
    "fig13": "Gross cooling capacity versus P_U for several T_pre3; figure 13",
    "fig14": "Last-stage CHEX terminal temperatures versus eta4; figure 14",
    "fig15": "Specific refrigeration and mass flow versus eta4; figure 15",
    "fig16": "Absolute and relative Q_C reduction versus eta4; figure 16",
}


def paper_pu_opt_mpa(T_pre3_K: Any) -> Any:
    """Published Eq. (36), P_U,opt [MPa] as a function of T_pre3 [K]."""
    return np.polynomial.polynomial.polyval(T_pre3_K, PAPER_PU_OPT_POLY_ASC)


def paper_qc_opt_kj_kg(P_U_opt_MPa: Any, T_pre3_K: Any) -> Any:
    """Published Eq. (35): q_C,opt [kJ/kg]."""
    return 11.9 - 11.5 * np.asarray(P_U_opt_MPa) + 0.027 * np.asarray(T_pre3_K)


def paper_mass_flux_at_temperature(T0_K: float, P_U_MPa: Any) -> Any:
    """Evaluate Eq. (49) from Table 2.

    The paper tabulates coefficients at T_C = 1.0, 2.0, 2.6 and 3.0 K and
    approximates the restriction inlet stagnation temperature by T_C. For a
    T0 between those table temperatures, this helper linearly interpolates the
    *computed G values* between adjacent fitted curves. That interpolation is a
    convenience for comparing the paper's fit with the current cycle's state 8;
    it is not itself a published correlation. Extrapolation is rejected.
    """
    temps = np.array(sorted(PAPER_MASS_FLUX_FITS), dtype=float)
    if not (temps[0] <= float(T0_K) <= temps[-1]):
        raise ValueError(
            f"Paper mass-flux fit supports T0 in [{temps[0]}, {temps[-1]}] K; "
            f"got {T0_K}. Do not extrapolate silently."
        )
    p = np.asarray(P_U_MPa, dtype=float)
    if np.any((p < 0.2) | (p > 2.0)):
        raise ValueError("The paper's Eq. (49) fit is only used here for 0.2 <= P_U <= 2.0 MPa")
    lo = max(t for t in temps if t <= T0_K)
    hi = min(t for t in temps if t >= T0_K)

    def one(t: float) -> Any:
        c = PAPER_MASS_FLUX_FITS[float(t)]
        return c["alpha"] * p ** c["beta"] + c["gamma"]

    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
        return one(float(lo))
    frac = (T0_K - lo) / (hi - lo)
    return (1.0 - frac) * one(float(lo)) + frac * one(float(hi))


# ---------------------------------------------------------------------------
# Shared data utilities


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("Plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc
    return plt


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        # Stable order: keys in first row first, followed by any later keys.
        keys: List[str] = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def error_metrics(reference: Sequence[float], predicted: Sequence[float]) -> Dict[str, float]:
    """Error/correlation metrics for paired curves, excluding non-finite pairs."""
    y_ref = np.asarray(reference, dtype=float).ravel()
    y_pred = np.asarray(predicted, dtype=float).ravel()
    if y_ref.shape != y_pred.shape:
        raise ValueError("reference and predicted arrays must have identical shapes")
    mask = np.isfinite(y_ref) & np.isfinite(y_pred)
    y_ref, y_pred = y_ref[mask], y_pred[mask]
    if y_ref.size == 0:
        return {"n": 0, "rmse": math.nan, "mae": math.nan, "max_abs_error": math.nan,
                "mape_percent": math.nan, "r_squared": math.nan, "pearson_r": math.nan}
    err = y_pred - y_ref
    nonzero = np.abs(y_ref) > np.finfo(float).eps
    if y_ref.size > 1 and np.std(y_ref) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_ref, y_pred)[0, 1])
    else:
        pearson = math.nan
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else math.nan
    return {
        "n": int(y_ref.size),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "mape_percent": float(np.mean(np.abs(err[nonzero] / y_ref[nonzero])) * 100.0) if np.any(nonzero) else math.nan,
        "r_squared": float(r2),
        "pearson_r": pearson,
    }


def compare_to_paper_fit(
    x: Sequence[float],
    independently_calculated: Sequence[float],
    paper_fit: Callable[[Any], Any],
) -> Dict[str, float]:
    """Compare independently calculated values with the paper fit at same x."""
    x_arr = np.asarray(x, dtype=float)
    return error_metrics(np.asarray(paper_fit(x_arr), dtype=float), independently_calculated)


def fit_polynomial_and_report(
    x: Sequence[float], y: Sequence[float], degree: int = 4
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fit a polynomial and return coefficients in ascending-power order + fit metrics."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size <= degree:
        raise ValueError(f"Need at least {degree + 1} finite points for degree-{degree} fit")
    coef_desc = np.polyfit(xa, ya, degree)
    prediction = np.polyval(coef_desc, xa)
    metrics = error_metrics(ya, prediction)
    return coef_desc[::-1], metrics


def fit_mass_flux_power_law(
    P_U_MPa: Sequence[float], G: Sequence[float]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Fit G=alpha*P_U**beta+gamma with scipy (optional)."""
    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Nonlinear mass-flux fitting requires scipy") from exc
    x, y = np.asarray(P_U_MPa, dtype=float), np.asarray(G, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[mask], y[mask]
    if x.size < 4:
        raise ValueError("Need at least four valid points to fit the three-parameter mass-flux curve")

    def model(p: Any, alpha: float, beta: float, gamma: float) -> Any:
        return alpha * p**beta + gamma

    guess = (max(float(np.max(y)), 1.0), 0.4, min(float(np.min(y)), 0.0))
    params, _ = curve_fit(model, x, y, p0=guess, maxfev=100000)
    fitted = model(x, *params)
    return ({"alpha": float(params[0]), "beta": float(params[1]), "gamma": float(params[2])},
            error_metrics(y, fitted))


# ---------------------------------------------------------------------------
# Reference-only figures; no fluid model required


def make_reference_only_outputs(output_dir: str | Path) -> Dict[str, Path]:
    """Create figures/data that can be reconstructed from paper fit coefficients alone."""
    out = _ensure_output_dir(output_dir)
    plt = _mpl()
    paths: Dict[str, Path] = {}

    # Figure 8: reconstruct all three pairwise relationships represented by fits.
    t = np.linspace(4.0, 34.0, 301)
    pu_opt = paper_pu_opt_mpa(t)
    qc_opt = paper_qc_opt_kj_kg(pu_opt, t)
    rows = [
        {"T_pre3_K": float(tt), "P_U_opt_MPa_paper_fit": float(pp), "q_C_opt_kJ_kg_paper_fit": float(qq)}
        for tt, pp, qq in zip(t, pu_opt, qc_opt)
    ]
    csv_path = out / "reference_figure08_published_fits.csv"
    _write_csv(csv_path, rows)
    paths["figure08_csv"] = csv_path

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].plot(t, pu_opt, linewidth=2)
    axes[0].axhline(PAPER.P_U_opt_MPa, linestyle="--", linewidth=1, label="Paper reference: 0.36 MPa")
    axes[0].set(xlabel=r"$T_{pre3}$ [K]", ylabel=r"$P_{U,opt}$ [MPa]", title="Eq. (36): optimum pressure")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, qc_opt, linewidth=2)
    axes[1].axhline(PAPER.q_C_opt_kJ_kg, linestyle="--", linewidth=1, label="Paper reference: 8 kJ/kg")
    axes[1].set(xlabel=r"$T_{pre3}$ [K]", ylabel=r"$q_{C,opt}$ [kJ/kg]", title="Eq. (35): optimum refrigeration")
    axes[1].legend(fontsize=8)
    axes[2].plot(pu_opt, qc_opt, linewidth=2)
    axes[2].set(xlabel=r"$P_{U,opt}$ [MPa]", ylabel=r"$q_{C,opt}$ [kJ/kg]", title="Optimum curve parameterized by $T_{pre3}$")
    fig.suptitle("Figure 8 — reconstruction of published fitted functions (not an independent calculation)")
    path = out / "reference_figure08_fits.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["figure08_png"] = path

    # Figure 11: published Table 2 curves.
    p = np.linspace(0.2, 2.0, 400)
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for t0 in sorted(PAPER_MASS_FLUX_FITS):
        G = paper_mass_flux_at_temperature(t0, p)
        ax.plot(p, G, linewidth=2, label=f"T0 ≈ TC = {t0:g} K")
        rows.extend({"T0_approx_K": t0, "P_U_MPa": float(pp), "G_paper_fit_kg_m2_s": float(gg)}
                    for pp, gg in zip(p, G))
    ax.set(xlabel=r"$P_U$ [MPa]", ylabel=r"$G$ [kg m$^{-2}$ s$^{-1}$]",
           title="Figure 11 — published mass-flux fits from Eq. (49), Table 2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = out / "reference_figure11_mass_flux_fits.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["figure11_png"] = path
    csv_path = out / "reference_figure11_mass_flux_fits.csv"
    _write_csv(csv_path, rows)
    paths["figure11_csv"] = csv_path

    # Small reference validation ledger.
    pu_ref = float(PAPER.P_U_opt_MPa)
    tpre_ref = float(PAPER.T_pre3_opt_K)
    pu_from_fit = float(paper_pu_opt_mpa(tpre_ref))
    qc_from_fit = float(paper_qc_opt_kj_kg(pu_from_fit, tpre_ref))
    G_at_ref = float(paper_mass_flux_at_temperature(PAPER.T_C_K, pu_ref))
    area = math.pi / 4.0 * (PAPER.d_um * 1e-6) ** 2
    mdot_fit = area * G_at_ref
    qc_gross_fit = qc_from_fit * 1000.0 * mdot_fit
    ledger = [
        {"quantity": "P_U_opt_from_Eq36", "computed": pu_from_fit, "paper_reference": PAPER.P_U_opt_MPa, "unit": "MPa"},
        {"quantity": "q_C_opt_from_Eq35", "computed": qc_from_fit, "paper_reference": PAPER.q_C_opt_kJ_kg, "unit": "kJ/kg"},
        {"quantity": "G_from_Eq49_at_reference", "computed": G_at_ref, "paper_reference": PAPER.mdot_mg_s * 1e-6 / area, "unit": "kg/(m^2 s)"},
        {"quantity": "mdot_from_Eq49_d20um", "computed": mdot_fit * 1e6, "paper_reference": PAPER.mdot_mg_s, "unit": "mg/s"},
        {"quantity": "Q_C_from_Eq35_Eq49", "computed": qc_gross_fit * 1e3, "paper_reference": PAPER.Q_C_ideal_mW, "unit": "mW"},
    ]
    csv_path = out / "reference_fit_sanity_check.csv"
    _write_csv(csv_path, ledger)
    paths["reference_ledger"] = csv_path

    # Figure 1 is a conceptual schematic and requires no fluid-property model.
    paths["fig01"] = _plot_system_schematic(out)

    report = {
        "mode": "reference_only",
        "fluid_properties_used": False,
        "published_fits_reconstructed": ["Eq. 35", "Eq. 36", "Eq. 49 / Table 2"],
        "reference_schematic_created": ["fig01"],
        "published_fit_curves_reconstructed": ["fig08", "fig11"],
        "not_independently_reproduced_from_thermodynamic_properties": list(PAPER_FIGURE_TARGETS.keys()),
        "note": "The schematic is a simplified redraw. Figure 8 and Figure 11 curves are reconstructed from published coefficients; they are not independent thermodynamic predictions.",
        "files": {k: str(v) for k, v in paths.items()},
    }
    report_path = out / "reference_only_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["report"] = report_path
    return paths


def _plot_system_schematic(output_dir: Path) -> Path:
    """Draw a simplified process schematic; it is not a mechanical CAD drawing."""
    plt = _mpl()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, label: str, color: str = "white") -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1.2,
                               edgecolor="black", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9)

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12,
                                     linewidth=1.2, color="black"))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, label, ha="center", fontsize=8)

    # High-pressure stream, warm to cold.
    box(0.25, 4.7, 1.35, 0.8, "Warm inlet\nstate 1", "#f5f5f5")
    box(2.05, 4.7, 1.5, 0.8, "CHEX 1\nstates 1–2")
    box(4.0, 4.7, 1.5, 0.8, "CHEX 2\nstates 3–4")
    box(5.95, 4.7, 1.5, 0.8, "CHEX 3\nstates 5–6")
    box(7.9, 4.7, 1.5, 0.8, "CHEX 4\nstates 7–8")
    box(10.0, 4.7, 1.45, 0.8, "JT restriction\n8 → 9")
    arrow(1.6, 5.1, 2.05, 5.1)
    arrow(3.55, 5.1, 4.0, 5.1, "PHEX 1")
    arrow(5.5, 5.1, 5.95, 5.1, "PHEX 2")
    arrow(7.45, 5.1, 7.9, 5.1, "PHEX 3")
    arrow(9.4, 5.1, 10.0, 5.1)

    # Evaporator and low-pressure return stream.
    box(10.05, 2.65, 1.35, 0.8, "Evaporator\n9 → 10", "#e8f4ff")
    arrow(10.72, 4.7, 10.72, 3.45)
    ax.text(11.12, 4.05, r"$Q_C$", fontsize=10)
    arrow(10.05, 3.05, 9.5, 3.05)
    arrow(9.5, 3.05, 9.5, 2.2)
    arrow(9.5, 2.2, 8.65, 2.2)
    box(7.65, 1.75, 1.0, 0.8, "Return\n11")
    arrow(7.65, 2.15, 6.8, 2.15)
    box(5.8, 1.75, 1.0, 0.8, "Return\n12")
    arrow(5.8, 2.15, 4.95, 2.15)
    box(3.95, 1.75, 1.0, 0.8, "Return\n13")
    arrow(3.95, 2.15, 3.1, 2.15)
    box(2.1, 1.75, 1.0, 0.8, "Return\n14")
    arrow(2.1, 2.15, 1.2, 2.15)
    arrow(1.2, 2.15, 1.2, 3.15)
    arrow(1.2, 3.15, 0.9, 3.15)
    box(0.25, 2.7, 1.3, 0.9, "Return to\ncompressor", "#f5f5f5")

    # Thermal links from PHEX stages to a generic multistage cooler.
    box(3.8, 0.25, 4.6, 0.75, "External multistage cooler (not modelled in the baseline)", "#fff6df")
    for x in (3.78, 5.73, 7.68):
        arrow(x, 4.65, x, 3.9)
        ax.text(x + 0.08, 4.15, r"$Q_{pre}$", fontsize=8, rotation=90, va="center")
    ax.text(6.1, 6.3, "Simplified process schematic for paper reproduction", ha="center", fontsize=13, weight="bold")
    ax.text(6.1, 5.95, "Conceptual flow connections only; not a geometric reproduction of the apparatus", ha="center", fontsize=9)
    path = output_dir / "fig01_system_schematic.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Fluid-backed calculations. These call the core as a black box.


def _solve_point(
    fluid: FluidTable,
    *,
    Tc: float,
    Pu_MPa: float,
    Th: float,
    tpre12: Tuple[float, float],
    Tpre3: float,
    eta: Sequence[float],
    d: Optional[float] = None,
) -> CycleResult:
    if len(eta) != 4:
        raise ValueError("The Zhang & Dang paper-reproduction cases use four CHEXs")
    tpre = [float(tpre12[0]), float(tpre12[1]), float(Tpre3)]
    return solve_cycle(fluid, Tc=Tc, Pu=Pu_MPa * 1e6, Th=Th, Tpre=tpre, eta=eta, d=d)


def _paper_flow_power(result: CycleResult, Pu_MPa: float, d_m: float) -> Dict[str, float]:
    """Use paper's Table 2 mass-flux fit at the current state-8 temperature.

    T0 is taken from state 8, matching the paper's discussion of T0≈T8. The
    interpolation between the tabulated T0 curves is explicitly an added
    approximation, not part of Eq. (49) itself.
    """
    labels = state_labels(4)
    t0 = result.states[int(labels["JT_in"])].T
    G = float(paper_mass_flux_at_temperature(t0, Pu_MPa))
    if G <= 0:
        raise ValueError(f"Paper mass-flux fit returned nonpositive G={G:g}")
    mdot = math.pi / 4.0 * d_m**2 * G  # Cd=1, as in the paper's equation
    return {
        "T0_fit_K": t0,
        "G_paper_fit": G,
        "mdot_paper_fit_kg_s": mdot,
        "Q_C_paper_fit_W": mdot * result.q_C,
    }


def _result_row(result: CycleResult, Pu_MPa: float, Tpre3: float, eta4: float, flow_mode: str) -> Dict[str, Any]:
    return {
        "T_C_K": None,
        "P_U_MPa": Pu_MPa,
        "T_pre3_K": Tpre3,
        "eta4": eta4,
        "q_C_J_kg": result.q_C,
        "q_C_kJ_kg": result.q_C / 1000.0,
        "Q_C_W": result.Q_C,
        "Q_C_mW": result.Q_C * 1000.0 if result.Q_C is not None else None,
        "mdot_kg_s": result.mdot,
        "mdot_mg_s": result.mdot * 1e6 if result.mdot is not None else None,
        "q_pre_J_kg": result.q_pre,
        "Q_pre_W": result.Q_pre,
        "P_D_Pa": result.P_D,
        "T8_K": result.states[8].T if 8 in result.states else None,
        "T11_K": result.states[11].T if 11 in result.states else None,
        "x_l_JT_outlet": result.x_l_JT_outlet,
        "flow_mode": flow_mode,
        "flags": result.flags,
    }


def _cycle_for_mode(
    fluid: FluidTable,
    *,
    Tc: float,
    Pu_MPa: float,
    Th: float,
    tpre12: Tuple[float, float],
    Tpre3: float,
    eta: Sequence[float],
    d_m: float,
    flow_mode: str,
) -> Tuple[CycleResult, Dict[str, float]]:
    """Run the core once, using HEM flow or post-processing with paper's G fit."""
    if flow_mode not in {"hem", "paper_fit"}:
        raise ValueError("flow_mode must be 'hem' or 'paper_fit'")
    if flow_mode == "hem":
        result = _solve_point(fluid, Tc=Tc, Pu_MPa=Pu_MPa, Th=Th, tpre12=tpre12,
                              Tpre3=Tpre3, eta=eta, d=d_m)
        flow = {
            "T0_fit_K": result.states[8].T,
            "G_hem": float(result.throat["G"]) if result.throat else math.nan,
            "mdot_hem_kg_s": result.mdot if result.mdot is not None else math.nan,
            "Q_C_hem_W": result.Q_C if result.Q_C is not None else math.nan,
            "throat_phase": result.throat["phase"] if result.throat else "unknown",
        }
        return result, flow

    # For paper-fit mass flow, ask the core only for the thermodynamic cycle
    # and then use the published G(P_U,T0) fit as a separate post-process.
    result = _solve_point(fluid, Tc=Tc, Pu_MPa=Pu_MPa, Th=Th, tpre12=tpre12,
                          Tpre3=Tpre3, eta=eta, d=None)
    flow = _paper_flow_power(result, Pu_MPa, d_m)
    return result, flow


def _safe_call(call: Callable[[], Any]) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return call(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _sweep_specific_qc(
    fluid: FluidTable,
    *,
    Tc: float,
    Th: float,
    tpre12: Tuple[float, float],
    Tpre3_values: Sequence[float],
    Pu_values_MPa: Sequence[float],
    eta: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for Tpre3 in Tpre3_values:
        for pu in Pu_values_MPa:
            result, err = _safe_call(lambda: _solve_point(
                fluid, Tc=Tc, Pu_MPa=float(pu), Th=Th, tpre12=tpre12,
                Tpre3=float(Tpre3), eta=eta, d=None
            ))
            if err or result is None:
                rows.append({"T_C_K": Tc, "P_U_MPa": float(pu), "T_pre3_K": float(Tpre3),
                             "q_C_J_kg": math.nan, "q_C_kJ_kg": math.nan,
                             "error": err, "flags": []})
            else:
                row = _result_row(result, float(pu), float(Tpre3), float(eta[-1]), "specific_only")
                row["T_C_K"] = Tc
                row["error"] = ""
                rows.append(row)
    return rows


def _sweep_capacity(
    fluid: FluidTable,
    *,
    Tc: float,
    Th: float,
    tpre12: Tuple[float, float],
    Tpre3_values: Sequence[float],
    Pu_values_MPa: Sequence[float],
    eta: Sequence[float],
    d_m: float,
    flow_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for Tpre3 in Tpre3_values:
        for pu in Pu_values_MPa:
            outcome, err = _safe_call(lambda: _cycle_for_mode(
                fluid, Tc=Tc, Pu_MPa=float(pu), Th=Th, tpre12=tpre12,
                Tpre3=float(Tpre3), eta=eta, d_m=d_m, flow_mode=flow_mode
            ))
            if err or outcome is None:
                rows.append({"T_C_K": Tc, "P_U_MPa": float(pu), "T_pre3_K": float(Tpre3),
                             "flow_mode": flow_mode, "q_C_kJ_kg": math.nan,
                             "Q_C_mW": math.nan, "G_kg_m2_s": math.nan,
                             "mdot_mg_s": math.nan, "error": err})
                continue
            result, flow = outcome
            G_key = "G_hem" if flow_mode == "hem" else "G_paper_fit"
            mdot_key = "mdot_hem_kg_s" if flow_mode == "hem" else "mdot_paper_fit_kg_s"
            qc_key = "Q_C_hem_W" if flow_mode == "hem" else "Q_C_paper_fit_W"
            rows.append({
                "T_C_K": Tc, "P_U_MPa": float(pu), "T_pre3_K": float(Tpre3),
                "eta4": float(eta[-1]), "flow_mode": flow_mode,
                "q_C_J_kg": result.q_C, "q_C_kJ_kg": result.q_C / 1000.0,
                "G_kg_m2_s": flow.get(G_key, math.nan),
                "mdot_kg_s": flow.get(mdot_key, math.nan),
                "mdot_mg_s": flow.get(mdot_key, math.nan) * 1e6,
                "Q_C_W": flow.get(qc_key, math.nan), "Q_C_mW": flow.get(qc_key, math.nan) * 1000,
                "T8_K": result.states[8].T, "T11_K": result.states[11].T,
                "x_l_JT_outlet": result.x_l_JT_outlet,
                "flags": result.flags, "error": "",
            })
    return rows


def _finite_rows(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str,
                 filter_fn: Optional[Callable[[Mapping[str, Any]], bool]] = None) -> Tuple[np.ndarray, np.ndarray]:
    selected = []
    for row in rows:
        if filter_fn is not None and not filter_fn(row):
            continue
        try:
            xv, yv = float(row[x_key]), float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(xv) and np.isfinite(yv):
            selected.append(row)
    return np.array([float(r[x_key]) for r in selected]), np.array([float(r[y_key]) for r in selected])


def _plot_qc_vs_pu(rows: Sequence[Mapping[str, Any]], output_path: Path, *, title: str, y_key: str = "q_C_kJ_kg") -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    temps = sorted({float(r["T_pre3_K"]) for r in rows if r.get("T_pre3_K") is not None})
    for temp in temps:
        x, y = _finite_rows(rows, "P_U_MPa", y_key, lambda r: math.isclose(float(r["T_pre3_K"]), temp))
        if x.size:
            order = np.argsort(x)
            ax.plot(x[order], y[order], label=f"Tpre3={temp:g} K", linewidth=1.8)
    ax.set(xlabel=r"$P_U$ [MPa]", ylabel=(r"$q_C$ [kJ/kg]" if y_key == "q_C_kJ_kg" else r"$Q_C$ [mW]"), title=title)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_ph(states: Mapping[int, Any], output_path: Path, title: str, state_numbers: Optional[Sequence[int]] = None) -> None:
    plt = _mpl()
    if state_numbers is None:
        state_numbers = list(range(1, 15))
    available = [i for i in state_numbers if i in states]
    if len(available) < 2:
        return
    h = np.array([states[i].h / 1000.0 for i in available])
    p = np.array([states[i].P / 1e6 for i in available])
    fig, ax = plt.subplots(figsize=(7.5, 5.3), constrained_layout=True)
    ax.plot(h, p, "o-", linewidth=1.4, markersize=3)
    for i, hi, pi_mpa in zip(available, h, p):
        ax.annotate(str(i), (hi, pi_mpa), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set(xlabel=r"Specific enthalpy $h$ [kJ/kg]", ylabel=r"Pressure $P$ [MPa]", title=title)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _call_cp(fluid: Any, P: float, T: float) -> float:
    """Call one of the conventional cp method names on an optional backend."""
    for name in ("cp", "cp_mass", "Cp"):
        method = getattr(fluid, name, None)
        if callable(method):
            return float(method(P, T))
    raise AttributeError("Fluid backend has no cp(P,T) or cp_mass(P,T) method")


def _plot_cp_if_available(fluid: Any, out: Path, Tc: float) -> Optional[Path]:
    plt = _mpl()
    pressures_pa = [(1.0e3, "1 kPa"), (2.0e3, "2 kPa"), (4.0e5, "400 kPa"), (5.0e5, "500 kPa")]
    temperatures = np.linspace(1.0, 10.0, 160)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for pressure, label in pressures_pa:
        # Retain NaN where the backend's domain excludes a state (e.g.
        # subcritical saturation ambiguity).
        cp = []
        for temp in temperatures:
            try:
                cp.append(_call_cp(fluid, pressure, float(temp)) / 1000.0)  # kJ/(kg K)
            except Exception:
                cp.append(math.nan)
        if np.count_nonzero(np.isfinite(cp)) > 1:
            ax.plot(temperatures, cp, linewidth=1.7, label=label)
    if not ax.lines:
        plt.close(fig)
        return None
    ax.set(xlabel="Temperature [K]", ylabel=r"$C_P$ [kJ/(kg K)]", title="Figure 4 — isobaric specific heat")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = out / "fig04_cp_vs_temperature.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _restriction_path_plot(fluid: FluidTable, out: Path, *, Pu_MPa: float, T0: float, Pback: float) -> Optional[Path]:
    plt = _mpl()
    P0 = Pu_MPa * 1e6
    s0 = fluid.s(P0, T0)
    pressures = np.geomspace(Pback, P0, 350)[::-1]
    hs, ts, gs = [], [], []
    h0 = fluid.h(P0, T0)
    for P in pressures:
        try:
            h = fluid.h_from_Ps(float(P), s0)
            T = fluid.T_from_Ph(float(P), h)
            rho = fluid.rho_from_Ps(float(P), s0)
            u = math.sqrt(max(0.0, 2.0 * (h0 - h)))
            hs.append(h / 1000.0)
            ts.append(T)
            gs.append(rho * u)
        except Exception:
            hs.append(math.nan); ts.append(math.nan); gs.append(math.nan)
    try:
        throat = critical_mass_flux_HEM(P0, T0, Pback, fluid)
    except Exception:
        throat = None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].plot(hs, pressures / 1e6, linewidth=1.8)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Enthalpy [kJ/kg]", ylabel="Pressure [MPa]", title="Isentropic expansion path")
    axes[1].plot(pressures / 1e6, gs, linewidth=1.8)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Pressure [MPa]", ylabel=r"Mass flux $G$ [kg/(m$^2$s)]", title="HEM mass-flux curve")
    if throat:
        for ax in axes:
            if ax is axes[0]:
                ax.axhline(throat["P_star"] / 1e6, linestyle="--", alpha=0.8, label="HEM maximum")
                ax.legend()
            else:
                ax.axvline(throat["P_star"] / 1e6, linestyle="--", alpha=0.8, label="HEM maximum")
                ax.legend()
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("Figure 9 — restriction expansion (HEM counterpart; not the paper's sonic-method calculation)")
    path = out / "fig09_restriction_expansion_HEM.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_deviation_if_possible(fluid: Any, out: Path, pu_grid: Sequence[float], t0_values: Sequence[float]) -> Optional[Path]:
    """Plot Gamma=G_HEM/G_ID only when backend supplies gamma and R_specific."""
    gamma_method = getattr(fluid, "gamma", None)
    Rsp = getattr(fluid, "R_specific", getattr(fluid, "gas_constant_specific", None))
    if not callable(gamma_method) or Rsp is None:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    any_curve = False
    for T0 in t0_values:
        xs, ys = [], []
        Pback_result, err = _safe_call(lambda: fluid.Psat(T0))
        if err or Pback_result is None:
            continue
        Pback = float(Pback_result)
        for pu in pu_grid:
            P0 = float(pu) * 1e6
            if P0 <= Pback:
                continue
            result, err = _safe_call(lambda p=P0, t=T0, pb=Pback: critical_mass_flux_HEM(p, t, pb, fluid))
            if err or result is None:
                continue
            try:
                gamma = float(gamma_method(P0, T0))
                r_spec = float(Rsp() if callable(Rsp) else Rsp)
                # Ideal-gas critical mass flux, expressed with specific gas constant.
                factor = math.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
                G_ideal = factor * math.sqrt(P0 / (r_spec * T0))
                xs.append(pu); ys.append(result["G"] / G_ideal)
            except Exception:
                continue
        if len(xs) > 1:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], label=f"T0={T0:g} K")
            any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.set(xlabel=r"$P_U$ [MPa]", ylabel=r"$\Gamma=G_{\mathrm{HEM}}/G_{\mathrm{ID}}$", title="Figure 10 — HEM-to-ideal mass-flux ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = out / "fig10_deviation_factor_HEM.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_full_reproduction(
    fluid: FluidTable,
    *,
    output_dir: str | Path,
    tpre12: Tuple[float, float],
    Th: float = 300.0,
    d_m: float = 20e-6,
    Tc: float = 1.0,
    pu_grid_mpa: Optional[Sequence[float]] = None,
    tpre3_grid_K: Optional[Sequence[float]] = None,
    eta4_grid: Optional[Sequence[float]] = None,
    figure11_compare_hem: bool = True,
) -> Dict[str, Path]:
    """Run fluid-backed paper reproduction calculations and save figures/CSVs.

    ``tpre12`` are the first two PHEX outlet temperatures required by the core.
    For plots focused exclusively on the final CHEX (with eta1=eta2=eta3=1),
    the paper's ideal-cycle equations make q_C and the final-stage states
    independent of Tpre1/Tpre2; they are still needed to populate the complete
    four-CHEX core cycle.
    """
    out = _ensure_output_dir(output_dir)
    paths = make_reference_only_outputs(out)
    plt = _mpl()

    if pu_grid_mpa is None:
        pu_grid_mpa = np.linspace(0.20, 1.10, 61)
    if tpre3_grid_K is None:
        tpre3_grid_K = np.linspace(4.0, 30.0, 66)
    if eta4_grid is None:
        eta4_grid = np.linspace(0.90, 1.00, 21)
    pu_grid = np.asarray(pu_grid_mpa, dtype=float)
    tpre3_grid = np.asarray(tpre3_grid_K, dtype=float)
    eta_grid = np.asarray(eta4_grid, dtype=float)
    d_um = d_m * 1e6
    eta_ideal = (1.0, 1.0, 1.0, 1.0)

    # Figures 2-3: complete cycle and final CHEX/JT/evaporator focus.
    base, err = _safe_call(lambda: _solve_point(
        fluid, Tc=Tc, Pu_MPa=PAPER.P_U_opt_MPa, Th=Th, tpre12=tpre12,
        Tpre3=PAPER.T_pre3_opt_K, eta=eta_ideal, d=None
    ))
    if err is None and base is not None:
        _plot_ph(base.states, out / "fig02_ph_cycle.png", "Figure 2 — complete JT cycle in P-h plane")
        paths["fig02"] = out / "fig02_ph_cycle.png"
        _plot_ph(base.states, out / "fig03_last_stage_ph.png", "Figure 3 — final CHEX, JT restriction and evaporator", [7, 8, 9, 10, 11])
        paths["fig03"] = out / "fig03_last_stage_ph.png"
    else:
        (out / "fig02_03_error.txt").write_text(str(err), encoding="utf-8")

    # Figure 4: cp curve, only if the property backend exposes cp(P,T)/cp_mass(P,T).
    cp_path = _plot_cp_if_available(fluid, out, Tc)
    if cp_path:
        paths["fig04"] = cp_path
    else:
        paths["fig04_status"] = out / "fig04_skipped.txt"
        paths["fig04_status"].write_text(
            "Skipped: backend lacks a usable cp(P,T) or cp_mass(P,T) method, or returned no valid values.\n",
            encoding="utf-8"
        )

    # Figure 5: q_C versus P_U at Tpre3=8 K, flow-independent.
    rows_fig5 = _sweep_specific_qc(fluid, Tc=Tc, Th=Th, tpre12=tpre12,
                                   Tpre3_values=[8.0], Pu_values_MPa=pu_grid, eta=eta_ideal)
    csv5 = out / "fig05_qc_vs_pu_Tpre3_8K.csv"; _write_csv(csv5, rows_fig5); paths["fig05_csv"] = csv5
    _plot_qc_vs_pu(rows_fig5, out / "fig05_qc_vs_pu_Tpre3_8K.png",
                   title="Figure 5 — specific refrigeration versus upstream pressure")
    paths["fig05"] = out / "fig05_qc_vs_pu_Tpre3_8K.png"

    # Figure 6: last-stage P-h cycle changes with Tpre3 at PU=0.36 MPa.
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for tpre3 in (5.0, 8.0, 12.0, 16.0):
        result, err = _safe_call(lambda tt=tpre3: _solve_point(
            fluid, Tc=Tc, Pu_MPa=PAPER.P_U_opt_MPa, Th=Th, tpre12=tpre12,
            Tpre3=tt, eta=eta_ideal, d=None
        ))
        if result is None:
            continue
        ids = [7, 8, 9, 10, 11]
        hs = [result.states[i].h / 1000 for i in ids]
        ps = [result.states[i].P / 1e6 for i in ids]
        ax.plot(hs, ps, "o-", markersize=3, label=f"Tpre3={tpre3:g} K")
    ax.set_yscale("log")
    ax.set(xlabel="Specific enthalpy [kJ/kg]", ylabel="Pressure [MPa]",
           title="Figure 6 — last-stage cycle for different precooling temperatures")
    ax.grid(True, which="both", alpha=0.25)
    if ax.get_legend_handles_labels()[0]: ax.legend()
    fig.savefig(out / "fig06_last_stage_cycles.png", dpi=180); plt.close(fig)
    paths["fig06"] = out / "fig06_last_stage_cycles.png"

    # Figure 7: q_C vs Tpre3 at several upstream pressures.
    pu_fig7 = [0.3, 0.5, 0.7, 0.9]
    rows_fig7 = _sweep_specific_qc(fluid, Tc=Tc, Th=Th, tpre12=tpre12,
                                   Tpre3_values=tpre3_grid, Pu_values_MPa=pu_fig7, eta=eta_ideal)
    csv7 = out / "fig07_qc_vs_tpre3.csv"; _write_csv(csv7, rows_fig7); paths["fig07_csv"] = csv7
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for pu in pu_fig7:
        selected = [r for r in rows_fig7 if math.isclose(float(r["P_U_MPa"]), pu) and np.isfinite(float(r["q_C_kJ_kg"]))]
        selected.sort(key=lambda r: float(r["T_pre3_K"]))
        if selected:
            ax.plot([r["T_pre3_K"] for r in selected], [r["q_C_kJ_kg"] for r in selected], label=f"PU={pu:g} MPa")
    ax.set(xlabel=r"$T_{pre3}$ [K]", ylabel=r"$q_C$ [kJ/kg]", title="Figure 7 — specific refrigeration versus precooling temperature")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]: ax.legend()
    fig.savefig(out / "fig07_qc_vs_tpre3.png", dpi=180); plt.close(fig)
    paths["fig07"] = out / "fig07_qc_vs_tpre3.png"

    # Independently optimize q_C on the sweep grid, then refit functions and compare to Eq 35/36.
    opt_rows = []
    for tpre3 in tpre3_grid:
        selected = [r for r in rows_fig7 if math.isclose(float(r["T_pre3_K"]), float(tpre3)) and np.isfinite(float(r["q_C_kJ_kg"]))]
        # Figure 7 only includes 4 pressures; use the dense PU sweep for actual optimum search.
        sweep, _ = _safe_call(lambda tt=float(tpre3): _sweep_specific_qc(
            fluid, Tc=Tc, Th=Th, tpre12=tpre12, Tpre3_values=[tt], Pu_values_MPa=pu_grid, eta=eta_ideal
        ))
        if not sweep:
            continue
        valid = [r for r in sweep
                 if np.isfinite(float(r.get("q_C_kJ_kg", math.nan)))
                 and float(r.get("q_C_kJ_kg", math.nan)) > 0
                 and not r.get("error")]
        if not valid:
            continue
        best = max(valid, key=lambda r: float(r["q_C_kJ_kg"]))
        opt_rows.append({"T_pre3_K": float(tpre3), "P_U_opt_sim_MPa": float(best["P_U_MPa"]),
                         "q_C_opt_sim_kJ_kg": float(best["q_C_kJ_kg"]),
                         "P_U_opt_paperfit_MPa": float(paper_pu_opt_mpa(float(tpre3))),
                         "q_C_opt_paperfit_kJ_kg": float(paper_qc_opt_kj_kg(paper_pu_opt_mpa(float(tpre3)), float(tpre3)))})
    csv8opt = out / "fig08_independent_optimum_comparison.csv"; _write_csv(csv8opt, opt_rows); paths["fig08_independent_csv"] = csv8opt
    if opt_rows:
        pu_metrics = error_metrics([r["P_U_opt_paperfit_MPa"] for r in opt_rows], [r["P_U_opt_sim_MPa"] for r in opt_rows])
        qc_metrics = error_metrics([r["q_C_opt_paperfit_kJ_kg"] for r in opt_rows], [r["q_C_opt_sim_kJ_kg"] for r in opt_rows])
        metrics_rows = [{"comparison": "P_U_opt_sim_vs_Eq36", **pu_metrics}, {"comparison": "q_C_opt_sim_vs_Eq35", **qc_metrics}]
        metric_path = out / "fig08_optimum_fit_error_metrics.csv"; _write_csv(metric_path, metrics_rows); paths["fig08_metrics"] = metric_path

        # Refit both published relationships from independently calculated
        # optimum points, then save coefficients and their fit/closeness metrics.
        coefficient_rows: List[Dict[str, Any]] = []
        t_fit = [r["T_pre3_K"] for r in opt_rows]
        pu_fit = [r["P_U_opt_sim_MPa"] for r in opt_rows]
        qc_fit = [r["q_C_opt_sim_kJ_kg"] for r in opt_rows]
        if len(opt_rows) >= 5:
            coef_pu, internal_pu_metrics = fit_polynomial_and_report(t_fit, pu_fit, degree=4)
            coefficient_rows.append({
                "fit": "independent_PUopt_fourth_order",
                **{f"a{i}": float(c) for i, c in enumerate(coef_pu)},
                **{f"internal_{k}": v for k, v in internal_pu_metrics.items()},
                **{f"paper_fit_vs_sim_{k}": v for k, v in pu_metrics.items()},
            })
        # qC,opt = b0 + b1*PU,opt + b2*Tpre3, matching Eq. (35)'s form.
        if len(opt_rows) >= 3:
            design = np.column_stack([np.ones(len(opt_rows)), pu_fit, t_fit])
            beta, *_ = np.linalg.lstsq(design, np.asarray(qc_fit, dtype=float), rcond=None)
            qc_linear_prediction = design @ beta
            qc_linear_metrics = error_metrics(qc_fit, qc_linear_prediction)
            coefficient_rows.append({
                "fit": "independent_qCopt_linear",
                "intercept_b0": float(beta[0]), "P_U_opt_coefficient_b1": float(beta[1]),
                "T_pre3_coefficient_b2": float(beta[2]),
                **{f"internal_{k}": v for k, v in qc_linear_metrics.items()},
                **{f"paper_fit_vs_sim_{k}": v for k, v in qc_metrics.items()},
            })
        coefficient_rows.append({
            "fit": "published_Eq36_coefficients",
            **{f"a{i}": float(c) for i, c in enumerate(PAPER_PU_OPT_POLY_ASC)},
        })
        coefficient_rows.append({
            "fit": "published_Eq35_coefficients",
            "intercept_b0": 11.9, "P_U_opt_coefficient_b1": -11.5,
            "T_pre3_coefficient_b2": 0.027,
        })
        coeff_path = out / "fig08_refitted_coefficients.csv"
        _write_csv(coeff_path, coefficient_rows)
        paths["fig08_coefficients"] = coeff_path

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        axes[0].plot([r["T_pre3_K"] for r in opt_rows], [r["P_U_opt_paperfit_MPa"] for r in opt_rows], label="Paper Eq. 36")
        axes[0].plot([r["T_pre3_K"] for r in opt_rows], [r["P_U_opt_sim_MPa"] for r in opt_rows], "o", markersize=3, label="Core sweep optimum")
        axes[0].set(xlabel="Tpre3 [K]", ylabel="PU,opt [MPa]", title="Optimum pressure")
        axes[1].plot([r["T_pre3_K"] for r in opt_rows], [r["q_C_opt_paperfit_kJ_kg"] for r in opt_rows], label="Paper Eq. 35+36")
        axes[1].plot([r["T_pre3_K"] for r in opt_rows], [r["q_C_opt_sim_kJ_kg"] for r in opt_rows], "o", markersize=3, label="Core sweep optimum")
        axes[1].set(xlabel="Tpre3 [K]", ylabel="qC,opt [kJ/kg]", title="Optimum specific refrigeration")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]: ax.legend()
        fig.suptitle("Figure 8 — published fits versus independent core sweep")
        fig.savefig(out / "fig08_independent_optimum_comparison.png", dpi=180); plt.close(fig)
        paths["fig08_independent_png"] = out / "fig08_independent_optimum_comparison.png"

    # Figure 9: HEM restriction path at the baseline point.
    pback, err = _safe_call(lambda: fluid.Psat(Tc))
    if err is None and pback is not None:
        p9 = _restriction_path_plot(fluid, out, Pu_MPa=PAPER.P_U_opt_MPa,
                                   T0=base.states[8].T if base is not None else Tc,
                                   Pback=float(pback))
        if p9: paths["fig09"] = p9

    # Figure 10: requires an ideal-gas comparison (gamma and specific gas constant).
    path10 = _plot_deviation_if_possible(fluid, out, np.linspace(0.05, 2.0, 55), [1.0, 2.0, 2.6, 3.0])
    if path10:
        paths["fig10"] = path10
    else:
        p = out / "fig10_skipped.txt"
        p.write_text("Skipped: backend must expose gamma(P,T) and R_specific (J/kg/K) for an ideal-gas mass-flux comparison.\n", encoding="utf-8")
        paths["fig10_status"] = p

    # Figure 11: paper mass-flux fits plus independent HEM sweep, where supported.
    if figure11_compare_hem:
        rows11 = []
        for t0 in sorted(PAPER_MASS_FLUX_FITS):
            try:
                pb = float(fluid.Psat(t0))
            except Exception:
                continue
            for pu in pu_grid:
                paper_g = float(paper_mass_flux_at_temperature(t0, float(pu)))
                hem_result, err = _safe_call(lambda p=float(pu), t=float(t0), b=pb: critical_mass_flux_HEM(p * 1e6, t, b, fluid))
                rows11.append({"T0_K": t0, "P_U_MPa": float(pu), "G_paper_fit": paper_g,
                               "G_HEM": hem_result["G"] if hem_result else math.nan,
                               "HEM_phase": hem_result["phase"] if hem_result else "",
                               "error": err or ""})
        csv11 = out / "fig11_mass_flux_paper_vs_HEM.csv"; _write_csv(csv11, rows11); paths["fig11_comparison_csv"] = csv11

        # Quantify HEM-versus-paper correlation differences and independently
        # refit each HEM curve to the same alpha*P_U**beta + gamma structure.
        mass_fit_rows: List[Dict[str, Any]] = []
        for t0 in sorted(PAPER_MASS_FLUX_FITS):
            subset = [r for r in rows11 if math.isclose(float(r["T0_K"]), float(t0))
                      and np.isfinite(float(r.get("G_HEM", math.nan)))]
            if not subset:
                mass_fit_rows.append({"T0_K": t0, "status": "No valid HEM points"})
                continue
            pu_vals = [float(r["P_U_MPa"]) for r in subset]
            g_hem = [float(r["G_HEM"]) for r in subset]
            g_paper = [float(r["G_paper_fit"]) for r in subset]
            comparison = error_metrics(g_paper, g_hem)
            row: Dict[str, Any] = {"T0_K": t0, "n": len(subset),
                                   **{f"HEM_vs_paper_{k}": v for k, v in comparison.items()}}
            try:
                fitted_coefficients, fit_metrics = fit_mass_flux_power_law(pu_vals, g_hem)
                row.update({f"HEM_refit_{k}": v for k, v in fitted_coefficients.items()})
                row.update({f"HEM_refit_internal_{k}": v for k, v in fit_metrics.items()})
            except Exception as exc:
                row["HEM_refit_error"] = f"{type(exc).__name__}: {exc}"
            paper_coefficients = PAPER_MASS_FLUX_FITS[float(t0)]
            row.update({f"paper_{k}": v for k, v in paper_coefficients.items()})
            mass_fit_rows.append(row)
        mass_fit_path = out / "fig11_mass_flux_fit_coefficients_and_metrics.csv"
        _write_csv(mass_fit_path, mass_fit_rows)
        paths["fig11_fit_metrics"] = mass_fit_path

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, t0 in zip(axes.ravel(), sorted(PAPER_MASS_FLUX_FITS)):
            sub = [r for r in rows11 if r["T0_K"] == t0]
            if not sub: continue
            sub.sort(key=lambda r: r["P_U_MPa"])
            ax.plot([r["P_U_MPa"] for r in sub], [r["G_paper_fit"] for r in sub], label="Paper fit")
            valid = [r for r in sub if np.isfinite(float(r["G_HEM"]))]
            if valid: ax.plot([r["P_U_MPa"] for r in valid], [r["G_HEM"] for r in valid], "--", label="HEM")
            ax.set(title=f"T0={t0:g} K", xlabel="PU [MPa]", ylabel="G [kg/(m² s)]")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("Figure 11 — paper mass-flux fit versus HEM calculations")
        fig.savefig(out / "fig11_mass_flux_paper_vs_HEM.png", dpi=180); plt.close(fig)
        paths["fig11_comparison_png"] = out / "fig11_mass_flux_paper_vs_HEM.png"

    # Figures 12 and 13: compare gross cooling using HEM with paper's published G-fit.
    rows12 = _sweep_capacity(fluid, Tc=Tc, Th=Th, tpre12=tpre12,
                             Tpre3_values=[8.0], Pu_values_MPa=pu_grid, eta=eta_ideal,
                             d_m=d_m, flow_mode="hem")
    rows12paper = _sweep_capacity(fluid, Tc=Tc, Th=Th, tpre12=tpre12,
                                  Tpre3_values=[8.0], Pu_values_MPa=pu_grid, eta=eta_ideal,
                                  d_m=d_m, flow_mode="paper_fit")
    csv12 = out / "fig12_capacity_Tpre3_8K.csv"; _write_csv(csv12, rows12 + rows12paper); paths["fig12_csv"] = csv12
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for rows, label, key, style in [(rows12, "HEM", "Q_C_mW", "-"), (rows12paper, "Paper G fit", "Q_C_mW", "--")]:
        x, y = _finite_rows(rows, "P_U_MPa", key)
        order = np.argsort(x)
        if x.size: ax.plot(x[order], y[order], style, label=label, linewidth=1.8)
    ax.axhline(PAPER.Q_C_ideal_mW, color="black", linewidth=1, alpha=0.6, label="Paper ideal reference 18.7 mW")
    ax.set(xlabel="PU [MPa]", ylabel="QC [mW]", title="Figure 12 — gross cooling capacity, Tpre3=8 K")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(out / "fig12_capacity_Tpre3_8K.png", dpi=180); plt.close(fig)
    paths["fig12"] = out / "fig12_capacity_Tpre3_8K.png"

    temps13 = [5.0, 8.0, 12.0]
    rows13 = []
    for mode in ("hem", "paper_fit"):
        rows13.extend(_sweep_capacity(fluid, Tc=Tc, Th=Th, tpre12=tpre12,
                                      Tpre3_values=temps13, Pu_values_MPa=pu_grid,
                                      eta=eta_ideal, d_m=d_m, flow_mode=mode))
    csv13 = out / "fig13_capacity_various_Tpre3.csv"; _write_csv(csv13, rows13); paths["fig13_csv"] = csv13
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    for ax, mode, label in [(axes[0], "hem", "HEM mass flux"), (axes[1], "paper_fit", "Paper mass-flux fit")]:
        for temp in temps13:
            selected = [r for r in rows13 if r["flow_mode"] == mode and math.isclose(float(r["T_pre3_K"]), temp)
                        and np.isfinite(float(r["Q_C_mW"]))]
            selected.sort(key=lambda r: float(r["P_U_MPa"]))
            if selected: ax.plot([r["P_U_MPa"] for r in selected], [r["Q_C_mW"] for r in selected], label=f"Tpre3={temp:g} K")
        ax.set(xlabel="PU [MPa]", ylabel="QC [mW]", title=label)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]: ax.legend()
    fig.suptitle("Figure 13 — gross cooling capacity for several precooling temperatures")
    fig.savefig(out / "fig13_capacity_various_Tpre3.png", dpi=180); plt.close(fig)
    paths["fig13"] = out / "fig13_capacity_various_Tpre3.png"

    # Figures 14-16: eta4 sweep at TC=1 K, Tpre3=8 K, PU=0.36 MPa.
    eta_rows: List[Dict[str, Any]] = []
    paper_eta_rows: List[Dict[str, Any]] = []
    for eta4 in eta_grid:
        etas = (1.0, 1.0, 1.0, float(eta4))
        res, err = _safe_call(lambda ee=etas: _solve_point(
            fluid, Tc=Tc, Pu_MPa=PAPER.P_U_opt_MPa, Th=Th, tpre12=tpre12,
            Tpre3=PAPER.T_pre3_opt_K, eta=ee, d=None
        ))
        if err or res is None:
            eta_rows.append({"eta4": float(eta4), "error": err})
            continue
        hem_res, hem_err = _safe_call(lambda ee=etas: _solve_point(
            fluid, Tc=Tc, Pu_MPa=PAPER.P_U_opt_MPa, Th=Th, tpre12=tpre12,
            Tpre3=PAPER.T_pre3_opt_K, eta=ee, d=d_m
        ))
        try:
            pflow = _paper_flow_power(res, PAPER.P_U_opt_MPa, d_m)
            pflow_error = ""
        except Exception as exc:
            pflow = {"T0_fit_K": res.states[8].T, "G_paper_fit": math.nan,
                     "mdot_paper_fit_kg_s": math.nan, "Q_C_paper_fit_W": math.nan}
            pflow_error = f"{type(exc).__name__}: {exc}"
        eta_rows.append({
            "eta4": float(eta4), "T8_K": res.states[8].T, "T11_K": res.states[11].T,
            "q_C_kJ_kg": res.q_C / 1000.0,
            "G_HEM": hem_res.throat["G"] if hem_res is not None and hem_res.throat else math.nan,
            "mdot_HEM_mg_s": hem_res.mdot * 1e6 if hem_res is not None and hem_res.mdot is not None else math.nan,
            "Q_C_HEM_mW": hem_res.Q_C * 1000 if hem_res is not None and hem_res.Q_C is not None else math.nan,
            "Q_C_paper_fit_mW": pflow["Q_C_paper_fit_W"] * 1000,
            "mdot_paper_fit_mg_s": pflow["mdot_paper_fit_kg_s"] * 1e6,
            "paper_fit_error": pflow_error,
            "HEM_error": hem_err or "",
            "flags": res.flags,
        })
    csv_eta = out / "fig14_16_eta4_sweep.csv"; _write_csv(csv_eta, eta_rows); paths["fig14_16_csv"] = csv_eta
    valid_eta = [r for r in eta_rows if r.get("T8_K") is not None and np.isfinite(float(r.get("T8_K", math.nan)))]

    # Figure 14: state 8 and 11 terminal temperatures.
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    if valid_eta:
        ax.plot([r["eta4"] for r in valid_eta], [r["T8_K"] for r in valid_eta], "o-", label="T8, high-side outlet")
        ax.plot([r["eta4"] for r in valid_eta], [r["T11_K"] for r in valid_eta], "s-", label="T11, return-side outlet")
    ax.set(xlabel=r"$\eta_4$", ylabel="Temperature [K]", title="Figure 14 — final CHEX terminal temperatures")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]: ax.legend()
    fig.savefig(out / "fig14_eta4_terminal_temperatures.png", dpi=180); plt.close(fig)
    paths["fig14"] = out / "fig14_eta4_terminal_temperatures.png"

    # Figure 15: q_C and mass flow versus eta4; dual y-axis.
    fig, ax1 = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax2 = ax1.twinx()
    if valid_eta:
        ax1.plot([r["eta4"] for r in valid_eta], [r["q_C_kJ_kg"] for r in valid_eta], "o-", label="qC", color="tab:blue")
        ax2.plot([r["eta4"] for r in valid_eta], [r["mdot_paper_fit_mg_s"] for r in valid_eta], "s--", label="mdot (paper G fit)", color="tab:orange")
        ax2.plot([r["eta4"] for r in valid_eta], [r["mdot_HEM_mg_s"] for r in valid_eta], "^:", label="mdot (HEM)", color="tab:green")
    ax1.set(xlabel=r"$\eta_4$", ylabel=r"$q_C$ [kJ/kg]", title="Figure 15 — refrigeration and mass flow versus CHEX efficiency")
    ax2.set_ylabel("Mass flow [mg/s]")
    ax1.grid(True, alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    if lines: ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    fig.savefig(out / "fig15_eta4_qc_mdot.png", dpi=180); plt.close(fig)
    paths["fig15"] = out / "fig15_eta4_qc_mdot.png"

    # Figure 16: compare absolute/relative reductions for both mass-flow methods.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    eta_ref_rows = [r for r in valid_eta if math.isclose(float(r["eta4"]), 1.0, abs_tol=1e-8)]
    for qkey, label, style in [("Q_C_paper_fit_mW", "Paper G fit", "o-"), ("Q_C_HEM_mW", "HEM", "s--")]:
        usable = [r for r in valid_eta if np.isfinite(float(r.get(qkey, math.nan)))]
        if not usable: continue
        q_at_1 = next((float(r[qkey]) for r in usable if math.isclose(float(r["eta4"]), 1.0, abs_tol=1e-8)), None)
        if q_at_1 is None: continue
        reductions_abs = [q_at_1 - float(r[qkey]) for r in usable]
        reductions_rel = [(q_at_1 - float(r[qkey])) / q_at_1 * 100.0 if q_at_1 else math.nan for r in usable]
        axes[0].plot([r["eta4"] for r in usable], reductions_abs, style, label=label)
        axes[1].plot([r["eta4"] for r in usable], reductions_rel, style, label=label)
    axes[0].set(xlabel=r"$\eta_4$", ylabel="Absolute reduction in QC [mW]", title="Absolute reduction")
    axes[1].set(xlabel=r"$\eta_4$", ylabel="Reduction in QC [%]", title="Relative reduction")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]: ax.legend()
    fig.suptitle("Figure 16 — effect of final-CHEX effectiveness")
    fig.savefig(out / "fig16_eta4_QC_reduction.png", dpi=180); plt.close(fig)
    paths["fig16"] = out / "fig16_eta4_QC_reduction.png"

    # Compare main reference case with published numeric anchors; never force agreement.
    ref_point_rows: List[Dict[str, Any]] = []
    ref_cases = [
        (1.0, (1.0, 1.0, 1.0, 1.0), PAPER.Q_C_ideal_mW, "ideal eta4=1 reference"),
        (PAPER.eta4_real, (1.0, 1.0, 1.0, PAPER.eta4_real), PAPER.Q_C_eta4_097_mW, "eta4=0.97 published real-case reference"),
    ]
    for eta4, etas, qref_mw, label in ref_cases:
        res, err = _safe_call(lambda ee=etas: _solve_point(
            fluid, Tc=Tc, Pu_MPa=PAPER.P_U_opt_MPa, Th=Th, tpre12=tpre12,
            Tpre3=PAPER.T_pre3_opt_K, eta=ee, d=None
        ))
        if err or res is None:
            ref_point_rows.append({"case": label, "error": err})
            continue
        paper_flow, p_err = _safe_call(lambda rr=res: _paper_flow_power(rr, PAPER.P_U_opt_MPa, d_m))
        hem_res, hem_err = _safe_call(lambda ee=etas: _solve_point(
            fluid, Tc=Tc, Pu_MPa=PAPER.P_U_opt_MPa, Th=Th, tpre12=tpre12,
            Tpre3=PAPER.T_pre3_opt_K, eta=ee, d=d_m
        ))
        ref_point_rows.append({
            "case": label, "eta4": eta4, "T8_K": res.states[8].T,
            "q_C_kJ_kg": res.q_C / 1000.0,
            "Q_C_paper_fit_mW": paper_flow["Q_C_paper_fit_W"] * 1000 if paper_flow else math.nan,
            "Q_C_HEM_mW": hem_res.Q_C * 1000 if hem_res and hem_res.Q_C is not None else math.nan,
            "Q_C_paper_reference_mW": qref_mw,
            "paper_fit_abs_error_mW": abs(paper_flow["Q_C_paper_fit_W"] * 1000 - qref_mw) if paper_flow else math.nan,
            "paper_fit_error": p_err or "", "HEM_error": hem_err or "",
        })
    csv_ref = out / "reference_case_comparison.csv"; _write_csv(csv_ref, ref_point_rows); paths["reference_case_csv"] = csv_ref

    # Metadata for what ran and what was skipped.
    report = {
        "mode": "full_reproduction_attempt",
        "fluid_backend_class": f"{type(fluid).__module__}.{type(fluid).__name__}",
        "fluid_backend_verified_for_helium3": False,
        "T_C_K": Tc,
        "T_H_K": Th,
        "T_pre1_K": tpre12[0],
        "T_pre2_K": tpre12[1],
        "orifice_diameter_m": d_m,
        "discharge_coefficient": 1.0,
        "restriction_method": "HEM for HEM curves; published Eq. (49) fit for comparison curves",
        "figures_targeted": list(PAPER_FIGURE_TARGETS.keys()),
        "figures_created": sorted(k for k in paths if k.startswith("fig")),
        "notes": [
            "Figure 4 is skipped unless the backend exposes a usable cp(P,T) method.",
            "Figure 10 is skipped unless backend provides gamma(P,T) and R_specific.",
            "Eq. (49) interpolation between tabulated temperatures is an added approximation for comparing the paper fit at the actual state-8 temperature.",
            "HEM results are not assumed equivalent to the paper's sonic-condition critical-flow model.",
            "Paper reference anchors are comparison targets, not forced values.",
        ],
        "files": {k: str(v) for k, v in paths.items()},
    }
    report_path = out / "full_reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["full_report"] = report_path
    return paths


# ---------------------------------------------------------------------------
# Command-line entry point


def _load_factory(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("Use --fluid-factory package.module:ClassName")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, class_name)
    return factory()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="paper_reproduction", help="Output directory for plots and CSV files")
    parser.add_argument("--reference-only", action="store_true", help="Generate plots available from published fit coefficients only")
    parser.add_argument("--fluid-factory", help="Optional backend factory as module:ClassName for full reproduction")
    parser.add_argument("--tpre1", type=float, help="First PHEX outlet temperature [K], required for full run")
    parser.add_argument("--tpre2", type=float, help="Second PHEX outlet temperature [K], required for full run")
    parser.add_argument("--th", type=float, default=300.0, help="Warm-end inlet temperature [K] (default: 300 K)")
    parser.add_argument("--tc", type=float, default=1.0, help="Target cooling temperature [K] (default: 1 K)")
    parser.add_argument("--diameter-um", type=float, default=20.0, help="Orifice diameter [um] (default: 20 um)")
    args = parser.parse_args(argv)

    if args.reference_only or not args.fluid_factory:
        paths = make_reference_only_outputs(args.outdir)
        print("Reference-only outputs created (no fluid properties used):")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        if not args.reference_only:
            print("\nFull cycle sweeps were not run. Supply --fluid-factory and --tpre1/--tpre2 when a valid backend is available.")
        return 0

    if args.tpre1 is None or args.tpre2 is None:
        parser.error("--tpre1 and --tpre2 are required with --fluid-factory")
    fluid = _load_factory(args.fluid_factory)
    paths = run_full_reproduction(
        fluid,
        output_dir=args.outdir,
        tpre12=(args.tpre1, args.tpre2),
        Th=args.th,
        Tc=args.tc,
        d_m=args.diameter_um * 1e-6,
    )
    print("Full reproduction attempt completed. Review the report and skipped/error files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
