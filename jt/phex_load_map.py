"""Map PHEX cooling loads at a fixed JT operating point.

This is phase 2 of the design workflow. It reads the named optimization JSON from
jt/design_optimize.py, fixes T_C, d, P_U*, T_pre,last*, eta_last and mdot*,
then sweeps the warmer PHEX temperatures. For each combination it calls the
shared full-cycle solver to calculate every specific PHEX duty and multiplies
those duties by the fixed mass flow from phase 1.

The mapping does NOT re-optimize the JT stage for each temperature combination.
It answers: for the selected optimal JT operating point, what loads must the
individual PHEX stages supply as the warmer PHEX temperatures vary?

Example: four-stage Stirling cooler (three PHEXs), with T_pre3 fixed at the
optimum from phase 1. The two warmer levels are swept over user-defined ranges:
    python -m jt.phex_load_map \
        --fluid he3 \
        --optimum-json results/optimization/Helium_3_2.1K_20um_4_stage.json \
        --warmer-ranges 30:200:16 8.5:80:16

Results are saved beside the input optimization JSON by default and reuse its
configuration stem, e.g. Helium_3_2.1K_20um_4_stage_phex_load_map.csv.

Each warmer range is ``MIN_K:MAX_K:POINTS`` in warm-to-cold order. Only
combinations satisfying T_pre1 > T_pre2 > ... > T_pre,last and T_H > T_pre1
are evaluated. For a three-stage Stirling cooler, there is one warmer PHEX, so
supply one range, e.g. ``--stirling-stages 3 --warmer-ranges 10:200:50``.

The property backend and jt/cycle_core.py must be available on the Python path.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from jt.cycle_core import CoolPropFluid, solve_cycle


def _load_fluid(fluid_name: Optional[str], factory_spec: Optional[str]) -> Tuple[Any, str]:
    if factory_spec:
        if ":" not in factory_spec:
            raise ValueError("Use --fluid-factory package.module:ClassName")
        module_name, class_name = factory_spec.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, class_name)
        return factory(), factory_spec
    if not fluid_name:
        raise ValueError("Specify either --fluid or --fluid-factory")
    normalized = fluid_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized in {"he3", "helium3"}:
        module = importlib.import_module("jt.properties")
        return getattr(module, "Helium3Fluid")(), "jt.properties:Helium3Fluid"
    return CoolPropFluid(fluid_name), f"CoolProp:{fluid_name}"


def _parse_range(token: str) -> Tuple[float, float, int]:
    try:
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError
        low, high, count = float(parts[0]), float(parts[1]), int(parts[2])
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid range {token!r}; use MIN_K:MAX_K:POINTS, e.g. 30:200:16"
        ) from exc
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high and count >= 2):
        raise argparse.ArgumentTypeError(f"Invalid range {token!r}; require 0 < MIN < MAX and POINTS >= 2")
    return low, high, count


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_or_nan(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _generate_ranges(
    n_warmer: int,
    *,
    supplied: Optional[Sequence[Tuple[float, float, int]]],
    tpre_last_k: float,
    th_k: float,
    default_points: int,
) -> List[Tuple[float, float, int]]:
    if supplied is not None:
        if len(supplied) != n_warmer:
            raise ValueError(
                f"This configuration needs {n_warmer} warmer-PHEX ranges, but got {len(supplied)}. "
                "Specify one MIN:MAX:POINTS range for each PHEX temperature above the fixed last PHEX."
            )
        ranges = list(supplied)
    else:
        # Broad default search; callers should normally provide stage-specific ranges
        # informed by the particular Stirling cryocooler.
        low = tpre_last_k + max(0.5, 0.05 * tpre_last_k)
        high = th_k - 1e-6
        if low >= high:
            raise ValueError("Cannot form default warmer-temperature ranges: Tpre_last must be below Th")
        ranges = [(low, high, default_points) for _ in range(n_warmer)]

    for i, (low, high, count) in enumerate(ranges, start=1):
        if low <= tpre_last_k:
            raise ValueError(
                f"Range {i} starts at {low:g} K, which is not above fixed T_pre,last={tpre_last_k:g} K"
            )
        if high >= th_k:
            raise ValueError(f"Range {i} ends at {high:g} K, which must be below ambient inlet Th={th_k:g} K")
        if count < 2 or low >= high:
            raise ValueError(f"Invalid range {i}: require 0 < MIN < MAX and POINTS >= 2")
    return ranges


def map_phex_loads(
    fluid: Any,
    *,
    optimum: Mapping[str, Any],
    stirling_stages: int,
    th_k: float = 300.0,
    eta_above: Optional[Sequence[float]] = None,
    warmer_ranges: Optional[Sequence[Tuple[float, float, int]]] = None,
    default_points: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Map valid warmer-PHEX temperature combinations for a fixed JT optimum."""
    if stirling_stages < 2:
        raise ValueError("stirling_stages must be >= 2")
    n_chex = int(stirling_stages)
    n_phex = n_chex - 1
    n_warmer = n_phex - 1

    opt = optimum.get("optimum", optimum)
    tc_k = float(optimum.get("T_C_K", opt.get("T_C_K", math.nan)))
    if not math.isfinite(tc_k):
        tc_k = float(opt.get("tc_k", math.nan))
    pu_mpa = float(opt.get("P_U_MPa", opt.get("pu_mpa", math.nan)))
    tpre_last_k = float(opt.get("T_pre_last_K", opt.get("tpre_last_k", math.nan)))
    mdot_kg_s = float(opt.get("mdot_kg_s", math.nan))
    eta_last = float(optimum.get("eta_last", opt.get("eta_last", 1.0)))
    q_c_opt_j_kg = float(opt.get("q_C_J_kg", opt.get("q_c_j_kg", math.nan)))
    qc_opt_w = float(opt.get("Q_C_W", opt.get("q_c_w", math.nan)))

    needed = {
        "T_C_K": tc_k,
        "P_U_MPa": pu_mpa,
        "T_pre_last_K": tpre_last_k,
        "mdot_kg_s": mdot_kg_s,
        "eta_last": eta_last,
    }
    missing = [name for name, value in needed.items() if not math.isfinite(value)]
    if missing:
        raise ValueError("Optimum JSON is missing required finite values: " + ", ".join(missing))
    if not (tc_k > 0 and pu_mpa > 0 and tpre_last_k > tc_k and mdot_kg_s > 0):
        raise ValueError("Optimum JSON contains physically invalid T_C/P_U/T_pre,last/mdot values")
    if not math.isfinite(th_k) or th_k <= tpre_last_k:
        raise ValueError("Th must be finite and greater than the fixed last PHEX temperature")

    if eta_above is None:
        eta_above_values = [1.0] * (n_chex - 1)
    else:
        eta_above_values = [float(x) for x in eta_above]
        if len(eta_above_values) != n_chex - 1:
            raise ValueError(
                f"Need {n_chex - 1} upstream CHEX efficiencies for {n_chex} Stirling stages; "
                f"got {len(eta_above_values)}"
            )
        if any(not math.isfinite(x) or x < 0 or x > 1 for x in eta_above_values):
            raise ValueError("All upstream CHEX efficiencies must lie in [0, 1]")
    eta = eta_above_values + [eta_last]
    ranges = _generate_ranges(
        n_warmer,
        supplied=warmer_ranges,
        tpre_last_k=tpre_last_k,
        th_k=th_k,
        default_points=default_points,
    )
    grids = [np.linspace(low, high, count) for low, high, count in ranges]

    rows: List[Dict[str, Any]] = []
    total_combinations = 0
    rejected_ordering = 0
    for warmer_values in itertools.product(*grids) if n_warmer else [tuple()]:
        total_combinations += 1
        tpre_values = [float(x) for x in warmer_values] + [tpre_last_k]
        # PHEX temperatures are supplied warm-to-cold; they should strictly decrease.
        if any(tpre_values[i] <= tpre_values[i + 1] for i in range(len(tpre_values) - 1)):
            rejected_ordering += 1
            continue
        if tpre_values[0] >= th_k:
            rejected_ordering += 1
            continue

        row: Dict[str, Any] = {
            "T_C_K": tc_k,
            "P_U_MPa_fixed": pu_mpa,
            "T_pre_last_K_fixed": tpre_last_k,
            "eta_last_fixed": eta_last,
            "flow_model_from_phase1": optimum.get("flow_model", "unknown"),
            "mdot_fixed_kg_s": mdot_kg_s,
            "mdot_fixed_mg_s": mdot_kg_s * 1e6,
        }
        for i, tpre in enumerate(tpre_values, start=1):
            row[f"T_pre{i}_K"] = tpre
        try:
            result = solve_cycle(
                fluid,
                Tc=tc_k,
                Pu=pu_mpa * 1.0e6,
                Th=th_k,
                Tpre=tpre_values,
                eta=eta,
                d=None,  # Keep mdot fixed from phase 1; do not rerun restriction flow for every map point.
            )
            row["q_C_core_J_kg"] = result.q_C
            row["q_C_core_kJ_kg"] = result.q_C / 1000.0
            row["q_C_phase1_J_kg"] = q_c_opt_j_kg
            row["q_C_difference_vs_phase1_J_kg"] = result.q_C - q_c_opt_j_kg
            row["Q_C_fixed_mW"] = mdot_kg_s * result.q_C * 1000.0
            row["Q_C_phase1_mW"] = qc_opt_w * 1000.0 if math.isfinite(qc_opt_w) else math.nan
            for i, qpre in enumerate(result.q_pre, start=1):
                row[f"q_pre{i}_J_kg"] = qpre
                row[f"q_pre{i}_kJ_kg"] = qpre / 1000.0
                row[f"Q_pre{i}_W"] = mdot_kg_s * qpre
                row[f"Q_pre{i}_mW"] = mdot_kg_s * qpre * 1000.0
            row["energy_residual_J_kg"] = result.energy_residual
            row["flags"] = " | ".join(result.flags)
            row["error"] = ""
        except Exception as exc:
            row["q_C_core_J_kg"] = math.nan
            row["q_C_core_kJ_kg"] = math.nan
            row["Q_C_fixed_mW"] = math.nan
            row["energy_residual_J_kg"] = math.nan
            row["flags"] = ""
            row["error"] = f"{type(exc).__name__}: {exc}"
            for i in range(1, n_phex + 1):
                row[f"q_pre{i}_J_kg"] = math.nan
                row[f"q_pre{i}_kJ_kg"] = math.nan
                row[f"Q_pre{i}_W"] = math.nan
                row[f"Q_pre{i}_mW"] = math.nan
        rows.append(row)

    valid_rows = [r for r in rows if not r.get("error")]
    report = {
        "schema_version": 1,
        "task": "PHEX_load_map_at_fixed_JT_optimum",
        "optimum_json_source": None,
        "T_C_K": tc_k,
        "P_U_MPa_fixed": pu_mpa,
        "T_pre_last_K_fixed": tpre_last_k,
        "diameter_um_from_phase1": optimum.get("diameter_um"),
        "flow_model_from_phase1": optimum.get("flow_model", "unknown"),
        "mdot_kg_s_fixed_from_phase1": mdot_kg_s,
        "mdot_mg_s_fixed_from_phase1": mdot_kg_s * 1e6,
        "Q_C_mW_phase1": qc_opt_w * 1000.0 if math.isfinite(qc_opt_w) else None,
        "stirling_stages": n_chex,
        "number_of_PHEXs": n_phex,
        "CHEX_effectiveness_order": eta,
        "warmer_temperature_ranges_K_points": [
            {"min": lo, "max": hi, "points": n} for lo, hi, n in ranges
        ],
        "combinations_generated": total_combinations,
        "combinations_rejected_for_temperature_order": rejected_ordering,
        "combinations_calculated": len(rows),
        "successful_calculations": len(valid_rows),
        "failed_calculations": len(rows) - len(valid_rows),
        "note": (
            "JT optimum and mass flow are fixed from phase 1. Upstream CHEXs/PHEXs are "
            "recalculated for each temperature combination. Loads are required loads, "
            "not proof of Stirling-stage feasibility; compare with actual capacity curves."
        ),
    }
    return rows, report


def _write_plots(
    outdir: Path, rows: Sequence[Mapping[str, Any]], stirling_stages: int, stem: str
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    good = [r for r in rows if not r.get("error")]
    if not good:
        return []
    n_phex = stirling_stages - 1
    n_warmer = n_phex - 1
    outputs: List[Path] = []
    try:
        if n_warmer == 1:
            x_key = "T_pre1_K"
            sorted_rows = sorted(good, key=lambda r: float(r[x_key]))
            fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
            for i in range(1, n_phex + 1):
                key = f"Q_pre{i}_mW"
                ax.plot([r[x_key] for r in sorted_rows], [r.get(key, math.nan) for r in sorted_rows],
                        marker="o", markersize=3, label=f"PHEX {i}")
            ax.set_xlabel(r"Warmer PHEX temperature $T_{pre,1}$ (K)")
            ax.set_ylabel("Required PHEX load (mW)")
            ax.set_title("PHEX load map at fixed JT optimum")
            ax.grid(True, alpha=0.3)
            ax.legend()
            out = outdir / f"{stem}_phex_load_map.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            outputs.append(out)
        elif n_warmer == 2:
            x_values = sorted({float(r["T_pre2_K"]) for r in good})
            y_values = sorted({float(r["T_pre1_K"]) for r in good})
            x_index = {v: i for i, v in enumerate(x_values)}
            y_index = {v: i for i, v in enumerate(y_values)}
            fig, axes = plt.subplots(1, n_phex, figsize=(5.2 * n_phex, 4.5), squeeze=False,
                                     constrained_layout=True)
            for i in range(1, n_phex + 1):
                matrix = np.full((len(y_values), len(x_values)), np.nan)
                key = f"Q_pre{i}_mW"
                for r in good:
                    yi = y_index[float(r["T_pre1_K"])]
                    xi = x_index[float(r["T_pre2_K"])]
                    matrix[yi, xi] = float(r.get(key, math.nan))
                mesh = axes[0, i - 1].pcolormesh(x_values, y_values, matrix, shading="auto")
                fig.colorbar(mesh, ax=axes[0, i - 1], label="Required load (mW)")
                axes[0, i - 1].set_xlabel(r"$T_{pre,2}$ (K)")
                axes[0, i - 1].set_ylabel(r"$T_{pre,1}$ (K)")
                axes[0, i - 1].set_title(f"PHEX {i}")
            fig.suptitle("PHEX load maps at fixed JT optimum")
            out = outdir / f"{stem}_phex_load_maps_2d.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            outputs.append(out)
    except Exception:
        return outputs
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    fluid_group = parser.add_mutually_exclusive_group(required=True)
    fluid_group.add_argument("--fluid", help="Fluid name for CoolProp, or 'he3' for the He-3 CSV backend")
    fluid_group.add_argument("--fluid-factory", help="Custom no-argument backend as module:ClassName")
    parser.add_argument("--optimum-json", required=True,
                        help="jt_optimum.json produced by jt/design_optimize.py")
    parser.add_argument("--stirling-stages", type=int, default=None,
                        help="Optional override; otherwise read from the optimization JSON")
    parser.add_argument("--th", type=float, default=300.0, help="Warm-end high-side inlet temperature [K]")
    parser.add_argument("--eta-above", type=float, nargs="+", default=None,
                        help="CHEX efficiencies above the last CHEX, warm-to-cold; default all 1")
    parser.add_argument("--warmer-ranges", nargs="+", type=_parse_range, default=None,
                        help="One MIN:MAX:POINTS triple per warmer PHEX temperature")
    parser.add_argument("--default-points", type=int, default=20,
                        help="Points per warmer temperature if --warmer-ranges is omitted")
    parser.add_argument("--outdir", default=None,
                        help="Result directory [default: directory containing the optimization JSON]")
    args = parser.parse_args(argv)

    try:
        source_path = Path(args.optimum_json)
        optimum = json.loads(source_path.read_text(encoding="utf-8"))
        saved_stages = optimum.get("stirling_stages")
        if args.stirling_stages is None:
            if saved_stages is None:
                raise ValueError("Optimization JSON has no stirling_stages field; rerun phase 1 or pass --stirling-stages")
            stirling_stages = int(saved_stages)
        else:
            stirling_stages = int(args.stirling_stages)
            if saved_stages is not None and stirling_stages != int(saved_stages):
                raise ValueError(
                    f"Stage count mismatch: JSON specifies {saved_stages}, but --stirling-stages specifies {stirling_stages}"
                )
        if stirling_stages < 2:
            raise ValueError("stirling_stages must be at least 2")
        fluid, backend_label = _load_fluid(args.fluid, args.fluid_factory)
        expected_backend = optimum.get("fluid_backend")
        if expected_backend and expected_backend != backend_label:
            print(f"WARNING: optimum was generated with {expected_backend!r}, but this run uses {backend_label!r}. Use the same fluid backend for a consistent map.")
        rows, report = map_phex_loads(
            fluid,
            optimum=optimum,
            stirling_stages=stirling_stages,
            th_k=args.th,
            eta_above=args.eta_above,
            warmer_ranges=args.warmer_ranges,
            default_points=args.default_points,
        )
        report["optimum_json_source"] = str(source_path.resolve())
        report["fluid_backend_used_for_map"] = backend_label
        outdir = Path(args.outdir) if args.outdir is not None else source_path.parent
        outdir.mkdir(parents=True, exist_ok=True)
        stem = source_path.stem
        csv_path = outdir / f"{stem}_phex_load_map.csv"
        report_path = outdir / f"{stem}_phex_load_map_report.json"
        _write_csv(csv_path, rows)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        plots = _write_plots(outdir, rows, stirling_stages, stem)
        print("PHEX load mapping complete.")
        print(f"  Stirling stages / CHEXs = {stirling_stages}")
        print(f"  PHEXs                   = {stirling_stages - 1}")
        print(f"  Valid load combinations= {report['successful_calculations']}")
        print(f"  Rejected temperature orderings = {report['combinations_rejected_for_temperature_order']}")
        print(f"  Load map CSV            : {csv_path.resolve()}")
        print(f"  Report JSON             : {report_path.resolve()}")
        for plot in plots:
            print(f"  Plot                    : {plot.resolve()}")
        print("Note: a valid cycle calculation is not, by itself, proof that the Stirling stages can meet the loads.")
    except Exception as exc:
        parser.exit(2, f"ERROR: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
