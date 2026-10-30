"""Extra figures used in the documents: Q_C vs P_U for several T_pre,last (cf. paper Fig. 13)
and the He-3 c_p curves (cf. paper Fig. 4).

    python make_extra_figures.py                     # default grid
    python make_extra_figures.py --n-pu 200 --n-t 400 --outdir results/extra
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jt.cycle_core import make_fluid
from jt import design_optimize as do


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results/extra")
    ap.add_argument("--n-pu", type=int, default=36, help="P_U points per Q_C curve")
    ap.add_argument("--n-t", type=int, default=200, help="temperature points per c_p curve")
    ap.add_argument("--tpre", nargs="+", type=float, default=[4, 5, 8, 10, 12], help="T_pre,last values [K]")
    ap.add_argument("--hem-grid", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    print("loading He-3 tables ...", flush=True)
    f = make_fluid("he3")
    total, done, t0 = len(a.tpre) * a.n_pu, 0, time.time()

    def bar() -> None:
        el = time.time() - t0
        eta = el / done * (total - done) if done else 0.0
        n = int(30 * done / total)
        sys.stdout.write(f"\r  Q_C curves [{'#' * n}{'.' * (30 - n)}] {done}/{total}  "
                         f"elapsed {el:5.0f} s  left ~{eta:5.0f} s ")
        sys.stdout.flush()

    # Q_C vs P_U (P_U from 0.12 MPa: below it the expansion crosses the tables' near-critical gap)
    pu = np.linspace(0.12, 0.95, a.n_pu)
    rows = []
    plt.figure(figsize=(5.4, 3.6))
    for tp in a.tpre:
        Q = []
        for p in pu:
            try:
                r = do.evaluate_last_stage(f, tc_k=1.0, pu_mpa=float(p), tpre_last_k=tp, d_m=20e-6,
                                           flow_model="hem", hem_grid=a.hem_grid)
                Q.append(r["q_c_w"] * 1e3)
            except Exception:
                Q.append(float("nan"))
            rows.append({"T_pre_last_K": tp, "P_U_MPa": float(p), "Q_C_mW": Q[-1]})
            done += 1
            bar()
        plt.plot(pu, Q, label=f"$T_{{pre,last}}$ = {tp:g} K")
    plt.xlabel("$P_U$ (MPa)"); plt.ylabel("$Q_C$ (mW)"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title("Q_C vs P_U, T_C = 1 K, d = 20 um (HEM)", fontsize=9)
    plt.tight_layout(); plt.savefig(out / "QC_vs_PU_Tpre.png", dpi=200); plt.close()
    with open(out / "QC_vs_PU_Tpre.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("\n  c_p curves ...", flush=True)
    # c_p from the tables at P_U = 0.36 MPa and P_D = P_sat(1 K)
    Pu, Pd = 0.36e6, float(f.Psat(1.0))
    T = np.linspace(1.05, 20, a.n_t)
    cp = lambda P, t, dT=0.02: (f.h(P, t + dT) - f.h(P, t - dT)) / (2 * dT) / 1e3
    plt.figure(figsize=(5.4, 3.4))
    plt.plot(T, [cp(Pu, t) for t in T], label="$P_U$ = 0.36 MPa")
    plt.plot(T, [cp(Pd, t) for t in T], label=f"$P_D$ = {Pd / 1e3:.2f} kPa")
    plt.xlabel("T (K)"); plt.ylabel("$c_p$ (kJ/kg/K)"); plt.yscale("log"); plt.grid(alpha=.3, which="both")
    plt.legend(fontsize=8); plt.title("He-3 isobaric heat capacity from the tables", fontsize=9)
    plt.tight_layout(); plt.savefig(out / "cp_He3.png", dpi=200); plt.close()
    print("wrote", out / "QC_vs_PU_Tpre.png", "and", out / "cp_He3.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())    