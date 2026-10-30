"""
Tests the CSV interpreter using CoolProp as the 'TA': fills the same two-sheet layout for He-4
(not He-3 physics), loads it with Helium3Fluid, and compares against CoolProp.
Run (from the project root):  python -m tests.test_interpreter_with_coolprop
"""
import math, os, random, sys, tempfile
from math import ceil, log
import numpy as np, pandas as pd
from CoolProp.CoolProp import PropsSI
from jt.properties import Helium3Fluid, SP_COLS, SAT_COLS

FL = "Helium"
def geo(lo, hi, step, inc=True):
    k = ceil(log(hi/lo)/log(1+step)); v = [lo*(hi/lo)**(i/k) for i in range(k)]
    return v + ([hi] if inc else [])
T = sorted(set(round(x, 5) for x in geo(2.2, 4.0, 0.02, False) + geo(4.0, 300.0, 0.05)))
P = sorted(set(round(x, 4) for x in geo(1e3, 3e6, 0.05)))
Tsat = [round(2.2 + 0.01*i, 3) for i in range(int(round((5.10-2.2)/0.01))+1)] + \
       [round(5.10 + 0.002*i, 3) for i in range(1, 47)]          # to 5.192

def make(outdir):
    rows = []
    for t in T:
        for p in P:
            try:
                rows.append((t, p, PropsSI("H","P",p,"T",t,FL), PropsSI("S","P",p,"T",t,FL), PropsSI("D","P",p,"T",t,FL)))
            except Exception:
                rows.append((t, p, np.nan, np.nan, np.nan))
    pd.DataFrame(rows, columns=SP_COLS).to_csv(os.path.join(outdir, "SinglePhase.csv"), index=False, encoding="utf-8-sig")
    srows = []
    for t in Tsat:
        g = lambda k, q: PropsSI(k, "T", t, "Q", q, FL)
        srows.append((t, g("P",1), g("H",0), g("H",1), g("S",0), g("S",1), g("D",0), g("D",1)))
    pd.DataFrame(srows, columns=SAT_COLS).to_csv(os.path.join(outdir, "Saturation.csv"), index=False, encoding="utf-8-sig")

def rel(a, b): return abs(a-b)/max(abs(b), 1e-30)

def main():
    d = tempfile.mkdtemp(); make(d)
    f = Helium3Fluid(os.path.join(d, "SinglePhase.csv"), os.path.join(d, "Saturation.csv"))
    print("grid", len(f.T), "x", len(f.P))
    random.seed(1)
    # off-grid random single-phase states
    errs = {"h": [], "s": [], "rho": [], "Tinv": [], "hPs": [], "rhoPs": []}
    worst = {}
    n = 0
    while n < 1500:
        t = math.exp(random.uniform(log(2.3), log(290)))
        p = math.exp(random.uniform(log(1.2e3), log(2.9e6)))
        try:
            ref_h = PropsSI("H","P",p,"T",t,FL); ref_s = PropsSI("S","P",p,"T",t,FL); ref_r = PropsSI("D","P",p,"T",t,FL)
        except Exception:
            continue
        n += 1
        try:
            h = f.h(p, t); s = f.s(p, t); r = f.rho(p, t)
        except Exception as e:
            errs.setdefault("fail", []).append((p, t, str(e))); continue
        # errors measured as absolute enthalpy error relative to the span of h on that isobar is not
        # meaningful, so use rel. error on h, s, rho but with absolute T for inverse
        errs["h"].append((abs(h-ref_h), p, t)); errs["s"].append((abs(s-ref_s), p, t)); errs["rho"].append((rel(r, ref_r), p, t))
        try:
            errs["Tinv"].append((abs(f.T_from_Ph(p, ref_h) - t), p, t))
            errs["hPs"].append((abs(f.h_from_Ps(p, ref_s) - ref_h), p, t))
            errs["rhoPs"].append((rel(f.rho_from_Ps(p, ref_s), ref_r), p, t))
        except Exception as e:
            errs.setdefault("fail_inv", []).append((p, t, str(e)))
    for k in ["h", "s", "rho", "Tinv", "hPs", "rhoPs"]:
        a = np.array([e[0] for e in errs[k]])
        print(f"{k:6s} median={np.median(a):.2e}  95%={np.percentile(a,95):.2e}  max={a.max():.2e}")
    for k in ("h", "s", "rho", "Tinv"):
        w = sorted(errs[k], reverse=True)[:3]
        print("  worst", k, [(f"{e:.2e}", f"P={p:.4g}", f"T={t:.4g}") for e, p, t in w])
    for k in ("fail", "fail_inv"):
        if k in errs: print(k, len(errs[k]), errs[k][:2])
    # two-phase (HEM) vs CoolProp
    tp = []
    for p in [8e3, 2e4, 5e4, 1e5, 1.5e5]:
        sat = f._sat_state(p)
        for x in [0.1, 0.5, 0.9]:
            s = sat["sl"] + x*(sat["sv"]-sat["sl"])
            ss = PropsSI("S","P",p,"Q",x,FL)  # reference at same pressure and quality
            tp.append((rel(f.h_from_Ps(p, ss), PropsSI("H","P",p,"Q",x,FL)),
                       rel(f.rho_from_Ps(p, ss), PropsSI("D","P",p,"Q",x,FL))))
    tp = np.array(tp); print("two-phase  h rel err max=%.2e  rho rel err max=%.2e" % (tp[:, 0].max(), tp[:, 1].max()))
    # saturation lookups
    print("Psat(3.0)=%.6g vs %.6g ; hv(3.0)=%.6g vs %.6g" % (f.Psat(3.0), PropsSI("P","T",3.0,"Q",1,FL), f.h_sat_vapor(3.0), PropsSI("H","T",3.0,"Q",1,FL)))
    # full cycle with both backends
    from jt import cycle_core as jt
    cp = jt.CoolPropFluid(FL)
    for Tc, Pu, eta4 in [(2.5, 0.36e6, 1.0), (2.5, 0.36e6, 0.97), (3.0, 0.5e6, 1.0)]:
        out = []
        for fl in (cp, f):
            r = jt.solve_cycle(fl, Tc=Tc, Pu=Pu, Th=300., Tpre=[80., 25., 8.], eta=[1, 1, 1, eta4], d=20e-6)
            out.append(r)
        a, b = out
        print(f"cycle Tc={Tc} eta4={eta4}: q_C {a.q_C:.1f} vs {b.q_C:.1f} J/kg ({rel(b.q_C,a.q_C):.1e}); "
              f"mdot {a.mdot:.4e} vs {b.mdot:.4e} ({rel(b.mdot,a.mdot):.1e}); Q_pre3 rel {rel(b.Q_pre[2],a.Q_pre[2]):.1e}; flags {b.flags}")

if __name__ == "__main__":
    main()
