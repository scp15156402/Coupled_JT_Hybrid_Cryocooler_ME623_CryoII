"""
He-3 property interpreter: a CoolProp-like backend built from two CSV tables.

Files (CSV UTF-8, columns exactly as in the request workbook; extra columns ignored):
  SinglePhase_He_3.csv : T [K], P [Pa], h [J/kg], s [J/(kg K)], rho [kg/m3]
  Saturation_He_3.csv  : T [K], P_sat [Pa], h_liquid, h_vapour, s_liquid, s_vapour,
                         rho_liquid, rho_vapour

Implements the FluidTable interface used by jt_cycle_core.py:
  h, s, T_from_Ph, Psat, h_sat_vapor, x_v_from_Ph, h_from_Ps, rho_from_Ps
plus rho(P,T), cp(P,T) (numerical, optional), gamma, R_specific.

Method
------
* The single-phase grid is split into a LIQUID-side class (P above Psat(T), or
  P >= Pcrit) and a VAPOUR-side class (P below Psat(T)).
* Where saturation data are available, the saturation table supplies an anchor
  at P = Psat(T), so interpolation near the phase boundary does not extrapolate.
* At a fixed pressure, the tabulated properties are interpolated across
  temperature using monotone cubic interpolation.
* Inverse lookups root-find on the same interpolated profile.
* Two-phase states use homogeneous-equilibrium relations:
      h = h_l + x*h_fg
      s = s_l + x*s_fg
      1/rho = (1-x)/rho_l + x/rho_v
* Saturation-table coverage is independent of the physical critical point.
* Saturation queries for the cold end are restricted to T <= 3.2 K.
* Requests requiring saturation information outside its tabulated coverage
  raise OutOfRange rather than extrapolating the saturation curve.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


R_UNIVERSAL = 8.314462618
M_HE3 = 3.0160293e-3  # kg/mol


# ---------------------------------------------------------------------------
# Physical helium-3 constants
# ---------------------------------------------------------------------------
# These constants are independent of the saturation table's coverage.
#
# T_CRIT is based on the 1962 He-3 temperature scale.
# P_CRIT is approximately 873 mmHg, converted to Pa.
T_CRIT = 3.3246       # K
P_CRIT = 116400.0     # Pa, approximately


# ---------------------------------------------------------------------------
# Operating-range restriction for this reproduction
# ---------------------------------------------------------------------------
# The cold-end / evaporator temperature Tc must not exceed 3.2 K.
# The saturation table must cover the requested temperature.
MAX_COLD_END_T = 3.2  # K


SP_COLS = [
    "T [K]",
    "P [Pa]",
    "h [J/kg]",
    "s [J/(kg K)]",
    "rho [kg/m3]",
]

SAT_COLS = [
    "T [K]",
    "P_sat [Pa]",
    "h_liquid [J/kg]",
    "h_vapour [J/kg]",
    "s_liquid [J/(kg K)]",
    "s_vapour [J/(kg K)]",
    "rho_liquid [kg/m3]",
    "rho_vapour [kg/m3]",
]


class OutOfRange(ValueError):
    """Requested state is outside the supported tabulated data."""


class DataError(ValueError):
    """The CSV data are malformed or not usable."""


def _read(path: str, cols):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(
            f"{os.path.basename(path)}: missing columns {missing}"
        )

    df = df[cols].apply(pd.to_numeric, errors="coerce")
    return df


class _Profile:
    """Isobar profile: h, s, ln(rho) as functions of T at fixed pressure."""

    __slots__ = ("T0", "T1", "h", "s", "lr")

    def __init__(self, Tn, h, s, lr):
        self.T0 = float(Tn[0])
        self.T1 = float(Tn[-1])

        self.h = PchipInterpolator(Tn, h)
        self.s = PchipInterpolator(Tn, s)
        self.lr = PchipInterpolator(Tn, lr)

    def check(self, T):
        tol = 1e-9 * max(T, 1.0)

        if T < self.T0 - tol or T > self.T1 + tol:
            raise OutOfRange(
                f"T={T:.6g} K outside available range "
                f"[{self.T0:.6g}, {self.T1:.6g}] K at this pressure"
            )


class Helium3Fluid:
    gamma = 5.0 / 3.0
    R_specific = R_UNIVERSAL / M_HE3
    molar_mass = M_HE3

    def __init__(
        self,
        singlephase_csv: Optional[str] = None,
        saturation_csv: Optional[str] = None,
    ):
        here = os.path.dirname(os.path.abspath(__file__))

        singlephase_csv = (
            singlephase_csv
            or os.path.join(here, "SinglePhase_He_3.csv")
        )
        saturation_csv = (
            saturation_csv
            or os.path.join(here, "Saturation_He_3.csv")
        )

        self._load_saturation(saturation_csv)
        self._load_grid(singlephase_csv)

        self._cls = {
            "L": self._build_class("L"),
            "V": self._build_class("V"),
        }

        self._profile = lru_cache(maxsize=512)(
            self._profile_uncached
        )

    # ------------------------------------------------------------------
    # Loading saturation properties
    # ------------------------------------------------------------------

    def _load_saturation(self, path):
        df = (
            _read(path, SAT_COLS)
            .dropna()
            .sort_values(SAT_COLS[0])
        )

        if len(df) < 10:
            raise DataError(
                "Saturation table has too few complete rows"
            )

        T = df[SAT_COLS[0]].to_numpy(float)
        Ps = df[SAT_COLS[1]].to_numpy(float)

        if np.any(np.diff(T) <= 0) or np.any(np.diff(np.log(Ps)) <= 0):
            raise DataError(
                "Saturation T and P_sat must both be strictly increasing"
            )

        self.sat_T = T

        # Actual saturation-table coverage.
        # These are NOT the physical critical-point properties.
        self.Tmin_sat = float(T[0])
        self.Tsat_max = float(T[-1])

        self.Pmin_sat = float(Ps[0])
        self.Psat_max = float(Ps[-1])

        # Physical critical-point constants are defined independently
        # of the final row in the saturation table.
        self.Tcrit = T_CRIT
        self.Pcrit = P_CRIT

        # Ensure the requested cold-end operating range is covered.
        if self.Tsat_max < MAX_COLD_END_T:
            raise DataError(
                f"Saturation table ends at {self.Tsat_max:.6g} K, "
                f"below the configured maximum cold-end temperature "
                f"{MAX_COLD_END_T:.6g} K"
            )

        lnP = np.log(Ps)

        self._lnPsat = PchipInterpolator(T, lnP)
        self._Tsat = PchipInterpolator(lnP, T)

        g = lambda c, f=lambda v: v: PchipInterpolator(
            T, f(df[c].to_numpy(float))
        )

        self._sat = {
            "hl": g(SAT_COLS[2]),
            "hv": g(SAT_COLS[3]),
            "sl": g(SAT_COLS[4]),
            "sv": g(SAT_COLS[5]),
            "lrl": g(SAT_COLS[6], np.log),
            "lrv": g(SAT_COLS[7], np.log),
        }

    # ------------------------------------------------------------------
    # Loading single-phase property grid
    # ------------------------------------------------------------------

    def _load_grid(self, path):
        df = _read(path, SP_COLS).dropna(
            subset=SP_COLS[:2]
        )

        self.T = np.unique(
            df[SP_COLS[0]].to_numpy(float)
        )
        self.P = np.unique(
            df[SP_COLS[1]].to_numpy(float)
        )

        iT = np.searchsorted(
            self.T, df[SP_COLS[0]].to_numpy(float)
        )
        iP = np.searchsorted(
            self.P, df[SP_COLS[1]].to_numpy(float)
        )

        shape = (len(self.T), len(self.P))

        self._H = np.full(shape, np.nan)
        self._S = np.full(shape, np.nan)
        self._LR = np.full(shape, np.nan)

        self._H[iT, iP] = df[SP_COLS[2]].to_numpy(float)
        self._S[iT, iP] = df[SP_COLS[3]].to_numpy(float)

        rho = df[SP_COLS[4]].to_numpy(float)

        with np.errstate(invalid="ignore", divide="ignore"):
            self._LR[iT, iP] = np.where(
                rho > 0,
                np.log(rho),
                np.nan,
            )

        self._lnP = np.log(self.P)

        if self.Tmin_sat > self.T[0] * (1 + 1e-12):
            raise DataError(
                "Grid contains temperatures below the saturation "
                "table's first row"
            )

    def _sat_at_T(self, key, T):
        return float(self._sat[key](T))

    # ------------------------------------------------------------------
    # Construct per-isotherm liquid-side and vapour-side arrays
    # ------------------------------------------------------------------

    def _build_class(self, cls):
        nT, nP = len(self.T), len(self.P)

        M = nP + 1

        K = np.full((nT, M), np.nan)
        V = np.full((3, nT, M), np.nan)
        D = np.full((3, nT, M), np.nan)

        nval = np.zeros(nT, int)

        for i, Ti in enumerate(self.T):
            ok = (
                np.isfinite(self._H[i])
                & np.isfinite(self._S[i])
                & np.isfinite(self._LR[i])
            )

            x = self._lnP[ok]

            ys = [
                self._H[i][ok],
                self._S[i][ok],
                self._LR[i][ok],
            ]

            # Split an isotherm at saturation only when:
            # 1. it is below the physical critical temperature, and
            # 2. its temperature is covered by the saturation table.
            #
            # Do not extrapolate the saturation curve above Tsat_max.
            if (
                Ti <= self.Tsat_max * (1 + 1e-12)
                and Ti < self.Tcrit
            ):
                lnPs = float(self._lnPsat(Ti))

                if cls == "L":
                    sel = x > lnPs + 1e-9
                else:
                    sel = x < lnPs - 1e-9

                x = x[sel]
                ys = [y[sel] for y in ys]

                suf = "l" if cls == "L" else "v"

                a = [
                    self._sat_at_T("h" + suf, Ti),
                    self._sat_at_T("s" + suf, Ti),
                    self._sat_at_T("lr" + suf, Ti),
                ]

                if cls == "L":
                    x = np.concatenate(([lnPs], x))
                    ys = [
                        np.concatenate(([a[p]], ys[p]))
                        for p in range(3)
                    ]
                else:
                    x = np.concatenate((x, [lnPs]))
                    ys = [
                        np.concatenate((ys[p], [a[p]]))
                        for p in range(3)
                    ]

            n = len(x)

            if n >= 2:
                K[i, :n] = x

                for p in range(3):
                    V[p, i, :n] = ys[p]
                    D[p, i, :n] = (
                        PchipInterpolator(x, ys[p])
                        .derivative()(x)
                    )

            nval[i] = n

        return K, V, D, nval

    # ------------------------------------------------------------------
    # Isobar profile at pressure P
    # ------------------------------------------------------------------

    def _row_values(self, cls, lnp):
        K, V, D, nval = self._cls[cls]

        m = (K < lnp).sum(axis=1)

        m = np.where(
            (m == 0) & (np.abs(K[:, 0] - lnp) < 1e-12),
            1,
            m,
        )

        ok = (m >= 1) & (m <= nval - 1)
        idx = np.nonzero(ok)[0]

        k0 = m[idx] - 1
        k1 = m[idx]

        x0 = K[idx, k0]
        x1 = K[idx, k1]

        hh = x1 - x0
        t = (lnp - x0) / hh

        t2 = t * t
        t3 = t * t * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        out = []

        for p in range(3):
            y0 = V[p, idx, k0]
            y1 = V[p, idx, k1]
            d0 = D[p, idx, k0]
            d1 = D[p, idx, k1]

            out.append(
                h00 * y0
                + h10 * hh * d0
                + h01 * y1
                + h11 * hh * d1
            )

        return idx, out

    def _profile_uncached(self, P: float, cls: str) -> _Profile:
        idx, vals = self._row_values(cls, math.log(P))

        Tn = self.T[idx]
        keep = np.ones(len(Tn), bool)

        # A saturation anchor is valid only within the pressure coverage
        # of the actual saturation table, not all the way to Pcrit.
        subcrit = self.Pmin_sat <= P <= self.Psat_max

        if cls == "L" and P < self.Pcrit:
            keep &= Tn < self.Tcrit

        Tn = Tn[keep]
        vals = [v[keep] for v in vals]

        if subcrit:
            Ts = float(self._Tsat(math.log(P)))

            suf = "l" if cls == "L" else "v"

            a = [
                self._sat_at_T("h" + suf, Ts),
                self._sat_at_T("s" + suf, Ts),
                self._sat_at_T("lr" + suf, Ts),
            ]

            far = np.abs(Tn - Ts) > 1e-9 * Ts

            Tn = Tn[far]
            vals = [v[far] for v in vals]

            if cls == "L":
                Tn = np.append(Tn, Ts)
                vals = [
                    np.append(vals[p], a[p])
                    for p in range(3)
                ]
            else:
                Tn = np.insert(Tn, 0, Ts)
                vals = [
                    np.insert(vals[p], 0, a[p])
                    for p in range(3)
                ]

        if len(Tn) < 3 or np.any(np.diff(Tn) <= 0):
            raise OutOfRange(
                f"No usable {cls}-side data at P={P:.6g} Pa"
            )

        for name, v in (("h", vals[0]), ("s", vals[1])):
            if np.any(np.diff(v) <= 0):
                raise DataError(
                    f"{name}(T) is not strictly increasing at "
                    f"P={P:.6g} Pa ({cls} side); check the table"
                )

        return _Profile(Tn, *vals)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _class_PT(self, P, T):
        # At or above critical pressure, use the single-phase L-side
        # interpolation class; this is a numerical branch label.
        if P >= self.Pcrit:
            return "L"

        if P < self.Pmin_sat:
            return "V"

        if P <= self.Psat_max:
            Ts = float(self._Tsat(math.log(P)))
            return "L" if T < Ts else "V"

        # In this pressure interval, P is below Pcrit but above the
        # highest saturation pressure supplied by the table. The
        # saturation boundary is unavailable, so do not guess the
        # phase branch for a subcritical-temperature state.
        if T < self.Tcrit:
            raise OutOfRange(
                f"Cannot classify state at P={P:.6g} Pa, "
                f"T={T:.6g} K: it lies in the subcritical region "
                "above the saturation-table pressure limit. "
                "Extend the saturation table to support it."
            )

        return "V"

    def _sat_state(self, P):
        """
        Saturation properties at pressure P.

        Returns None when P is outside the pressure coverage of the
        saturation table.
        """
        if not (self.Pmin_sat <= P <= self.Psat_max):
            return None

        Ts = float(self._Tsat(math.log(P)))
        g = lambda k: self._sat_at_T(k, Ts)

        return {
            "T": Ts,
            "hl": g("hl"),
            "hv": g("hv"),
            "sl": g("sl"),
            "sv": g("sv"),
            "rl": math.exp(g("lrl")),
            "rv": math.exp(g("lrv")),
        }

    @staticmethod
    def _solve(f, prof: _Profile, target, what):
        a, b = prof.T0, prof.T1

        fa = f(a) - target
        fb = f(b) - target

        if fa * fb > 0:
            raise OutOfRange(
                f"{what}={target:.6g} outside tabulated range "
                "at this pressure"
            )

        if fa == 0:
            return a

        if fb == 0:
            return b

        return float(
            brentq(
                lambda T: f(T) - target,
                a,
                b,
                xtol=1e-12,
                rtol=1e-13,
            )
        )

    # ------------------------------------------------------------------
    # FluidTable interface
    # ------------------------------------------------------------------

    def h(self, P: float, T: float) -> float:
        prof = self._profile(P, self._class_PT(P, T))
        prof.check(T)
        return float(prof.h(T))

    def s(self, P: float, T: float) -> float:
        prof = self._profile(P, self._class_PT(P, T))
        prof.check(T)
        return float(prof.s(T))

    def rho(self, P: float, T: float) -> float:
        prof = self._profile(P, self._class_PT(P, T))
        prof.check(T)
        return float(math.exp(prof.lr(T)))

    def cp(self, P: float, T: float) -> float:
        """Optional: dh/dT at constant P from the interpolated table."""
        prof = self._profile(P, self._class_PT(P, T))
        prof.check(T)
        return float(prof.h.derivative()(T))

    def Psat(self, T: float) -> float:
        """
        Saturation pressure at temperature T.

        Cold-end operating temperature is restricted to <= 3.2 K,
        and the saturation table must cover the requested temperature.
        """
        Tmax_supported = min(
            self.Tsat_max,
            MAX_COLD_END_T,
        )

        if not (self.Tmin_sat <= T <= Tmax_supported):
            raise OutOfRange(
                f"Saturation T={T} K outside supported cold-end range "
                f"[{self.Tmin_sat}, {Tmax_supported}] K"
            )

        return float(math.exp(self._lnPsat(T)))

    def h_sat_vapor(self, T: float) -> float:
        """Saturated-vapour enthalpy within the supported cold-end range."""
        Tmax_supported = min(
            self.Tsat_max,
            MAX_COLD_END_T,
        )

        if not (self.Tmin_sat <= T <= Tmax_supported):
            raise OutOfRange(
                f"Saturation T={T} K outside supported cold-end range "
                f"[{self.Tmin_sat}, {Tmax_supported}] K"
            )

        return self._sat_at_T("hv", T)

    def T_from_Ph(self, P: float, h: float) -> float:
        sat = self._sat_state(P)

        if self.Psat_max < P < self.Pcrit:
            raise OutOfRange(
                f"P={P:.6g} Pa lies above the saturation-table "
                "pressure limit but below Pcrit; the saturation "
                "boundary is unavailable there."
            )

        if sat is not None:
            if sat["hl"] <= h <= sat["hv"]:
                return sat["T"]

            cls = "L" if h < sat["hl"] else "V"
        else:
            cls = "L" if P >= self.Pcrit else "V"

        prof = self._profile(P, cls)

        return self._solve(
            lambda T: float(prof.h(T)),
            prof,
            h,
            "h",
        )

    def x_v_from_Ph(self, P: float, h: float) -> Optional[float]:
        sat = self._sat_state(P)

        if sat is None or not (sat["hl"] <= h <= sat["hv"]):
            return None

        return (h - sat["hl"]) / (sat["hv"] - sat["hl"])

    def _hr_from_Ps(self, P: float, s: float):
        sat = self._sat_state(P)

        if self.Psat_max < P < self.Pcrit:
            raise OutOfRange(
                f"P={P:.6g} Pa lies above the saturation-table "
                "pressure limit but below Pcrit; the saturation "
                "boundary is unavailable there."
            )

        if sat is not None:
            if sat["sl"] <= s <= sat["sv"]:
                # Equilibrium two-phase state (HEM).
                x = (s - sat["sl"]) / (sat["sv"] - sat["sl"])

                h = sat["hl"] + x * (sat["hv"] - sat["hl"])

                rho = 1.0 / (
                    (1 - x) / sat["rl"]
                    + x / sat["rv"]
                )

                return h, rho

            cls = "L" if s < sat["sl"] else "V"

        else:
            cls = "L" if P >= self.Pcrit else "V"

        prof = self._profile(P, cls)

        T = self._solve(
            lambda T: float(prof.s(T)),
            prof,
            s,
            "s",
        )

        return (
            float(prof.h(T)),
            float(math.exp(prof.lr(T))),
        )

    def h_from_Ps(self, P: float, s: float) -> float:
        return self._hr_from_Ps(P, s)[0]

    def rho_from_Ps(self, P: float, s: float) -> float:
        return self._hr_from_Ps(P, s)[1]