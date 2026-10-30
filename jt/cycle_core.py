
"""Generic thermodynamic JT-cycle solver with replaceable property backends.

Architecture
------------
- FluidTable defines the property interface required by the solver.
- Helium-3 uses the local CSV-backed Helium3Fluid implementation.
- Other fluids use CoolProp through CoolPropFluid.
- make_fluid() selects the backend.
- The cycle calculations do not depend on the chosen backend.

Baseline assumptions
--------------------
* The cycle has n counterflow CHEXs and n-1 PHEXs.
* PHEX high-side outlet temperatures are prescribed as Tpre[0:n-1].
* CHEX and PHEX pressure drops are neglected.
* The JT restriction is isenthalpic.
* The evaporator outlet is saturated vapour at Tc, so PD = Psat(Tc).
* Mass flow is calculated from the restriction model when an orifice diameter
  is supplied; otherwise only specific performance is returned.
* Restriction mass flux uses the Homogeneous Equilibrium Model (HEM).
  The property backend must support equilibrium h(P,s) and rho(P,s) over the
  pressure range evaluated by the restriction model.

Units: SI (Pa, K, J/kg, J/(kg K), kg/m3, kg/s, W, m).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi, sqrt
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


# ---------------------------------------------------------------------------
# Fluid-property interface and state representation


class FluidTable(Protocol):
    """Property interface required by the generic JT-cycle solver."""

    def h(self, P: float, T: float) -> float:
        """Specific enthalpy [J/kg] at pressure P [Pa], temperature T [K]."""
        ...

    def T_from_Ph(self, P: float, h: float) -> float:
        """Temperature [K] from pressure [Pa] and enthalpy [J/kg]."""
        ...

    def Psat(self, T: float) -> float:
        """Saturation pressure [Pa] at temperature T [K]."""
        ...

    def h_sat_vapor(self, T: float) -> float:
        """Saturated-vapour enthalpy [J/kg] at temperature T [K]."""
        ...

    def x_v_from_Ph(self, P: float, h: float) -> Optional[float]:
        """Equilibrium vapour quality in [0, 1], or None if single-phase.

        Return 0 for saturated liquid, 1 for saturated vapour, and a value
        strictly between 0 and 1 inside the two-phase dome.
        """
        ...

    def s(self, P: float, T: float) -> float:
        """Specific entropy [J/(kg K)] at pressure P and temperature T."""
        ...

    def h_from_Ps(self, P: float, s: float) -> float:
        """Specific enthalpy [J/kg] at pressure P and entropy s."""
        ...

    def rho_from_Ps(self, P: float, s: float) -> float:
        """Density [kg/m3] at pressure P and entropy s."""
        ...


@dataclass(frozen=True)
class State:
    P: float
    T: float
    h: float
    x_v: Optional[float] = None

    @property
    def x_l(self) -> Optional[float]:
        """Liquid mass fraction; undefined for single-phase states."""
        return None if self.x_v is None else 1.0 - self.x_v

    @property
    def phase(self) -> str:
        if self.x_v is None:
            return "single_phase_or_unknown"
        if abs(self.x_v) <= 1e-12:
            return "saturated_liquid"
        if abs(self.x_v - 1.0) <= 1e-12:
            return "saturated_vapour"
        return "two_phase"


def _normalise_quality(
    fluid: FluidTable, P: float, h: float
) -> Optional[float]:
    """Return valid equilibrium quality, or None for non-dome states."""
    x_v = fluid.x_v_from_Ph(P, h)
    if x_v is None:
        return None

    x_v = float(x_v)
    if not isfinite(x_v) or x_v < -1e-10 or x_v > 1.0 + 1e-10:
        return None

    return min(1.0, max(0.0, x_v))


def state_PT(fluid: FluidTable, P: float, T: float) -> State:
    if not (isfinite(P) and P > 0 and isfinite(T) and T > 0):
        raise ValueError("State P and T must be finite and positive")

    h = fluid.h(P, T)
    if not isfinite(h):
        raise ValueError("Property backend returned non-finite enthalpy")

    return State(P=P, T=T, h=h, x_v=_normalise_quality(fluid, P, h))


def state_Ph(fluid: FluidTable, P: float, h: float) -> State:
    if not (isfinite(P) and P > 0 and isfinite(h)):
        raise ValueError("State P must be positive and h finite")

    T = fluid.T_from_Ph(P, h)
    if not (isfinite(T) and T > 0):
        raise ValueError(
            "Property backend returned invalid temperature from P,h"
        )

    return State(P=P, T=T, h=h, x_v=_normalise_quality(fluid, P, h))


def saturated_vapour_state(fluid: FluidTable, T: float) -> State:
    if not (isfinite(T) and T > 0):
        raise ValueError("Saturation temperature must be finite and positive")

    P = fluid.Psat(T)
    h = fluid.h_sat_vapor(T)

    if not (isfinite(P) and P > 0 and isfinite(h)):
        raise ValueError(
            "Property backend returned invalid saturated-vapour properties"
        )

    return State(P=P, T=T, h=h, x_v=1.0)


# ---------------------------------------------------------------------------
# Paper state numbering, generalized to n CHEXs


def state_labels(n: int) -> Dict[str, List[int] | int]:
    """Return paper state numbering; for n=4, states run from 1 to 14."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")

    return {
        "A": [2 * i - 1 for i in range(1, n + 1)],
        "B": [2 * i for i in range(1, n + 1)],
        "C": [3 * n + 2 - i for i in range(1, n + 1)],
        "D": [3 * n + 3 - i for i in range(1, n + 1)],
        "JT_in": 2 * n,
        "JT_out": 2 * n + 1,
        "evap_out": 2 * n + 2,
    }


def _validate_eta(eta: float) -> None:
    if not isfinite(eta) or not (0.0 <= eta <= 1.0):
        raise ValueError(
            f"CHEX effectiveness must lie in [0, 1]; got {eta!r}"
        )


# ---------------------------------------------------------------------------
# CHEX effectiveness models


def chex_limits(
    A: State,
    C: State,
    P_low: float,
    fluid: FluidTable,
) -> Tuple[float, float, float, str]:
    """Maximum-heat-transfer limits of a counterflow CHEX with inlets A and C.

    The heat Q transferred per unit mass flow is bounded at each end:

    * warm-end limit   W_warm = h(P_low, T_A) - h_C
      (the cold stream cannot leave warmer than the hot inlet);
    * cold-end limit   W_cold = h_A - h(P_high, T_C)
      (the hot stream cannot leave colder than the cold inlet).

    Q_max = min(W_warm, W_cold). Returns (Q_max, W_warm, W_cold, binding),
    where binding is "warm_end", "cold_end" or "both" (equal to 1e-9 relative).
    """
    h_AD = fluid.h(P_low, A.T)
    h_B_min = fluid.h(A.P, C.T)
    W_warm = h_AD - C.h
    W_cold = A.h - h_B_min
    scale = max(abs(W_warm), abs(W_cold), 1.0)
    if abs(W_warm - W_cold) <= 1e-9 * scale:
        binding = "both"
    else:
        binding = "warm_end" if W_warm < W_cold else "cold_end"
    return min(W_warm, W_cold), W_warm, W_cold, binding


def _chex_solve(
    A: State,
    C: State,
    P_low: float,
    eta: float,
    fluid: FluidTable,
    enthalpy_tol: float,
) -> Tuple[State, State, str]:
    _validate_eta(eta)

    if not (isfinite(P_low) and P_low > 0 and A.P > P_low):
        raise ValueError("CHEX requires positive low-side pressure below A.P")

    if abs(C.P - P_low) > max(1e-6, 1e-9 * P_low):
        raise ValueError(
            "C.P must equal P_low when CHEX pressure drop is neglected"
        )

    Q_max, _, _, binding = chex_limits(A, C, P_low, fluid)

    if Q_max < -enthalpy_tol:
        raise ValueError(
            "CHEX inlet states give a negative available enthalpy difference"
        )

    Q = eta * max(Q_max, 0.0)
    h_D = C.h + Q
    h_B = A.h - Q

    B = state_Ph(fluid, A.P, h_B)
    D = state_Ph(fluid, P_low, h_D)
    return B, D, binding


def chex_AC(
    A: State,
    C: State,
    P_low: float,
    eta: float,
    fluid: FluidTable,
    *,
    enthalpy_tol: float = 1e-8,
) -> Tuple[State, State]:
    """Given hot inlet A, cold inlet C and effectiveness, solve B and D.

    Effectiveness is the textbook definition, eta = Q / Q_max, where Q is the
    heat transferred per unit mass flow and Q_max = min(W_warm, W_cold) is the
    largest heat a counterflow exchanger could transfer between these two
    inlet streams (see chex_limits). eta = 1 therefore always yields a
    thermodynamically admissible exchanger: B.T >= C.T and D.T <= A.T.

    When the warm-end limit binds (W_warm <= W_cold) this equals the
    cold-side definition used by Zhang & Dang (2021), eta = (h_D - h_C) /
    (h(P_low, T_A) - h_C). The two definitions differ only when the cold-end
    limit binds, e.g. above the optimum upstream pressure of the last stage.

    Pressure is constant on each side of the CHEX. A.P is the high-side
    pressure and P_low is the low-side pressure.
    """
    B, D, _ = _chex_solve(A, C, P_low, eta, fluid, enthalpy_tol)
    return B, D


def chex_profile_check(
    A: State,
    B: State,
    C: State,
    D: State,
    fluid: FluidTable,
    *,
    n_points: int = 20,
    tol_T: float = 1e-3,
    index: Optional[int] = None,
) -> List[str]:
    """Coarse internal temperature-profile check of one counterflow CHEX.

    Both streams carry the same mass flow, so the heat q transferred from the
    cold end labels a position along the exchanger. There
        hot  stream: h = h_B + q at the high-side pressure,
        cold stream: h = h_C + q at the low-side pressure,
    for q from 0 to Q = h_A - h_B. Temperatures come from fluid.T_from_Ph, so
    no heat-capacity data is required. Flags only; nothing is rejected.

    Flags
    -----
    INTERNAL_TEMPERATURE_CROSSOVER : hot - cold < -tol_T somewhere inside.
    INTERNAL_PINCH                 : the smallest interior gap is more than
                                     tol_T below the smaller end gap.
    PINCH_CHECK_SKIPPED            : a property lookup failed.
    """
    where = f"CHEX {index}" if index is not None else "CHEX"
    Q = A.h - B.h
    if n_points < 3 or not isfinite(Q) or Q <= 0.0:
        return []

    try:
        gaps = []
        for k in range(n_points + 1):
            q = Q * k / n_points
            T_hot = fluid.T_from_Ph(A.P, B.h + q)
            T_cold = fluid.T_from_Ph(C.P, C.h + q)
            gaps.append(T_hot - T_cold)
    except Exception as exc:  # property lookup out of range, etc.
        return [f"PINCH_CHECK_SKIPPED: {where} ({type(exc).__name__}: {exc})"]

    end_gap = min(gaps[0], gaps[-1])
    k_min = min(range(1, n_points), key=lambda k: gaps[k])
    int_gap = gaps[k_min]
    flags: List[str] = []

    if int_gap < -tol_T:
        flags.append(
            f"INTERNAL_TEMPERATURE_CROSSOVER: {where} (min gap {int_gap:.4g} K "
            f"at {100.0 * k_min / n_points:.0f}% of duty)"
        )
    elif int_gap < end_gap - tol_T:
        flags.append(
            f"INTERNAL_PINCH: {where} (interior gap {int_gap:.4g} K at "
            f"{100.0 * k_min / n_points:.0f}% of duty < end gap {end_gap:.4g} K)"
        )
    return flags


def chex_terminal_flags(
    A: State,
    B: State,
    C: State,
    D: State,
    *,
    tol_T: float = 1e-6,
    index: Optional[int] = None,
) -> List[str]:
    """Check counterflow terminal temperature differences, not internal pinch."""
    where = f" CHEX {index}" if index is not None else " CHEX"
    flags: List[str] = []

    if B.T < C.T - tol_T:
        flags.append(
            f"COLD_END_TEMPERATURE_CROSSOVER:{where.strip()} "
            f"(B={B.T:.6g} K < C={C.T:.6g} K)"
        )

    if D.T > A.T + tol_T:
        flags.append(
            f"WARM_END_TEMPERATURE_CROSSOVER:{where.strip()} "
            f"(D={D.T:.6g} K > A={A.T:.6g} K)"
        )

    return flags


# ---------------------------------------------------------------------------
# Isentropic restriction / orifice model


def critical_mass_flux_HEM(
    P0: float,
    T0: float,
    P_back: float,
    fluid: FluidTable,
    n_grid: int = 500,
) -> dict:
    """Calculate restriction mass flux using the Homogeneous Equilibrium Model.

    At each trial pressure P, the backend evaluates the equilibrium state on
    the inlet isentrope s=s0:

        u(P) = sqrt(2 * (h0 - h(P, s0)))
        G(P) = rho(P, s0) * u(P)

    The maximum mass flux over P_back <= P <= P0 determines the candidate
    throat state. This calculation requires valid h(P,s) and rho(P,s) from
    the selected backend throughout the evaluated pressure range.
    """
    if not (isfinite(P0) and isfinite(P_back) and P0 > P_back > 0):
        raise ValueError("Restriction model requires P0 > P_back > 0")

    if not (isfinite(T0) and T0 > 0):
        raise ValueError(
            "Restriction upstream temperature must be finite and positive"
        )

    if not isinstance(n_grid, int) or n_grid < 10:
        raise ValueError("n_grid must be an integer >= 10")

    h0 = fluid.h(P0, T0)
    s0 = fluid.s(P0, T0)

    if not (isfinite(h0) and isfinite(s0)):
        raise ValueError(
            "Property backend returned invalid restriction inlet properties"
        )

    def evaluate(P: float) -> Tuple[float, float, float, Optional[float]]:
        """Return mass flux, density, velocity and vapour quality."""
        h = fluid.h_from_Ps(P, s0)
        rho = fluid.rho_from_Ps(P, s0)

        if not (isfinite(h) and isfinite(rho) and rho > 0):
            raise ValueError(
                "Property backend returned invalid h(P,s) or rho(P,s) "
                "during HEM expansion"
            )

        dh = h0 - h
        # Round-trip noise of h(P, s(P0,T0)) at the stagnation pressure can exceed
        # 1e-9 relative; 1e-3 J/kg is physically negligible (u ~ 0.04 m/s).
        energy_tol = 1e-9 * max(abs(h0), abs(h), 1.0) + 1e-3

        if dh < -energy_tol:
            raise ValueError("Isentropic expansion returned h > h0")

        u = sqrt(max(2.0 * dh, 0.0))
        G = rho * u

        if not (isfinite(G) and G >= 0):
            raise ValueError("HEM calculation returned invalid mass flux")

        x_v = _normalise_quality(fluid, P, h)
        return G, rho, u, x_v

    # Logarithmic pressure grid, including both interval boundaries.
    Ps = [
        P_back * (P0 / P_back) ** (k / n_grid)
        for k in range(n_grid + 1)
    ]

    samples = [evaluate(P) for P in Ps]
    Gs = [sample[0] for sample in samples]
    k_max = max(range(len(Gs)), key=Gs.__getitem__)

    if k_max == 0:
        P_star = P_back
        choked = False
    elif k_max == n_grid:
        raise ValueError(
            "Mass-flux maximum occurred at upstream stagnation pressure; "
            "check the property backend and inlet state"
        )
    else:
        # Refine the local maximum using golden-section maximization.
        lo, hi = Ps[k_max - 1], Ps[k_max + 1]
        phi = (sqrt(5.0) - 1.0) / 2.0
        x1 = hi - phi * (hi - lo)
        x2 = lo + phi * (hi - lo)
        f1 = evaluate(x1)[0]
        f2 = evaluate(x2)[0]

        for _ in range(80):
            if f1 < f2:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + phi * (hi - lo)
                f2 = evaluate(x2)[0]
            else:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - phi * (hi - lo)
                f1 = evaluate(x1)[0]

        refined_P = 0.5 * (lo + hi)
        candidates = [Ps[k_max - 1], Ps[k_max], Ps[k_max + 1], refined_P]
        P_star = max(candidates, key=lambda P: evaluate(P)[0])
        choked = (
            P_star > P_back * (1.0 + 1e-12)
            and P_star < P0 * (1.0 - 1e-12)
        )

        if abs(P_star - P_back) <= 1e-10 * P_back:
            P_star = P_back
            choked = False

    G_star, rho_star, u_star, x_v_star = evaluate(P_star)
    a_star = None
    sonic_relative_error = None

    # Sound speed is optional and diagnostic only.
    if x_v_star is None:
        a_method = getattr(fluid, "a_from_Ps", None)
        if callable(a_method):
            try:
                a_value = float(a_method(P_star, s0))
                if isfinite(a_value) and a_value > 0:
                    a_star = a_value
                    if choked:
                        sonic_relative_error = abs(u_star - a_star) / a_star
            except Exception:
                # Sound speed is not required for the HEM mass-flux calculation.
                a_star = None
                sonic_relative_error = None

    if x_v_star is None:
        phase = "single_phase_or_unknown"
        x_l_star = None
    elif x_v_star <= 1e-12:
        phase = "saturated_liquid"
        x_l_star = 1.0
    elif x_v_star >= 1.0 - 1e-12:
        phase = "saturated_vapour"
        x_l_star = 0.0
    else:
        phase = "two_phase"
        x_l_star = 1.0 - x_v_star

    return {
        "model": "HEM",
        "G": G_star,
        "P_star": P_star,
        "rho": rho_star,
        "a": a_star,
        "u": u_star,
        "choked": choked,
        "phase": phase,
        "x_v": x_v_star,
        "x_l": x_l_star,
        "sonic_relative_error": sonic_relative_error,
    }


def critical_mass_flux(
    P0: float,
    T0: float,
    P_low: float,
    fluid: FluidTable,
    n_grid: int = 500,
) -> dict:
    """Compatibility name for the HEM restriction model."""
    return critical_mass_flux_HEM(P0, T0, P_low, fluid, n_grid=n_grid)


def mass_flow(
    P0: float,
    T0: float,
    P_low: float,
    d: float,
    fluid: FluidTable,
) -> Tuple[float, dict]:
    """Calculate mdot=A*G using HEM and baseline discharge coefficient Cd=1."""
    if not (isfinite(d) and d > 0):
        raise ValueError("Orifice diameter d must be finite and positive")

    throat = critical_mass_flux(P0, T0, P_low, fluid)
    mdot = (pi / 4.0) * d**2 * throat["G"]

    if not (isfinite(mdot) and mdot >= 0):
        raise ValueError("Restriction model returned invalid mass flow")

    return mdot, throat


# ---------------------------------------------------------------------------
# Cycle result and solver


@dataclass
class CycleResult:
    states: Dict[int, State]
    q_C: float
    q_pre: List[float]
    P_D: float
    energy_residual: float
    flags: List[str] = field(default_factory=list)
    mdot: Optional[float] = None
    Q_C: Optional[float] = None
    Q_pre: Optional[List[float]] = None
    throat: Optional[dict] = None
    x_l_JT_outlet: Optional[float] = None
    chex_binding: List[str] = field(default_factory=list)


def solve_cycle(
    fluid: FluidTable,
    Tc: float,
    Pu: float,
    Th: float,
    Tpre: Sequence[float],
    eta: Sequence[float],
    d: Optional[float] = None,
    mdot_given: Optional[float] = None,
    tol_T: float = 1e-6,
    tol_energy_rel: float = 1e-8,
    pinch_points: int = 20,
    tol_pinch_T: float = 1e-3,
) -> CycleResult:
    """Solve the prescribed-Tpre cycle for n CHEXs and n-1 PHEXs.

    The user cannot prescribe mass flow. If d is supplied, mass flow is
    calculated using the HEM restriction model. If d is None, only specific
    duties are returned and total powers remain None.
    """
    n = len(eta)

    if n < 1:
        raise ValueError("eta must contain at least one CHEX effectiveness")

    if len(Tpre) != n - 1:
        raise ValueError("n CHEXs require exactly n-1 precooling temperatures")

    if not (
        isfinite(Tc) and Tc > 0
        and isfinite(Pu) and Pu > 0
        and isfinite(Th) and Th > 0
    ):
        raise ValueError("Tc, Pu and Th must be finite and positive")

    if any(not isfinite(float(t)) or float(t) <= 0 for t in Tpre):
        raise ValueError("All precooling temperatures must be finite and positive")

    for effectiveness in eta:
        _validate_eta(float(effectiveness))

    if d is not None and (not isfinite(d) or d <= 0):
        raise ValueError("Orifice diameter must be finite and positive")

    if mdot_given is not None:
        if d is not None:
            raise ValueError("give either an orifice diameter d or a measured mass flow mdot_given, not both")
        if not isfinite(mdot_given) or mdot_given <= 0:
            raise ValueError("mdot_given must be finite and positive [kg/s]")

    if not (
        isfinite(tol_T) and tol_T >= 0
        and isfinite(tol_energy_rel) and tol_energy_rel > 0
    ):
        raise ValueError("Invalid validation tolerance")

    lab = state_labels(n)
    flags: List[str] = []

    # Boundary states and pressure consistency.
    C = saturated_vapour_state(fluid, Tc)
    P_D = C.P

    if P_D >= Pu:
        raise ValueError(
            "Evaporator saturation pressure must be below high-side pressure Pu"
        )

    # A_i temperatures: A_1 at ambient, A_2...A_n after preceding PHEXs.
    T_in = [Th] + [float(t) for t in Tpre]
    states: Dict[int, State] = {int(lab["evap_out"]): C}
    A_list: List[Optional[State]] = [None] * n
    B_list: List[Optional[State]] = [None] * n
    binding_list: List[str] = [""] * n

    # Backward sweep: C_n is known; solve CHEX n, n-1, ..., 1.
    for i in range(n - 1, -1, -1):
        A = state_PT(fluid, Pu, T_in[i])
        B, D, binding = _chex_solve(A, C, P_D, float(eta[i]), fluid, 1e-8)
        binding_list[i] = binding

        states[int(lab["A"][i])] = A
        states[int(lab["B"][i])] = B
        states[int(lab["D"][i])] = D
        A_list[i], B_list[i] = A, B

        if i > 0:
            C = D
            states[int(lab["C"][i - 1])] = C

    # Validate terminal temperature differences at every CHEX.
    for i in range(n):
        A = states[int(lab["A"][i])]
        B = states[int(lab["B"][i])]
        C_i = states[int(lab["C"][i])]
        D = states[int(lab["D"][i])]
        flags.extend(
            chex_terminal_flags(A, B, C_i, D, tol_T=tol_T, index=i + 1)
        )
        if pinch_points:
            flags.extend(
                chex_profile_check(
                    A, B, C_i, D, fluid,
                    n_points=int(pinch_points), tol_T=tol_pinch_T, index=i + 1,
                )
            )

    # JT inlet is B_n; throttling is isenthalpic and outlet pressure is P_D.
    JT_in = B_list[n - 1]
    assert JT_in is not None

    states[int(lab["JT_out"])] = state_Ph(fluid, P_D, JT_in.h)
    JT_out = states[int(lab["JT_out"])]

    q_C = float(states[int(lab["evap_out"])].h - JT_out.h)
    if q_C <= 0:
        flags.append("NONPOSITIVE_REFRIGERATION: q_C <= 0 J/kg")

    # PHEX j cools high-side stream B_j to prescribed state A_(j+1).
    q_pre = [
        float(B_list[j].h - A_list[j + 1].h)
        for j in range(n - 1)
    ]  # type: ignore[union-attr]

    if any(q < 0 for q in q_pre):
        flags.append("NEGATIVE_PHEX_LOAD: at least one calculated q_pre is negative")

    # Global balance: q_C = sum(q_pre) + (h_D1 - h_A1).
    D1 = states[int(lab["D"][0])]
    A1 = states[int(lab["A"][0])]
    R = q_C - sum(q_pre) - (D1.h - A1.h)
    scale = max(
        abs(q_C),
        sum(abs(q) for q in q_pre),
        abs(D1.h - A1.h),
        1.0,
    )

    if abs(R) > tol_energy_rel * scale:
        flags.append(f"GLOBAL_ENERGY_BALANCE: residual {R:.6g} J/kg")

    # Calculate flow and total duties if an orifice diameter is supplied.
    throat = None
    mdot = None

    if d is not None:
        mdot, throat = mass_flow(Pu, JT_in.T, P_D, d, fluid)

        if not throat["choked"]:
            flags.append("UNCHOKED_RESTRICTION_FLOW")

        if (
            throat["choked"]
            and throat["sonic_relative_error"] is not None
            and throat["sonic_relative_error"] > 1e-2
        ):
            flags.append(
                "CHOKING_CHECK: maximum mass flux does not closely satisfy u=a"
            )

    if mdot_given is not None:
        mdot = float(mdot_given)          # measured / prescribed flow: no restriction model, no throat data

    Q_C = mdot * q_C if mdot is not None else None
    Q_pre = [mdot * q for q in q_pre] if mdot is not None else None

    return CycleResult(
        states=states,
        q_C=q_C,
        q_pre=q_pre,
        P_D=P_D,
        energy_residual=R,
        flags=flags,
        mdot=mdot,
        Q_C=Q_C,
        Q_pre=Q_pre,
        throat=throat,
        x_l_JT_outlet=JT_out.x_l,
        chex_binding=binding_list,
    )


# ---------------------------------------------------------------------------
# CoolProp adapter


class CoolPropFluid:
    """FluidTable adapter for fluids supported by the installed CoolProp build.

    The supplied CoolProp fluid name is passed directly to PropsSI. The name
    must identify the intended fluid; "Helium" must not be assumed to represent
    helium-3.
    """

    def __init__(self, name: str):
        from CoolProp.CoolProp import PropsSI

        self._p = PropsSI
        self.name = name

    def h(self, P: float, T: float) -> float:
        return float(self._p("H", "P", P, "T", T, self.name))

    def T_from_Ph(self, P: float, h: float) -> float:
        return float(self._p("T", "P", P, "H", h, self.name))

    def Psat(self, T: float) -> float:
        return float(self._p("P", "T", T, "Q", 1, self.name))

    def h_sat_vapor(self, T: float) -> float:
        return float(self._p("H", "T", T, "Q", 1, self.name))

    def x_v_from_Ph(self, P: float, h: float) -> Optional[float]:
        try:
            x_v = float(self._p("Q", "P", P, "H", h, self.name))
        except Exception:
            # Quality is undefined for some single-phase states and limits.
            return None

        if 0.0 <= x_v <= 1.0:
            return x_v

        return None

    def s(self, P: float, T: float) -> float:
        return float(self._p("S", "P", P, "T", T, self.name))

    def h_from_Ps(self, P: float, s: float) -> float:
        return float(self._p("H", "P", P, "S", s, self.name))

    def rho_from_Ps(self, P: float, s: float) -> float:
        return float(self._p("D", "P", P, "S", s, self.name))

    def a_from_Ps(self, P: float, s: float) -> float:
        return float(self._p("A", "P", P, "S", s, self.name))


# ---------------------------------------------------------------------------
# Backend selection


def make_fluid(
    name: str,
    *,
    singlephase_csv: Optional[str] = None,
    saturation_csv: Optional[str] = None,
) -> FluidTable:
    """Create the property backend appropriate for the requested fluid.

    Names "He3", "Helium3", and "Helium-3" select Helium3Fluid, which reads
    the local CSV property tables. Any other name is passed to CoolProp.

    CSV arguments apply only to the helium-3 backend. If omitted, the
    Helium3Fluid class uses its own default CSV paths.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Fluid name must be a non-empty string")

    normalized = (
        name.strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    if normalized in {"he3", "helium3"}:
        from jt.properties import Helium3Fluid

        return Helium3Fluid(
            singlephase_csv=singlephase_csv,
            saturation_csv=saturation_csv,
        )

    if singlephase_csv is not None or saturation_csv is not None:
        raise ValueError(
            "CSV paths can only be supplied when selecting helium-3"
        )

    return CoolPropFluid(name.strip())
