"""Core thermodynamic JT-cycle solver for n CHEXs and n-1 PHEXs.

Baseline assumptions
--------------------
* The cycle has n counterflow CHEXs and n-1 PHEXs.
* PHEX high-side outlet temperatures are prescribed as T_pre[0:n-1].
* CHEX pressure drops and PHEX pressure drops are neglected.
* The JT restriction is isenthalpic.
* The evaporator outlet is saturated vapour at T_C, so P_D = P_sat(T_C).
* Mass flow is never a user input. If an orifice diameter is supplied, it is
  calculated by the restriction model; otherwise only specific performance
  (J/kg) is returned.
* The restriction uses the Homogeneous Equilibrium Model (HEM): the isentropic
  expansion path may pass through the two-phase region, provided the property
  backend can evaluate equilibrium h(P,s) and rho(P,s) there.

Units: SI (Pa, K, J/kg, kg/s, W, m).

The FluidTable protocol is deliberately small and replaceable. A future EOS,
property library, or tabular backend can implement it without changing the
cycle topology or component calculations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi, sqrt
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


# ---------------------------------------------------------------------------
# Fluid-property interface and state representation

class FluidTable(Protocol):
    def h(self, P: float, T: float) -> float:
        """Specific enthalpy [J/kg] at pressure P [Pa], temperature T [K]."""
        ...

    def T_from_Ph(self, P: float, h: float) -> float:
        """Temperature [K] from pressure and specific enthalpy."""
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
        strictly between 0 and 1 inside the two-phase dome. Do not return -1
        for a single-phase state; normalize that to None in the adapter.
        """
        ...

    # These methods are required by the HEM restriction model.
    def s(self, P: float, T: float) -> float:
        ...

    def h_from_Ps(self, P: float, s: float) -> float:
        ...

    def rho_from_Ps(self, P: float, s: float) -> float:
        ...

    # An optional, single-phase diagnostic only. HEM does not need sound speed
    # to locate its mass-flux maximum, so this method is intentionally optional.


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


def _normalise_quality(fluid: FluidTable, P: float, h: float) -> Optional[float]:
    """Return a valid equilibrium quality, normalizing non-dome results to None."""
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
        raise ValueError("State P must be finite and positive and h finite")
    T = fluid.T_from_Ph(P, h)
    if not (isfinite(T) and T > 0):
        raise ValueError("Property backend returned invalid temperature from P,h")
    return State(P=P, T=T, h=h, x_v=_normalise_quality(fluid, P, h))


def saturated_vapour_state(fluid: FluidTable, T: float) -> State:
    if not (isfinite(T) and T > 0):
        raise ValueError("Saturation temperature must be finite and positive")
    P = fluid.Psat(T)
    h = fluid.h_sat_vapor(T)
    if not (isfinite(P) and P > 0 and isfinite(h)):
        raise ValueError("Property backend returned invalid saturated-vapour properties")
    return State(P=P, T=T, h=h, x_v=1.0)


# ---------------------------------------------------------------------------
# Paper state numbering, generalized to n CHEXs

def state_labels(n: int) -> Dict[str, List[int] | int]:
    """Return the paper numbering; for n=4, the states run from 1 to 14."""
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
        raise ValueError(f"CHEX effectiveness must lie in [0, 1]; got {eta!r}")


# ---------------------------------------------------------------------------
# CHEX effectiveness models

def chex_AC(
    A: State,
    C: State,
    P_low: float,
    eta: float,
    fluid: FluidTable,
    *,
    enthalpy_tol: float = 1e-8,
) -> Tuple[State, State]:
    """Given A (hot inlet), C (cold inlet), and effectiveness, solve B and D.

    Baseline assumptions: the pressure on each side is constant through the
    CHEX, with A.P the high-side pressure and P_low the low-side pressure.
    """
    _validate_eta(eta)
    if not (isfinite(P_low) and P_low > 0 and A.P > P_low):
        raise ValueError("CHEX requires positive low-side pressure below A.P")
    if abs(C.P - P_low) > max(1e-6, 1e-9 * P_low):
        raise ValueError("C.P must equal P_low when CHEX pressure drop is neglected")

    h_AD = fluid.h(P_low, A.T)
    available = h_AD - C.h
    if available < -enthalpy_tol:
        raise ValueError("CHEX inlet states give a negative available enthalpy difference")

    # A zero available difference implies zero possible heat transfer; the
    # effectiveness relation is degenerate but the outlet enthalpies are clear.
    if abs(available) <= enthalpy_tol:
        h_D = C.h
    else:
        h_D = C.h + eta * available

    h_B = A.h - (h_D - C.h)  # CHEX energy balance
    B = state_Ph(fluid, A.P, h_B)
    D = state_Ph(fluid, P_low, h_D)
    return B, D


def chex_AB(
    A: State,
    B: State,
    P_low: float,
    eta: float,
    fluid: FluidTable,
) -> Tuple[State, State]:
    """Given A, B, and effectiveness, infer C and D.

    Useful for measured/characterized CHEXs and round-trip tests. The inverse
    problem is not identifiable at eta=0, so that case is rejected.
    """
    _validate_eta(eta)
    if eta == 0.0:
        raise ValueError("Cannot infer C and D from A and B when eta=0")
    if not (isfinite(P_low) and P_low > 0 and A.P > P_low):
        raise ValueError("CHEX requires positive low-side pressure below A.P")
    if abs(B.P - A.P) > max(1e-6, 1e-9 * A.P):
        raise ValueError("A.P must equal B.P when CHEX pressure drop is neglected")

    dh = A.h - B.h
    h_AD = fluid.h(P_low, A.T)
    h_C = h_AD - dh / eta
    h_D = h_C + dh
    C = state_Ph(fluid, P_low, h_C)
    D = state_Ph(fluid, P_low, h_D)
    return C, D


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

    The inlet stagnation state defines h0 and s0. At each trial pressure P,
    the backend evaluates the equilibrium state on the isentrope s=s0, including
    inside the two-phase dome when supported:

        u(P) = sqrt(2 * (h0 - h(P, s0)))
        G(P) = rho(P, s0) * u(P)

    The candidate critical mass flux is the maximum of G over
    P_back <= P <= P0. An interior maximum indicates choked flow; if the maximum
    occurs at P_back, the flow is treated as unchoked. No speed-of-sound value
    or numerical differentiation is required to locate the maximum.

    This is the homogeneous-equilibrium model: phases are assumed to be in
    local thermodynamic equilibrium and to share a common velocity. Its validity
    depends on the fluid, restriction geometry, and operating regime. The
    property backend MUST support equilibrium h(P,s) and rho(P,s) throughout
    any two-phase part of the expansion path.

    Returns a dictionary containing G, P_star, rho, u, choked, x_v, x_l,
    phase, and (for single-phase states only, if the backend provides it) a and
    sonic_relative_error. The speed of sound is diagnostic only; it is not used
    in the optimization and is never numerically differentiated here.
    """
    if not (isfinite(P0) and isfinite(P_back) and P0 > P_back > 0):
        raise ValueError("Restriction model requires P0 > P_back > 0")
    if not (isfinite(T0) and T0 > 0):
        raise ValueError("Restriction upstream temperature must be finite and positive")
    if not isinstance(n_grid, int) or n_grid < 10:
        raise ValueError("n_grid must be an integer >= 10")

    h0 = fluid.h(P0, T0)
    s0 = fluid.s(P0, T0)
    if not (isfinite(h0) and isfinite(s0)):
        raise ValueError("Property backend returned invalid restriction inlet properties")

    def evaluate(P: float) -> Tuple[float, float, float, Optional[float]]:
        """Return (mass flux, density, velocity, vapour quality) on s=s0."""
        h = fluid.h_from_Ps(P, s0)
        rho = fluid.rho_from_Ps(P, s0)
        if not (isfinite(h) and isfinite(rho) and rho > 0):
            raise ValueError(
                "Property backend returned invalid h(P,s) or rho(P,s) during HEM expansion"
            )

        dh = h0 - h
        energy_tol = 1e-9 * max(abs(h0), abs(h), 1.0)
        if dh < -energy_tol:
            raise ValueError("Isentropic expansion returned h > h0")

        u = sqrt(max(2.0 * dh, 0.0))
        G = rho * u
        if not (isfinite(G) and G >= 0):
            raise ValueError("HEM calculation returned invalid mass flux")

        # Quality is diagnostic: HEM flux uses equilibrium density and enthalpy
        # directly. A missing quality result does not invalidate otherwise-valid
        # equilibrium h and rho evaluations, but phase metadata then remains unknown.
        x_v = _normalise_quality(fluid, P, h)
        return G, rho, u, x_v

    # Include both boundaries. A logarithmic pressure grid resolves the broad
    # pressure range common in cryogenic JT systems.
    Ps = [P_back * (P0 / P_back) ** (k / n_grid) for k in range(n_grid + 1)]
    samples = [evaluate(P) for P in Ps]
    Gs = [sample[0] for sample in samples]
    k_max = max(range(len(Gs)), key=Gs.__getitem__)

    if k_max == 0:
        # Highest mass flux available in the allowed interval is at back pressure.
        P_star = P_back
        choked = False
    elif k_max == n_grid:
        # At P=P0, ideal velocity should be zero. Reaching the maximum here
        # indicates a degenerate/inconsistent property path rather than a throat.
        raise ValueError(
            "Mass-flux maximum occurred at upstream stagnation pressure; "
            "check the property backend and inlet state"
        )
    else:
        # Refine the bracket surrounding the largest grid sample. Golden-section
        # maximization uses only property evaluations; no derivatives are used.
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
        # Compare the refined point to the local grid points as a safeguard
        # around the saturation boundary, where the property curve may have a kink.
        candidate_Ps = [Ps[k_max - 1], Ps[k_max], Ps[k_max + 1], refined_P]
        P_star = max(candidate_Ps, key=lambda P: evaluate(P)[0])
        choked = P_star > P_back * (1.0 + 1e-12) and P_star < P0 * (1.0 - 1e-12)
        if not choked:
            # A numerically boundary-located maximum is not considered choked.
            choked = False
            if abs(P_star - P_back) <= 1e-10 * P_back:
                P_star = P_back

    G_star, rho_star, u_star, x_v_star = evaluate(P_star)
    a_star = None
    sonic_relative_error = None

    # CoolProp's ordinary speed of sound is a single-phase diagnostic only. It
    # is not appropriate to substitute either saturated-phase sound speed for a
    # two-phase equilibrium sound speed, and HEM optimization does not need it.
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
                # Mass flux is already determined by the HEM maximization; a
                # diagnostic sound-speed failure must not discard that result.
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
    """Backward-compatible name; now uses the HEM restriction model."""
    return critical_mass_flux_HEM(P0, T0, P_low, fluid, n_grid=n_grid)


def mass_flow(
    P0: float,
    T0: float,
    P_low: float,
    d: float,
    fluid: FluidTable,
) -> Tuple[float, dict]:
    """Mass flow mdot=A*G using HEM, with baseline discharge coefficient Cd=1."""
    if not (isfinite(d) and d > 0):
        raise ValueError("Orifice diameter d must be finite and positive")
    throat = critical_mass_flux(P0, T0, P_low, fluid)
    # Zhang and Dang's baseline formulation does not include a discharge
    # coefficient, so the geometric orifice area is used directly (Cd = 1).
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


def solve_cycle(
    fluid: FluidTable,
    Tc: float,
    Pu: float,
    Th: float,
    Tpre: Sequence[float],
    eta: Sequence[float],
    d: Optional[float] = None,
    tol_T: float = 1e-6,
    tol_energy_rel: float = 1e-8,
) -> CycleResult:
    """Solve the prescribed-Tpre baseline cycle for n CHEXs, n-1 PHEXs.

    The user cannot prescribe mdot. The baseline fixes Cd=1 as in the paper's
    mass-flow expression. If d is supplied, mdot is calculated by the HEM
    restriction model, including through a two-phase isentropic region
    when the property backend supports equilibrium h(P,s) and rho(P,s) there.
    If d is None, only specific duties (J/kg) are returned and total powers
    remain None.
    """
    n = len(eta)
    if n < 1:
        raise ValueError("eta must contain at least one CHEX effectiveness")
    if len(Tpre) != n - 1:
        raise ValueError("n CHEXs require exactly n-1 precooling temperatures")
    if not (isfinite(Tc) and Tc > 0 and isfinite(Pu) and Pu > 0 and isfinite(Th) and Th > 0):
        raise ValueError("Tc, Pu and Th must be finite and positive")
    if any(not isfinite(float(t)) or float(t) <= 0 for t in Tpre):
        raise ValueError("All precooling temperatures must be finite and positive")
    for effectiveness in eta:
        _validate_eta(float(effectiveness))
    if d is not None and (not isfinite(d) or d <= 0):
        raise ValueError("Orifice diameter must be finite and positive")
    if not (isfinite(tol_T) and tol_T >= 0 and isfinite(tol_energy_rel) and tol_energy_rel > 0):
        raise ValueError("Invalid validation tolerance")

    lab = state_labels(n)
    flags: List[str] = []

    # Boundary states and pressure consistency.
    C = saturated_vapour_state(fluid, Tc)
    P_D = C.P
    if P_D >= Pu:
        raise ValueError("Evaporator saturation pressure must be below high-side pressure Pu")

    # A_i temperatures: A_1 at ambient, A_2...A_n after preceding PHEXs.
    T_in = [Th] + [float(t) for t in Tpre]
    states: Dict[int, State] = {int(lab["evap_out"]): C}
    A_list: List[Optional[State]] = [None] * n
    B_list: List[Optional[State]] = [None] * n

    # Backward sweep: C_n is known; solve CHEX n, n-1, ..., 1.
    for i in range(n - 1, -1, -1):
        A = state_PT(fluid, Pu, T_in[i])
        B, D = chex_AC(A, C, P_D, float(eta[i]), fluid)
        states[int(lab["A"][i])] = A
        states[int(lab["B"][i])] = B
        states[int(lab["D"][i])] = D
        A_list[i], B_list[i] = A, B
        if i > 0:
            C = D  # return outlet from CHEX i is inlet to CHEX i-1
            states[int(lab["C"][i - 1])] = C

    # Validate terminal temperature differences at every CHEX.
    for i in range(n):
        A = states[int(lab["A"][i])]
        B = states[int(lab["B"][i])]
        C_i = states[int(lab["C"][i])]
        D = states[int(lab["D"][i])]
        flags.extend(chex_terminal_flags(A, B, C_i, D, tol_T=tol_T, index=i + 1))

    # JT inlet is B_n, throttle is isenthalpic, and the outlet is at P_D.
    JT_in = B_list[n - 1]
    assert JT_in is not None
    states[int(lab["JT_out"])] = state_Ph(fluid, P_D, JT_in.h)
    JT_out = states[int(lab["JT_out"])]
    q_C = float(states[int(lab["evap_out"])].h - JT_out.h)
    if q_C <= 0:
        flags.append("NONPOSITIVE_REFRIGERATION: q_C <= 0 J/kg")

    # PHEX j cools high-side stream B_j to the prescribed state A_(j+1).
    q_pre = [float(B_list[j].h - A_list[j + 1].h) for j in range(n - 1)]  # type: ignore[union-attr]
    if any(q < 0 for q in q_pre):
        flags.append("NEGATIVE_PHEX_LOAD: at least one calculated q_pre is negative")

    # Global balance: q_C = sum(q_pre) + (h_D1 - h_A1).
    D1 = states[int(lab["D"][0])]
    A1 = states[int(lab["A"][0])]
    R = q_C - sum(q_pre) - (D1.h - A1.h)
    scale = max(abs(q_C), sum(abs(q) for q in q_pre), abs(D1.h - A1.h), 1.0)
    if abs(R) > tol_energy_rel * scale:
        flags.append(f"GLOBAL_ENERGY_BALANCE: residual {R:.6g} J/kg")

    # Flow and total duties.
    throat = None
    mdot = None
    if d is not None:
        # No user-supplied mass flow: derive it from the HEM restriction model.
        mdot, throat = mass_flow(Pu, JT_in.T, P_D, d, fluid)
        if not throat["choked"]:
            flags.append("UNCHOKED_RESTRICTION_FLOW")
        if throat["choked"] and throat["sonic_relative_error"] is not None:
            if throat["sonic_relative_error"] > 1e-2:
                flags.append("CHOKING_CHECK: maximum mass flux does not closely satisfy u=a")

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
    )


# ---------------------------------------------------------------------------
# CoolProp adapter (only for fluids supported by the installed CoolProp build)

class CoolPropFluid:
    """FluidTable adapter for CoolProp-supported fluids.

    Note: the selected CoolProp fluid name must represent the actual working
    fluid. Do not assume that the standard "Helium" entry is a validated He-3
    equation of state.
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
            # Quality is undefined for some single-phase states or near certain
            # limits. A property failure for h/T/rho remains a hard error.
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
