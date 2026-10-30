import math
import unittest

from jt.cycle_core import (
    State, chex_AC, chex_limits, chex_profile_check, critical_mass_flux,
    critical_mass_flux_HEM, solve_cycle, state_labels,
)


class IdealGasTestFluid:
    """Lightweight software-test fluid; not a helium property model."""
    cp = 1000.0
    R = 200.0
    gamma = cp / (cp - R)

    def h(self, P, T):
        return self.cp * T

    def T_from_Ph(self, P, h):
        return h / self.cp

    def Psat(self, T):
        return 100.0 * T

    def h_sat_vapor(self, T):
        return self.cp * T

    def x_v_from_Ph(self, P, h):
        return None

    def s(self, P, T):
        return self.cp * math.log(T) - self.R * math.log(P)

    def h_from_Ps(self, P, s):
        T = math.exp((s + self.R * math.log(P)) / self.cp)
        return self.cp * T

    def rho_from_Ps(self, P, s):
        T = math.exp((s + self.R * math.log(P)) / self.cp)
        return P / (self.R * T)

    def a_from_Ps(self, P, s):
        T = math.exp((s + self.R * math.log(P)) / self.cp)
        return math.sqrt(self.gamma * self.R * T)


class TestJTCore(unittest.TestCase):
    def setUp(self):
        self.fluid = IdealGasTestFluid()

    def test_labels_four_stages(self):
        labels = state_labels(4)
        self.assertEqual(labels["A"], [1, 3, 5, 7])
        self.assertEqual(labels["B"], [2, 4, 6, 8])
        self.assertEqual(labels["C"], [13, 12, 11, 10])
        self.assertEqual(labels["D"], [14, 13, 12, 11])
        self.assertEqual(labels["JT_out"], 9)
        self.assertEqual(labels["evap_out"], 10)

    def test_chex_forward_ideal_gas(self):
        f = self.fluid
        A = State(P=1.0e6, T=300.0, h=f.h(1.0e6, 300.0))
        C = State(P=1.0e4, T=10.0, h=f.h(1.0e4, 10.0))
        B, D = chex_AC(A, C, C.P, 0.8, f)
        self.assertAlmostEqual(B.T, 68.0, places=8)
        self.assertAlmostEqual(D.T, 242.0, places=8)
        # Equal heat capacities: both walls coincide.
        Q_max, W_warm, W_cold, binding = chex_limits(A, C, C.P, f)
        self.assertAlmostEqual(W_warm, W_cold, places=6)
        self.assertEqual(binding, "both")

    def test_eta_one_cannot_cross_cold_end(self):
        class CpMismatchFluid(IdealGasTestFluid):
            # Liquid-like high-pressure side: smaller cp than the low side,
            # so the cold-end wall is closer than the warm-end wall.
            def h(self, P, T):
                return (500.0 if P > 5.0e5 else 1000.0) * T

            def T_from_Ph(self, P, h):
                return h / (500.0 if P > 5.0e5 else 1000.0)

        f = CpMismatchFluid()
        A = State(P=1.0e6, T=100.0, h=f.h(1.0e6, 100.0))
        C = State(P=1.0e4, T=20.0, h=f.h(1.0e4, 20.0))
        Q_max, W_warm, W_cold, binding = chex_limits(A, C, C.P, f)
        self.assertEqual(binding, "cold_end")
        self.assertLess(W_cold, W_warm)
        B, D = chex_AC(A, C, C.P, 1.0, f)
        self.assertGreaterEqual(B.T, C.T - 1e-9)   # no cold-end crossover
        self.assertAlmostEqual(B.T, C.T, places=8)  # touches the cold wall
        self.assertLess(D.T, A.T)                   # warm-end wall not reached
        # The cold-side-only definition would have forced B.T below C.T.
        h_D_old = C.h + (f.h(C.P, A.T) - C.h)
        self.assertLess(A.h - (h_D_old - C.h), f.h(A.P, C.T))

    def test_profile_check_flags_interior_crossing(self):
        class KinkFluid(IdealGasTestFluid):
            def T_from_Ph(self, P, h):
                if P > 5.0e5:
                    return h / 1000.0
                # Low side warms very quickly in the middle of the duty.
                return h / 1000.0 + (30.0 if h > 40000.0 else 0.0) * min((h - 40000.0) / 5000.0, 1.0)

        f = KinkFluid()
        A = State(P=1.0e6, T=100.0, h=100000.0)
        B = State(P=1.0e6, T=20.0, h=20000.0)
        C = State(P=1.0e4, T=10.0, h=10000.0)
        D = State(P=1.0e4, T=90.0, h=90000.0)
        flags = chex_profile_check(A, B, C, D, f)
        self.assertTrue(any("INTERNAL" in x for x in flags))
        ok = chex_profile_check(A, B, C, D, self.fluid)
        self.assertEqual(ok, [])

    def test_cycle_four_chex(self):
        result = solve_cycle(
            self.fluid,
            Tc=1.0,
            Pu=1.0e6,
            Th=300.0,
            Tpre=[200.0, 100.0, 20.0],
            eta=[0.8, 0.8, 0.8, 0.8],
            d=1.0e-4,
        )
        self.assertEqual(len(result.q_pre), 3)
        self.assertTrue(abs(result.energy_residual) < 1e-6)
        self.assertAlmostEqual(result.states[9].h, result.states[8].h, places=8)
        self.assertIsNotNone(result.mdot)
        self.assertGreater(result.mdot, 0.0)
        self.assertIsNotNone(result.Q_C)
        self.assertEqual(len(result.states), 14)

    def test_choked_and_unchoked_branches(self):
        f = self.fluid
        choked = critical_mass_flux(1.0e6, 300.0, 1.0e4, f)
        unchoked = critical_mass_flux(1.0e6, 300.0, 9.0e5, f)
        self.assertTrue(choked["choked"])
        self.assertFalse(unchoked["choked"])
        self.assertAlmostEqual(unchoked["P_star"], 9.0e5, delta=1e-6)
        self.assertLess(choked["sonic_relative_error"], 1e-4)


    def test_HEM_allows_two_phase_critical_throat_without_sound_speed(self):
        class TwoPhasePathFluid(IdealGasTestFluid):
            def x_v_from_Ph(self, P, h):
                # Mimic a backend identifying the throat state as a two-phase
                # equilibrium mixture. The ideal-gas h/rho relations remain only
                # a software test fixture, not a physical mixture model.
                return 0.5 if P < 7.0e5 else None

        throat = critical_mass_flux_HEM(
            1.0e6, 300.0, 1.0e4, TwoPhasePathFluid()
        )
        self.assertTrue(throat["choked"])
        self.assertEqual(throat["model"], "HEM")
        self.assertEqual(throat["phase"], "two_phase")
        self.assertAlmostEqual(throat["x_v"], 0.5)
        self.assertAlmostEqual(throat["x_l"], 0.5)
        self.assertIsNone(throat["a"])
        self.assertIsNone(throat["sonic_relative_error"])


    def test_mdot_is_not_an_input(self):
        with self.assertRaises(TypeError):
            solve_cycle(
                self.fluid,
                Tc=1.0,
                Pu=1.0e6,
                Th=300.0,
                Tpre=[200.0, 100.0, 20.0],
                eta=[0.8, 0.8, 0.8, 0.8],
                mdot=1.0e-5,
            )

    def test_two_phase_downstream_of_single_phase_throat_is_allowed(self):
        class DownstreamTwoPhaseFluid(IdealGasTestFluid):
            def x_v_from_Ph(self, P, h):
                # Phase transition occurs below the ideal-gas critical pressure;
                # the single-phase sonic throat is upstream of this transition.
                return 0.5 if P < 4.0e5 else None

        throat = critical_mass_flux(1.0e6, 300.0, 1.0e4, DownstreamTwoPhaseFluid())
        self.assertTrue(throat["choked"])
        self.assertGreater(throat["P_star"], 4.0e5)
        self.assertLess(throat["sonic_relative_error"], 1e-4)

    def test_baseline_uses_cd_equal_to_one(self):
        f = self.fluid
        d = 1.0e-4
        result = solve_cycle(
            f,
            Tc=1.0,
            Pu=1.0e6,
            Th=300.0,
            Tpre=[200.0, 100.0, 20.0],
            eta=[0.8, 0.8, 0.8, 0.8],
            d=d,
        )
        expected_mdot = (math.pi / 4.0) * d**2 * result.throat["G"]
        self.assertAlmostEqual(result.mdot, expected_mdot, places=16)
        with self.assertRaises(TypeError):
            solve_cycle(
                f,
                Tc=1.0,
                Pu=1.0e6,
                Th=300.0,
                Tpre=[200.0, 100.0, 20.0],
                eta=[0.8, 0.8, 0.8, 0.8],
                d=d,
                Cd=0.8,
            )

    def test_specific_duties_without_orifice_diameter(self):
        result = solve_cycle(
            self.fluid,
            Tc=1.0,
            Pu=1.0e6,
            Th=300.0,
            Tpre=[200.0, 100.0, 20.0],
            eta=[0.8, 0.8, 0.8, 0.8],
        )
        self.assertIsNone(result.mdot)
        self.assertIsNone(result.Q_C)
        self.assertIsNotNone(result.q_C)

    def test_invalid_eta_rejected(self):
        f = self.fluid
        A = State(P=1.0e6, T=300.0, h=f.h(1.0e6, 300.0))
        C = State(P=1.0e4, T=10.0, h=f.h(1.0e4, 10.0))
        with self.assertRaises(ValueError):
            chex_AC(A, C, C.P, 1.1, f)


if __name__ == "__main__":
    unittest.main()
