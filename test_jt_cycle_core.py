import math
import unittest

from jt_cycle_core import (
    State, chex_AC, chex_AB, critical_mass_flux, critical_mass_flux_HEM,
    solve_cycle, state_labels,
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

    def test_chex_inverse_round_trip(self):
        f = self.fluid
        A = State(P=1.0e6, T=300.0, h=f.h(1.0e6, 300.0))
        C = State(P=1.0e4, T=10.0, h=f.h(1.0e4, 10.0))
        eta = 0.8
        B, D = chex_AC(A, C, C.P, eta, f)
        C2, D2 = chex_AB(A, B, C.P, eta, f)
        self.assertAlmostEqual(B.T, 68.0, places=8)
        self.assertAlmostEqual(D.T, 242.0, places=8)
        self.assertAlmostEqual(C2.h, C.h, places=8)
        self.assertAlmostEqual(D2.h, D.h, places=8)

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
