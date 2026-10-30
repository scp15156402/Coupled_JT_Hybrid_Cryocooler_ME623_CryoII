"""Tests for jt/workflow.py (naming, saving, feasibility, solve, upstream choice, run storage).
Uses CoolProp He-4 as a fast stand-in; folders are redirected to a temp directory."""
import json, tempfile, unittest
from pathlib import Path
from jt import workflow as wf

BASE = dict(fluid="Helium", T_C=2.5, P_U=0.36, T_H=300, T_pre=[80, 25, 8], eta=[1, 1, 1, 1], d_um=20)


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._old = (wf.RUNS, wf.CASES)
        wf.RUNS, wf.CASES = self.tmp / "runs", self.tmp / "cases"

    def tearDown(self):
        wf.RUNS, wf.CASES = self._old

    def test_naming(self):
        self.assertEqual(wf.config_id(wf.norm_cfg({**BASE, "fluid": "he3", "T_C": 1.0})), "Helium_3_1K_20um_4_stage")
        self.assertEqual(wf.config_id(wf.norm_cfg({**BASE, "eta": [1, 1, 1, 0.97]})), "Helium_4_2.5K_20um_4_stage_eta0.97")
        self.assertTrue(wf.config_id(wf.norm_cfg({**BASE, "d_um": None})).endswith("nod_4_stage"))

    def test_save_same_content_reuses_file_different_content_versions(self):
        a = wf.save_config(wf.norm_cfg(BASE))
        b = wf.save_config(wf.norm_cfg(BASE))
        c = wf.save_config(wf.norm_cfg({**BASE, "P_U": 0.5}))
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(c["id"], a["id"] + "_v2")
        self.assertTrue((wf.CASES / (a["id"] + ".json")).is_file())

    def test_blank_t_pre_allowed_and_wrong_length_rejected(self):
        cfg = wf.norm_cfg({**BASE, "T_pre": [None, None, 8]})
        self.assertEqual(cfg["T_pre"], [None, None, 8.0])
        with self.assertRaises(ValueError):
            wf.norm_cfg({**BASE, "T_pre": [80, 8]})
        with self.assertRaises(ValueError):
            wf.norm_cfg({**BASE, "eta": [1, 1, 1, 1.2]})

    def test_feasibility_messages(self):
        ok = wf.check_config(wf.norm_cfg(BASE))
        self.assertIn(ok["verdict"], ("feasible", "warnings"), ok)
        bad = lambda **k: wf.check_config(wf.norm_cfg({**BASE, **k}))
        self.assertEqual(bad(T_pre=[80, 90, 8])["verdict"], "infeasible")          # not decreasing
        self.assertEqual(bad(P_U=0.0001)["verdict"], "infeasible")                 # below P_D
        self.assertEqual(bad(T_C=5.18)["verdict"], "infeasible")                    # above He-4 critical T
        self.assertEqual(bad(P_U=5.0)["verdict"], "infeasible")                    # q_L <= 0

    def test_solve_states_loads_and_run_storage(self):
        cfg = wf.norm_cfg(BASE)
        run = wf.new_run(cfg, "solve")
        res = wf.run_solve(cfg, run=run)
        wf.finish_run(run, cfg, "solve", res, res["headline"])
        self.assertEqual(len(res["states"]), 14)
        self.assertEqual(len(res["loads"]), 3)
        self.assertTrue((run / "ph_cycle.png").is_file())
        runs = wf.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertFalse(runs[0]["config_saved"])                                  # not saved yet
        wf.save_config(cfg)
        self.assertTrue(wf.list_runs()[0]["config_saved"])                         # now linked by hash
        wf.delete_run(runs[0]["run"])
        self.assertEqual(wf.list_runs(), [])

    def test_upstream_choice_keeps_last_stage_and_orders_temperatures(self):
        cfg = wf.norm_cfg({**BASE, "T_pre": [None, None, 8]})
        fl = wf.get_fluid("Helium")
        up = wf.optimise_upstream(fl, cfg, 0.36, 8.0, log=lambda s: None)
        T = up["T_pre"]
        self.assertTrue(up["feasible"])
        self.assertTrue(300 > T[0] > T[1] > T[2] == 8.0, T)
        a = wf._solve_point(fl, {**cfg, "P_U": 0.36}, T_pre=T)
        b = wf._solve_point(fl, {**cfg, "P_U": 0.36}, T_pre=[80, 25, 8])
        self.assertAlmostEqual(a.q_C, b.q_C, delta=1e-6 * abs(b.q_C))              # upstream cannot change q_C
        self.assertAlmostEqual(a.mdot, b.mdot, delta=1e-9 * b.mdot)

    def test_reference_is_kept_with_run_and_does_not_change_the_config_hash(self):
        ref = {"citation": "Zhang & Dang 2021", "link": "10.1016/x", "location": "Fig. 5", "kind": "Own design", "junk": ""}
        a = wf.norm_cfg(BASE)
        b = wf.norm_cfg({**BASE, "reference": ref})
        self.assertEqual(wf.config_hash(a), wf.config_hash(b))                    # adding a reference keeps run links valid
        self.assertEqual(b["reference"], {k: v for k, v in ref.items() if v})     # empty entries dropped
        self.assertEqual(b["source"], "Zhang & Dang 2021")                         # also readable by the command line
        run = wf.new_run(b, "solve")
        wf.finish_run(run, b, "solve", {"headline": "x"}, "x")
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["reference"]["location"], "Fig. 5")
        self.assertEqual(wf.list_runs()[0]["reference"]["citation"], "Zhang & Dang 2021")


if __name__ == "__main__":
    unittest.main()


class MeasuredFlowTests(unittest.TestCase):
    def test_mdot_input_and_naming(self):
        from jt import workflow as wf
        from jt.cycle_core import make_fluid, solve_cycle
        raw = dict(fluid="Helium", T_C=4.5, P_U=2.0, T_pre=[100.0, 15.0], eta=[0.97] * 3, mdot_mg_s=10.5)
        cfg = wf.norm_cfg(raw)
        self.assertEqual(wf.config_id(cfg), "Helium_4_4.5K_10.5mgs_3_stage_eta0.97")
        f = make_fluid("Helium")
        r = solve_cycle(f, Tc=4.5, Pu=2e6, Th=300.0, Tpre=[100.0, 15.0], eta=[0.97] * 3, mdot_given=10.5e-6)
        self.assertAlmostEqual(r.Q_C, r.mdot * r.q_C, places=12)
        self.assertAlmostEqual(r.mdot, 10.5e-6, places=12)
        self.assertIsNone(r.throat)

    def test_d_and_mdot_together_rejected(self):
        from jt import workflow as wf
        with self.assertRaises(ValueError):
            wf.norm_cfg(dict(fluid="Helium", T_C=4.5, P_U=2.0, T_pre=[100.0, 15.0], eta=[1.0] * 3,
                             d_um=20, mdot_mg_s=10))
