"""Tests for the jt_run front door: case loading, scorecard logic, and a CoolProp solve."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import jt_run

PAPER_REPORT = {
    "paper_comparison_meaningful": True,
    "sections": {
        "fig5": {"P_U_opt_MPa": 0.36, "q_C_opt_kJ_kg": 8.0, "P_U_max_MPa": 1.13,
                 "max_rel_dev_qC_vs_min_deltahT_qL": 0.0, "optimum_at_search_bound": False},
        "fits": {"max_rel_gap_deltahT_vs_qL_at_optimum": 1e-4, "at_search_bound": [],
                 "P_U_opt_vs_Eq36": {"mape_percent": 1.0, "max_rel_percent": 2.0},
                 "q_C_opt_vs_Eq35_36_chain": {"mape_percent": 1.0}},
        "fig7": {"per_P_U": [{"P_U_MPa": 0.5, "difference_K": 0.2}]},
        "flux": {"per_T_C": [{"T_C_K": 1.0, "HEM_vs_paper_curve": {"mape_percent": 5.0},
                              "HEM_refit": {"fit_vs_HEM_points": {"max_rel_percent": 0.1}}}]},
        "fig15": {"q_C_eta1_kJ_kg": 8.0, "Q_C_eta1_mW": 18.7, "mdot_eta1_mg_s": 2.34,
                  "Q_C_eta097_mW": 14.7, "Q_C_drop_097_percent": 21.4,
                  "paper": {"q_C_opt_kJ_kg": 8.0, "Q_C_ideal_mW": 18.7, "mdot_mg_s": 2.34,
                            "Q_C_eta097_mW": 14.7, "Q_C_drop_097_percent": 21.4}},
    },
}


def _case(**over):
    base = {"name": "t", "fluid": "Helium", "T_C": 2.6, "P_U": 0.5, "T_pre": [80.0, 25.0, 8.0],
            "eta": [1.0, 1.0, 1.0, 0.97], "d_um": 20.0}
    base.update(over)
    return base


class TestScorecard(unittest.TestCase):
    def test_paper_values_pass(self):
        rows = jt_run.build_scorecard(PAPER_REPORT)
        self.assertGreater(len(rows), 10)
        self.assertEqual([r["check"] for r in rows if r["status"] in ("FAIL", "N/A")], [])

    def test_wrong_anchor_fails_and_model_miss_only_warns(self):
        bad = json.loads(json.dumps(PAPER_REPORT))
        bad["sections"]["fig5"]["q_C_opt_kJ_kg"] = 9.5
        bad["sections"]["fig15"]["Q_C_eta1_mW"] = 25.0
        rows = {r["check"]: r["status"] for r in jt_run.build_scorecard(bad)}
        self.assertEqual(rows["q_C,opt"], "FAIL")
        self.assertEqual(rows["Q_C at eta = 1"], "WARN")

    def test_failed_section_is_reported(self):
        rep = {"sections": {"fig5": {"error": "boom"}}}
        rows = jt_run.build_scorecard(rep)
        self.assertEqual(rows[0]["status"], "FAIL")


class TestCaseFile(unittest.TestCase):
    def _load(self, data):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            return jt_run.load_case(p)

    def test_valid_case_and_underscore_keys_ignored(self):
        c = self._load(dict(_help="x", sweep={"_h": "x", "P_U": [0.3, 0.4]}, **_case()))
        self.assertEqual(c["sweep"], {"P_U": [0.3, 0.4]})
        self.assertEqual(c["T_H"], 300.0)

    def test_typo_key_rejected(self):
        with self.assertRaises(ValueError):
            self._load(dict(_case(), Tpre=[1, 2, 3]))

    def test_wrong_number_of_precooling_temperatures_rejected(self):
        with self.assertRaises(ValueError):
            self._load(_case(T_pre=[80.0, 25.0]))

    def test_bad_sweep_key_rejected(self):
        with self.assertRaises(ValueError):
            self._load(_case(sweep={"eta": [0.9]}))


class TestSolveWithCoolProp(unittest.TestCase):
    def test_solve_writes_outputs_and_compares(self):
        with tempfile.TemporaryDirectory() as d:
            case = _case(reported={"Q_C_mW": 1.0, "not_a_metric": 5},
                         sweep={"eta_last": [1.0, 0.95]})
            p = Path(d) / "c.json"
            p.write_text(json.dumps(case), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                ok = jt_run.run_case(p, Path(d) / "out", pinch=False)
            self.assertTrue(ok)
            out = Path(d) / "out" / "t"
            for name in ("case.json", "result.json", "states.csv", "sweep.csv"):
                self.assertTrue((out / name).is_file(), name)
            res = json.loads((out / "result.json").read_text(encoding="utf-8"))
            self.assertGreater(res["metrics"]["Q_C_mW"], 0.0)
            cmp = {r["metric"]: r for r in res["comparison"]}
            self.assertIn("difference", cmp["Q_C_mW"])
            self.assertIn("note", cmp["not_a_metric"])


if __name__ == "__main__":
    unittest.main()
