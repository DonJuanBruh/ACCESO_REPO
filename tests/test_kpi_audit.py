import unittest

import config
import kpi_audit
import scenarios


class CoreKPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, cls.summary = kpi_audit.test_scalability("quick")

    def test_t1_placement_time(self):
        self.assertLessEqual(self.summary["T1_p95_ms"], config.TARGETS["T1"])

    def test_t2_near_linear_scaling_and_memory(self):
        self.assertLess(self.summary["T2_slope"], 1.3)
        self.assertLess(self.summary["T2_mb"], config.TARGETS["T2_mb"])

    def test_t3_road_closure_recompute(self):
        self.assertLessEqual(self.summary["T3_p95_ms"], config.TARGETS["T3"])

    def test_t4_approximation_factor(self):
        result = kpi_audit.test_approximation("quick")
        self.assertLessEqual(result["max"], config.TARGETS["T4_max"])
        self.assertLessEqual(result["mean"], config.TARGETS["T4_mean"])
        self.assertEqual(result["within_bound"], 100.0)

    def test_n1_table_iii_reduction(self):
        _, result = kpi_audit.test_table_iii("quick")
        self.assertGreaterEqual(result["N1"], config.TARGETS["N1"])


class SystemKPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = scenarios.prepare_base()
        cls.scenario = scenarios.compute_scenario(cls.base, scenarios.parameters())

    def test_t11_core_works_offline(self):
        _, pct = kpi_audit.test_offline(self.base, self.scenario["params"])
        self.assertEqual(pct, config.TARGETS["T11"])

    def test_t9_t10_guided_assistant(self):
        _, result = kpi_audit.test_assistant(self.base, self.scenario, 10, force_offline=True)
        self.assertGreaterEqual(result["T9"], config.TARGETS["T9"])
        self.assertTrue(result["T9_no_fabrication"])
        self.assertLessEqual(result["T10"], config.TARGETS["T10"])

    def test_sus_score_range(self):
        self.assertEqual(kpi_audit.sus_score([5, 1] * 5), 100.0)
        self.assertEqual(kpi_audit.sus_score([1, 5] * 5), 0.0)


if __name__ == "__main__":
    unittest.main()
