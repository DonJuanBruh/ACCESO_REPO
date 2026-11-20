import unittest

import config
import scenarios


class ScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = scenarios.prepare_base()
        cls.scenario = scenarios.compute_scenario(cls.base, scenarios.parameters())

    def test_every_center_type_gets_its_configured_k(self):
        for kind, rule in config.CENTER_TYPES.items():
            self.assertEqual(len(self.scenario["types"][kind]["centers"]), rule["k"])

    def test_centers_respect_allowed_terrain_classes(self):
        for kind, rule in config.CENTER_TYPES.items():
            allowed = [config.CLASSES[c] for c in rule["allowed_classes"]]
            for center in self.scenario["types"][kind]["centers"]:
                self.assertIn(center["ai_suitability"], allowed)

    def test_n1_access_beats_most_populated_baseline(self):
        for kind in config.CENTER_TYPES:
            ours = self.scenario["types"][kind]["eval"]["kpi"]["max_time_min"]
            baseline = self.scenario["bases"][kind]["eval"]["kpi"]["max_time_min"]
            self.assertLessEqual(ours, baseline)

    def test_n4_no_center_on_unsuitable_terrain(self):
        _, summary = scenarios.compare_strategies(self.base, self.scenario, "Centro de salud", draws=5)
        self.assertEqual(summary["pct_with_ai_unsuitable"], 0)

    def test_road_closure_recomputes_scenario(self):
        closed = self.base["territory"]["roads"].index[self.base["territory"]["roads"]["bridges"] > 0].tolist()[:1]
        params = scenarios.parameters(closed=closed)
        other = scenarios.compute_scenario(self.base, params)
        self.assertNotEqual(other["id"], self.scenario["id"])
        self.assertEqual(other["params"]["closed"], tuple(closed))


if __name__ == "__main__":
    unittest.main()
