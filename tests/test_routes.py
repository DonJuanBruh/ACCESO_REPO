import math
import unittest

import numpy as np

import config
import data_generator
import routes


class RoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.territory = data_generator.generate_territory()
        cls.network = routes.prepare_network(cls.territory)
        villages = cls.territory["settlements"][cls.territory["settlements"]["type"] == "vereda"]
        cls.a, cls.b = villages.iloc[0], villages.iloc[-1]

    def walking_route(self):
        graphs = routes.build_graphs(self.network, config.TRAVEL_MODES[0])
        cost, next_node, _ = routes.cost_to(graphs, "time", [self.b["cell"]])
        path = routes.path_to(next_node, cost, self.a["cell"])
        return routes.measure_path(self.network, graphs, path, self.territory["relief"]["altitude"])

    def test_tobler_speed_peaks_on_gentle_downhill(self):
        self.assertAlmostEqual(float(routes.tobler_speed(np.array(-0.05))), config.TOBLER_VMAX_KMH)
        self.assertLess(float(routes.tobler_speed(np.array(0.3))), config.TOBLER_VMAX_KMH)

    def test_terrain_route_is_longer_than_straight_line(self):
        straight = math.hypot(self.a["x_km"] - self.b["x_km"], self.a["y_km"] - self.b["y_km"])
        measure = self.walking_route()
        self.assertGreater(measure["km"], straight)
        self.assertGreater(measure["ascent_m"], 0)

    def test_vehicle_is_faster_than_walking(self):
        walking = self.walking_route()["minutes"]
        graphs = routes.build_graphs(self.network, config.TRAVEL_MODES[1])
        cost, _, _ = routes.cost_to(graphs, "time", [self.b["cell"]])
        self.assertLess(cost[self.a["cell"]], walking)

    def test_closing_roads_never_shortens_trips(self):
        graphs = routes.build_graphs(self.network, config.TRAVEL_MODES[1])
        open_cost, _, _ = routes.cost_to(graphs, "time", [self.b["cell"]])
        with_bridge = self.territory["roads"].index[self.territory["roads"]["bridges"] > 0].tolist()
        closed = routes.build_graphs(self.network, config.TRAVEL_MODES[1], with_bridge)
        closed_cost, _, _ = routes.cost_to(closed, "time", [self.b["cell"]])
        self.assertGreaterEqual(closed_cost[self.a["cell"]], open_cost[self.a["cell"]])


if __name__ == "__main__":
    unittest.main()
