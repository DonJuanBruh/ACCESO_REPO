import unittest

import numpy as np

import config
import data_generator


class TerritoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.territory = data_generator.generate_territory()

    def test_grid_matches_configuration(self):
        grid = self.territory["grid"]
        self.assertEqual(grid["ny"] * grid["nx"], self.territory["relief"]["altitude"].size)
        self.assertGreater(grid["ny"] * grid["nx"], 10000)

    def test_relief_is_within_andean_range(self):
        altitude = self.territory["relief"]["altitude"]
        self.assertGreaterEqual(altitude.min(), 0)
        self.assertLessEqual(altitude.max(), 5000)

    def test_settlements_cover_every_municipality(self):
        settlements = self.territory["settlements"]
        villages = settlements[settlements["type"] == "vereda"]
        self.assertEqual(len(villages), config.VILLAGES_PER_MUNICIPALITY * len(config.TOWN_CENTERS))
        self.assertTrue(set(config.TOWN_CENTERS).issubset(set(settlements["municipality"])))
        self.assertTrue((settlements.loc[settlements["type"] != "salida", "population"] > 0).all())

    def test_road_network_has_bridges(self):
        roads = self.territory["roads"]
        self.assertGreater(len(roads), 0)
        self.assertGreater((roads["bridges"] > 0).sum(), 0)
        self.assertTrue((roads["length_km"] > 0).all())

    def test_generation_is_reproducible(self):
        again = data_generator.generate_territory()
        np.testing.assert_array_equal(again["relief"]["altitude"], self.territory["relief"]["altitude"])
        self.assertEqual(len(again["simma"]), len(self.territory["simma"]))


if __name__ == "__main__":
    unittest.main()
