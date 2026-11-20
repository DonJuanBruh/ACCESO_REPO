import unittest

import config
import data_generator
import terrain_ai


class TerrainAITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.territory = data_generator.generate_territory()
        cls.info = terrain_ai.train_model(cls.territory)

    def test_t8_spatial_cross_validation_auc(self):
        self.assertGreaterEqual(self.info["auc"], config.TARGETS["T8_auc"])

    def test_t8_recall_on_unsuitable_terrain(self):
        self.assertGreaterEqual(self.info["recall_unsuitable"], config.TARGETS["T8_recall"])

    def test_classification_returns_three_classes(self):
        suitability, probabilities = terrain_ai.classify_cells(self.info["model"], self.territory)
        self.assertEqual(set(suitability.ravel().tolist()), {0, 1, 2})
        self.assertEqual(probabilities.shape[-1], 3)

    def test_feature_importances_sum_to_one(self):
        self.assertAlmostEqual(float(self.info["importances"]["importance"].sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
