import itertools
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay

import k_centers


def random_network(n, seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 30, (n, 2))
    edges = {tuple(sorted((a, b))) for t in Delaunay(points).simplices for a, b in itertools.combinations(t, 2)}
    o, d = np.array(list(edges)).T
    w = np.linalg.norm(points[o] - points[d], axis=1) * rng.uniform(1.15, 1.6, len(o))
    graph = csr_matrix((np.r_[w, w], (np.r_[o, d], np.r_[d, o])), shape=(n, n))
    return graph, rng.integers(50, 500, n)


class KCentersTest(unittest.TestCase):
    def test_returns_k_distinct_centers(self):
        graph, population = random_network(14, 7)
        result = k_centers.place_k_centers(graph, graph, np.arange(14), population, np.ones(14, bool), 3, unit="km")
        nodes = [c["node"] for c in result["centers"]]
        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(set(nodes)), 3)
        self.assertEqual(len(result["steps"]), 3)

    def test_first_center_is_most_populated_suitable_community(self):
        graph, population = random_network(14, 7)
        result = k_centers.place_k_centers(graph, graph, np.arange(14), population, np.ones(14, bool), 3, unit="km")
        self.assertEqual(result["centers"][0]["node"], int(np.argmax(population)))

    def test_respects_two_approximation_bound(self):
        for seed in range(25):
            graph, population = random_network(12, seed)
            result = k_centers.place_k_centers(graph, graph, np.arange(12), population, np.ones(12, bool), 3)
            optimum, _ = k_centers.brute_force_optimum(dijkstra(graph), 3)
            self.assertLessEqual(result["cost"].max(), 2 * optimum + 1e-9)

    def test_never_places_centers_on_forbidden_nodes(self):
        graph, population = random_network(20, 3)
        allowed = np.ones(20, bool)
        allowed[::2] = False
        result = k_centers.place_k_centers(graph, graph, np.arange(20), population, allowed, 4)
        self.assertTrue(all(allowed[c["node"]] for c in result["centers"]))


if __name__ == "__main__":
    unittest.main()
