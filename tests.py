# tests.py - unit tests for graph, travel time, and pathfinder modules
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph_builder import buildGraph, haversineDistance
from travel_time import calcTravelTime
from pathfinder import PathFinder


# tiny 5-node graph so tests don't need Excel or trained models
MOCK_COORDS = {
    '970':  (-37.800, 145.010),
    '3685': (-37.805, 145.015),
    '2000': (-37.810, 145.020),
    '2846': (-37.795, 145.005),
    '4043': (-37.800, 145.025),
}

MOCK_CONNECTIONS = {
    '970':  ['3685', '2846'],
    '3685': ['970',  '2000'],
    '2000': ['3685', '4043'],
    '2846': ['4043', '970'],
    '4043': ['2846', '2000'],
}


class MockPredictor:
    def predict(self, *args, **kwargs):
        return 100


def makeMockGraph():
    return buildGraph(MOCK_CONNECTIONS, MOCK_COORDS)


def makePathFinder():
    graph = makeMockGraph()
    pf = PathFinder(graph, MockPredictor(), MOCK_COORDS)
    return pf


# test 1: buildGraph returns a non-empty dict
class TestBuildGraph(unittest.TestCase):
    def test_buildGraph_returnsNonEmpty(self):
        graph = makeMockGraph()
        self.assertIsInstance(graph, dict)
        self.assertGreater(len(graph), 0)

    # test 2: haversineDistance is positive for two known coords
    def test_haversineDistance_positive(self):
        dist = haversineDistance(-37.800, 145.010, -37.810, 145.020)
        self.assertGreater(dist, 0)

    # test 3: calcTravelTime returns positive value for normal inputs
    def test_calcTravelTime_positive(self):
        t = calcTravelTime(1.5, 100)
        self.assertGreater(t, 0)

    # test 4: calcTravelTime increases with higher flow (congestion slows travel)
    def test_calcTravelTime_higherFlowSlower(self):
        # congested road should take longer than free-flow
        tFreeFlow = calcTravelTime(1.0, 10)
        tCongested = calcTravelTime(1.0, 500)
        self.assertGreater(tCongested, tFreeFlow)



class TestPathFinderBasic(unittest.TestCase):
    def setUp(self):
        self.pf = makePathFinder()

    # test 5: A* finds a path between two connected nodes
    def test_astar_findsPath(self):
        self.pf.setAlgorithm('astar')
        path, cost, _ = self.pf.findPath('970', '2000')
        self.assertIsNotNone(path)
        self.assertIn(970, path)
        self.assertIn(2000, path)

    # test 6: BFS finds a path between two connected nodes
    def test_bfs_findsPath(self):
        path, cost, _ = self.pf.bfs('970', '2000')
        self.assertIsNotNone(path)
        self.assertIn(2000, path)

    # test 7: DFS finds a path between two connected nodes
    def test_dfs_findsPath(self):
        path, cost, _ = self.pf.dfs('970', '2000')
        self.assertIsNotNone(path)
        self.assertIn(2000, path)

    # test 8: Dijkstra finds a path between two connected nodes
    def test_dijkstra_findsPath(self):
        path, cost, _ = self.pf.dijkstra('970', '2000')
        self.assertIsNotNone(path)
        self.assertIn(2000, path)

    # test 9: same origin and destination returns None or cost 0
    def test_sameOriginDest_returnNoneOrZero(self):
        path, cost, _ = self.pf.astar('970', '970')
        # either no path or trivial zero-length path with cost 0
        self.assertTrue(path is None or cost == 0)



class TestPathFinderEdgeCases(unittest.TestCase):
    def setUp(self):
        self.pf = makePathFinder()

    # test 10: unknown node returns None path
    def test_invalidNode_returnsNone(self):
        path, cost, _ = self.pf.astar('970', '9999')
        self.assertIsNone(path)

    # test 11: path cost is > 0 for a valid route
    def test_pathCost_greaterThanZero(self):
        path, cost, _ = self.pf.astar('970', '2000')
        self.assertIsNotNone(path)
        self.assertGreater(cost, 0)

    # test 12: path is a list of ints
    def test_path_isListOfInts(self):
        path, _, _ = self.pf.astar('970', '4043')
        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)
        for node in path:
            self.assertIsInstance(node, int)



class TestFindUniquePaths(unittest.TestCase):
    def setUp(self):
        self.pf = makePathFinder()

    # test 13: findUniquePaths returns at most 5 routes
    def test_findUniquePaths_atMostFive(self):
        routes = self.pf.findUniquePaths('970', '2000', maxPaths=5)
        self.assertLessEqual(len(routes), 5)

    # test 14: routes are sorted ascending by cost
    def test_findUniquePaths_sortedByCost(self):
        routes = self.pf.findUniquePaths('970', '2000', maxPaths=5)
        costs = [c for _, c, _ in routes]
        self.assertEqual(costs, sorted(costs))

    # test 15: no duplicate paths returned
    def test_findUniquePaths_noDuplicates(self):
        routes = self.pf.findUniquePaths('970', '2000', maxPaths=5)
        pathTuples = [tuple(p) for p, _, _ in routes]
        self.assertEqual(len(pathTuples), len(set(pathTuples)))


# Test that returned paths are structurally valid
class TestRouteValidity(unittest.TestCase):
    def setUp(self):
        self.pf = makePathFinder()
        self.graph = makeMockGraph()

    # Test path starts at origin and ends at destination
    def test_path_starts_at_origin_ends_at_dest(self):
        path, _, _ = self.pf.astar('970', '2000')
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 970)
        self.assertEqual(path[-1], 2000)

    # Test path contains no repeated nodes (no cycles)
    def test_path_has_no_cycles(self):
        path, _, _ = self.pf.astar('970', '4043')
        self.assertIsNotNone(path)
        self.assertEqual(len(path), len(set(path)))

    # Test every consecutive node pair in the path is actually connected in the graph
    def test_path_edges_exist_in_graph(self):
        path, _, _ = self.pf.astar('970', '2000')
        self.assertIsNotNone(path)
        for i in range(len(path) - 1):
            fromNode = str(path[i])
            toNode = str(path[i + 1])
            neighbours = [n for n, _ in self.graph.get(fromNode, [])]
            self.assertIn(toNode, neighbours)


if __name__ == '__main__':
    unittest.main(verbosity=2)
