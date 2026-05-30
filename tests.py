# tests.py - unit tests for graph, travel time, and pathfinder modules
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph_builder import build_graph, haversine_distance
from travel_time import calc_travel_time
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
    # always return a steady 100 vehicles per 15 min
    def predict(self, model_name, last_seq, hour_of_day=12, day_of_week=2):
        return 100


def make_mock_graph():
    return build_graph(MOCK_CONNECTIONS, MOCK_COORDS)


def make_path_finder():
    graph = make_mock_graph()
    pf = PathFinder(graph, MockPredictor(), MOCK_COORDS)
    return pf


# Test graph building and basic utilities
class TestBuildGraph(unittest.TestCase):
    def test_build_graph_returns_non_empty(self):
        graph = make_mock_graph()
        self.assertIsInstance(graph, dict)
        self.assertGreater(len(graph), 0)

    # Test that haversine distance is positive for two known coords
    def test_haversine_distance_positive(self):
        dist = haversine_distance(-37.800, 145.010, -37.810, 145.020)
        self.assertGreater(dist, 0)

    # Test that calc_travel_time returns positive value for normal inputs
    def test_calc_travel_time_positive(self):
        t = calc_travel_time(1.5, 100)
        self.assertGreater(t, 0)

    # Test that higher flow means slower travel (congestion)
    def test_calc_travel_time_higher_flow_slower(self):
        # congested road should take longer than free-flow
        t_free_flow = calc_travel_time(1.0, 10)
        t_congested = calc_travel_time(1.0, 500)
        self.assertGreater(t_congested, t_free_flow)



class TestPathFinderBasic(unittest.TestCase):
    def setUp(self):
        self.pf = make_path_finder()

    # Test that A* finds a path between two connected nodes
    def test_astar_finds_path(self):
        self.pf.set_algorithm('astar')
        path, cost, _ = self.pf.find_path('970', '2000')
        self.assertIsNotNone(path)
        self.assertIn(970, path)
        self.assertIn(2000, path)

    # Test that BFS finds a path between two connected nodes
    def test_bfs_finds_path(self):
        path, cost, _ = self.pf.bfs('970', '2000')
        self.assertIsNotNone(path)
        self.assertIn(2000, path)

    # Test that DFS finds a path between two connected nodes
    def test_dfs_finds_path(self):
        path, cost, _ = self.pf.dfs('970', '2000')
        self.assertIsNotNone(path)
        self.assertIn(2000, path)

    # Test that Dijkstra finds a path between two connected nodes
    def test_dijkstra_finds_path(self):
        path, cost, _ = self.pf.dijkstra('970', '2000')
        self.assertIsNotNone(path)
        self.assertIn(2000, path)

    # Test same origin and destination returns None or cost 0
    def test_same_origin_dest_returns_none_or_zero(self):
        path, cost, _ = self.pf.astar('970', '970')
        # either no path or trivial zero-length path with cost 0
        self.assertTrue(path is None or cost == 0)



class TestPathFinderEdgeCases(unittest.TestCase):
    def setUp(self):
        self.pf = make_path_finder()

    # Test that unknown node returns None path
    def test_invalid_node_returns_none(self):
        path, cost, _ = self.pf.astar('970', '9999')
        self.assertIsNone(path)

    # Test that path cost is > 0 for a valid route
    def test_path_cost_greater_than_zero(self):
        path, cost, _ = self.pf.astar('970', '2000')
        self.assertIsNotNone(path)
        self.assertGreater(cost, 0)

    # Test that path is a list of ints
    def test_path_is_list_of_ints(self):
        path, _, _ = self.pf.astar('970', '4043')
        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)
        for node in path:
            self.assertIsInstance(node, int)



class TestFindUniquePaths(unittest.TestCase):
    def setUp(self):
        self.pf = make_path_finder()

    # Test that find_unique_paths returns at most 5 routes
    def test_find_unique_paths_at_most_five(self):
        routes = self.pf.find_unique_paths('970', '2000', max_paths=5)
        self.assertLessEqual(len(routes), 5)

    # Test that routes are sorted ascending by cost
    def test_find_unique_paths_sorted_by_cost(self):
        routes = self.pf.find_unique_paths('970', '2000', max_paths=5)
        costs = [c for _, c, _ in routes]
        self.assertEqual(costs, sorted(costs))

    # Test that no duplicate paths are returned
    def test_find_unique_paths_no_duplicates(self):
        routes = self.pf.find_unique_paths('970', '2000', max_paths=5)
        path_tuples = [tuple(p) for p, _, _ in routes]
        self.assertEqual(len(path_tuples), len(set(path_tuples)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
