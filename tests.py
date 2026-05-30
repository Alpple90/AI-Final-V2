# tests.py - unit tests for graph, travel time, and pathfinder modules
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph_builder import build_graph, haversine_distance
from travel_time import calc_travel_time
from pathfinder import PathFinder
# import NODE_CONNECTIONS directly to avoid tkintermapview import in map.py
NODE_CONNECTIONS = {
    '970':  ['3685', '2846'],
    '2000': ['3685', '3682', '3812', '4043'],
    '2200': ['3126', '4063'],
    '2820': ['3662', '4321', '2825'],
    '2825': ['2820', '4030', '2827'],
    '2827': ['2825', '4051'],
    '2846': ['970'],
    '3001': ['4262', '3002', '3662', '4821'],
    '3002': ['4263', '3662', '3001'],
    '3120': ['4040', '3122', '4035'],
    '3122': ['3804', '3127', '3120'],
    '3126': ['3682', '2200', '3127'],
    '3127': ['3126', '4063', '3122'],
    '3180': ['4057', '4051'],
    '3662': ['3001', '3002', '4324', '4335', '2820'],
    '3682': ['2000', '3126', '3804'],
    '3685': ['970',  '2000'],
    '3804': ['3812', '3682', '3122', '4040'],
    '3812': ['2000', '3804', '4040'],
    '4030': ['4321', '4032', '4051', '2825'],
    '4032': ['4034', '4057', '4030', '4321'],
    '4034': ['4035', '4063', '4032', '4324'],
    '4035': ['3120', '4034'],
    '4040': ['4043', '3812', '3804', '3120', '4264', '4272'],
    '4043': ['2000', '4040', '4273'],
    '4051': ['4030', '3180', '2827'],
    '4057': ['4063', '3180', '4032'],
    '4063': ['3127', '2200', '4057', '4034'],
    '4262': ['4263', '3001'],
    '4263': ['4264', '3002', '4262'],
    '4264': ['4270', '4040', '4324', '4263'],
    '4270': ['4272', '4264', '4812'],
    '4272': ['4273', '4040', '4270'],
    '4273': ['4043', '4272'],
    '4321': ['4335', '4032', '4030', '2820'],
    '4324': ['4264', '4034', '3662'],
    '4335': ['3662', '4321'],
    '4812': ['4270'],
    '4821': ['3001'],
}


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


# Test that returned paths are structurally valid
class TestRouteValidity(unittest.TestCase):
    def setUp(self):
        self.pf = make_path_finder()
        self.graph = make_mock_graph()

    # Test that path starts at origin and ends at destination
    def test_path_starts_at_origin_ends_at_dest(self):
        path, _, _ = self.pf.astar('970', '2000')
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 970)
        self.assertEqual(path[-1], 2000)

    # Test that path contains no repeated nodes (no cycles)
    def test_path_has_no_cycles(self):
        path, _, _ = self.pf.astar('970', '4043')
        self.assertIsNotNone(path)
        self.assertEqual(len(path), len(set(path)))

    # Test that every consecutive pair in the path is actually connected in the graph
    def test_path_edges_exist_in_graph(self):
        path, _, _ = self.pf.astar('970', '2000')
        self.assertIsNotNone(path)
        for i in range(len(path) - 1):
            from_node = str(path[i])
            to_node = str(path[i + 1])
            neighbours = [n for n, _ in self.graph.get(from_node, [])]
            self.assertIn(to_node, neighbours)

    # Test same check holds for BFS path
    def test_bfs_path_edges_exist_in_graph(self):
        path, _, _ = self.pf.bfs('970', '4043')
        self.assertIsNotNone(path)
        for i in range(len(path) - 1):
            from_node = str(path[i])
            to_node = str(path[i + 1])
            neighbours = [n for n, _ in self.graph.get(from_node, [])]
            self.assertIn(to_node, neighbours)


# Test edge cases for pathfinder and graph
class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.pf = make_path_finder()

    # Test that a disconnected node (no outgoing edges) returns None
    def test_disconnected_node_returns_none(self):
        # build a graph where '999' exists but has no neighbours
        isolated_coords = dict(MOCK_COORDS)
        isolated_coords['999'] = (-37.820, 145.030)
        isolated_connections = dict(MOCK_CONNECTIONS)
        isolated_connections['999'] = []
        graph = build_graph(isolated_connections, isolated_coords)
        pf = PathFinder(graph, MockPredictor(), isolated_coords)
        path, _, _ = pf.astar('999', '2000')
        self.assertIsNone(path)

    # Test that find_unique_paths with max_paths=1 returns exactly 1 route
    def test_find_unique_paths_max_one(self):
        routes = self.pf.find_unique_paths('970', '2000', max_paths=1)
        self.assertEqual(len(routes), 1)

    # Test that an empty graph returns None for any search
    def test_empty_graph_returns_none(self):
        pf = PathFinder({}, MockPredictor(), {})
        path, _, _ = pf.astar('970', '2000')
        self.assertIsNone(path)


# Test structural properties of the real graph built from NODE_CONNECTIONS
class TestGraphStructure(unittest.TestCase):

    def setUp(self):
        # use real coords from the xlsx — fall back to mock if file not present
        try:
            from map import load_sites
            sites_df = load_sites()
            self.coords = {
                row['SCATS Number']: (row['LAT'], row['LNG'])
                for _, row in sites_df.iterrows()
            }
            self.graph = build_graph(NODE_CONNECTIONS, self.coords)
            self.real_data = True
        except Exception:
            self.real_data = False

    # Test that the real graph is not empty
    def test_real_graph_not_empty(self):
        if not self.real_data:
            self.skipTest('scatsTrueLongLat.xlsx not available')
        self.assertGreater(len(self.graph), 0)

    # Test that every edge distance is within the 0.1–10 km filter in build_graph
    def test_all_edge_distances_in_valid_range(self):
        if not self.real_data:
            self.skipTest('scatsTrueLongLat.xlsx not available')
        for node, neighbours in self.graph.items():
            for neighbour, dist in neighbours:
                self.assertGreaterEqual(dist, 0.1)
                self.assertLessEqual(dist, 10.0)

    # Test that every node in the real graph has at least one neighbour
    def test_all_nodes_have_neighbours(self):
        if not self.real_data:
            self.skipTest('scatsTrueLongLat.xlsx not available')
        # only check nodes that appear in NODE_CONNECTIONS (not isolated coord-only nodes)
        for node in NODE_CONNECTIONS:
            if node in self.graph:
                self.assertGreater(len(self.graph[node]), 0,
                                   msg=f'Node {node} has no neighbours in graph')


if __name__ == '__main__':
    unittest.main(verbosity=2)
