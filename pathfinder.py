# pathfinder.py
"""
Pathfinding with 6 Search Algorithms - Supports Top-K Routes
"""

from heapq import heappush, heappop
import math
from config import DEFAULT_K_ROUTES, MAX_VISITS_PER_NODE


class PathFinder:
    def __init__(self, graph, travel_time_calculator, traffic_predictor, coords=None):
        self.graph = graph
        self.tt_calc = travel_time_calculator
        self.traffic_predictor = traffic_predictor
        self.coords = coords or {}
        self.current_model = 'lstm'
        self.current_algorithm = 'astar'
        
        self.algorithms = {
            'bfs': self._bfs,
            'dfs': self._dfs,
            'greedy': self._greedy,
            'astar': self._astar,
            'dijkstra': self._dijkstra,
            'bidirectional': self._bidirectional_astar
        }

    def set_model(self, model_name):
        self.current_model = model_name

    def set_algorithm(self, algorithm_name):
        if algorithm_name in self.algorithms:
            self.current_algorithm = algorithm_name
            return True
        return False
    
    def set_coordinates(self, coords):
        self.coords = coords

    def _get_edge_cost(self, from_node, to_node, distance, hour):
        predicted_flow = self.traffic_predictor.predict(self.current_model, None, hour)
        return self.tt_calc.calculate_travel_time(distance, predicted_flow)
    
    def _calculate_path_time(self, path, hour):
        if len(path) < 2:
            return 0
        total_time = 0
        for i in range(len(path) - 1):
            from_node = str(path[i])
            to_node = str(path[i+1])
            for neighbor, dist in self.graph.get(from_node, []):
                if neighbor == to_node:
                    total_time += self._get_edge_cost(from_node, to_node, dist, hour)
                    break
        return round(total_time, 2)
    
    def _heuristic(self, node, goal):
        if not self.coords or node not in self.coords or goal not in self.coords:
            return 0
        lat1, lon1 = self.coords[node]
        lat2, lon2 = self.coords[goal]
        R = 6371
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance_km = R * c
        return (distance_km / 60) * 60

    # ============================================================
    # ALGORITHM IMPLEMENTATIONS (BFS, DFS, Greedy, A*, Dijkstra, Bidirectional)
    # ============================================================
    
    def _bfs(self, start, goal, hour=12):
        from collections import deque
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        queue = deque([(start_str, [start_str])])
        visited = {start_str}
        nodes_explored = 0
        
        while queue:
            current, path = queue.popleft()
            nodes_explored += 1
            if current == goal_str:
                total_time = self._calculate_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], total_time, nodes_explored
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None, float('inf'), nodes_explored

    def _dfs(self, start, goal, hour=12, max_depth=50):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0

        stack = [(start_str, [start_str], 0)]
        visited = set()
        nodes_explored = 0

        while stack:
            current, path, depth = stack.pop()
            nodes_explored += 1
            if current == goal_str:
                total_time = self._calculate_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], total_time, nodes_explored
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            for neighbor, _ in reversed(self.graph.get(current, [])):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor], depth + 1))
        return None, float('inf'), nodes_explored

    def _greedy(self, start, goal, hour=12):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        pq = [(self._heuristic(start_str, goal_str), start_str, [start_str])]
        visited = set()
        nodes_explored = 0
        
        while pq:
            _, current, path = heappop(pq)
            nodes_explored += 1
            if current in visited:
                continue
            visited.add(current)
            if current == goal_str:
                total_time = self._calculate_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], total_time, nodes_explored
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    h = self._heuristic(neighbor, goal_str)
                    heappush(pq, (h, neighbor, path + [neighbor]))
        return None, float('inf'), nodes_explored

    def _astar(self, start, goal, hour=12):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        pq = [(0, 0, start_str, [start_str], 0)]
        visited = {}
        nodes_explored = 0
        counter = 0
        
        while pq:
            est_total, _, current, path, actual_cost = heappop(pq)
            nodes_explored += 1
            if current in visited and visited[current] <= actual_cost:
                continue
            visited[current] = actual_cost
            if current == goal_str:
                return [int(n) for n in path], round(actual_cost, 2), nodes_explored
            for neighbor, distance in self.graph.get(current, []):
                if neighbor in path:
                    continue
                edge_cost = self._get_edge_cost(current, neighbor, distance, hour)
                new_cost = actual_cost + edge_cost
                h = self._heuristic(neighbor, goal_str)
                counter += 1
                heappush(pq, (new_cost + h, counter, neighbor, path + [neighbor], new_cost))
        return None, float('inf'), nodes_explored

    def _dijkstra(self, start, goal, hour=12):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        pq = [(0, start_str, [start_str])]
        visited = {}
        nodes_explored = 0
        
        while pq:
            cost, current, path = heappop(pq)
            nodes_explored += 1
            if current in visited and visited[current] <= cost:
                continue
            visited[current] = cost
            if current == goal_str:
                return [int(n) for n in path], round(cost, 2), nodes_explored
            for neighbor, distance in self.graph.get(current, []):
                if neighbor in path:
                    continue
                edge_cost = self._get_edge_cost(current, neighbor, distance, hour)
                new_cost = cost + edge_cost
                heappush(pq, (new_cost, neighbor, path + [neighbor]))
        return None, float('inf'), nodes_explored

    def _bidirectional_astar(self, start, goal, hour=12):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        forward_pq = [(self._heuristic(start_str, goal_str), 0, start_str, [start_str], 0)]
        forward_visited = {}
        backward_pq = [(self._heuristic(goal_str, start_str), 0, goal_str, [goal_str], 0)]
        backward_visited = {}

        nodes_explored = 0
        best_path = None
        best_cost = float('inf')
        counter = 0

        while forward_pq and backward_pq:
            # Forward step
            _, _, current, path, cost = heappop(forward_pq)
            nodes_explored += 1
            if current in forward_visited and forward_visited[current][0] <= cost:
                pass
            else:
                forward_visited[current] = (cost, path)
                if current in backward_visited:
                    back_cost, back_path = backward_visited[current]
                    total_cost = cost + back_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = [int(n) for n in (path[:-1] + back_path[::-1])]
                for neighbor, distance in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    edge_cost = self._get_edge_cost(current, neighbor, distance, hour)
                    new_cost = cost + edge_cost
                    h = self._heuristic(neighbor, goal_str)
                    counter += 1
                    heappush(forward_pq, (new_cost + h, counter, neighbor, path + [neighbor], new_cost))

            # Backward step
            _, _, current, path, cost = heappop(backward_pq)
            nodes_explored += 1
            if current in backward_visited and backward_visited[current][0] <= cost:
                pass
            else:
                backward_visited[current] = (cost, path)
                if current in forward_visited:
                    fwd_cost, fwd_path = forward_visited[current]
                    total_cost = fwd_cost + cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = [int(n) for n in (fwd_path[:-1] + path[::-1])]
                for neighbor, distance in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    edge_cost = self._get_edge_cost(current, neighbor, distance, hour)
                    new_cost = cost + edge_cost
                    h = self._heuristic(neighbor, start_str)
                    counter += 1
                    heappush(backward_pq, (new_cost + h, counter, neighbor, path + [neighbor], new_cost))

            if best_path and forward_pq and backward_pq:
                if forward_pq[0][0] + backward_pq[0][0] >= best_cost:
                    break

        if best_path:
            return best_path, round(best_cost, 2), nodes_explored
        return None, float('inf'), nodes_explored

    # ============================================================
    # FIND PATH (Single)
    # ============================================================
    
    def find_path(self, start, goal, hour=12):
        algorithm_func = self.algorithms.get(self.current_algorithm, self._astar)
        return algorithm_func(start, goal, hour)

    # ============================================================
    # FIND TOP-K PATHS (The Important One!)
    # ============================================================
    
    def find_top_k_paths(self, start, goal, k=DEFAULT_K_ROUTES, hour=12):
        """
        Find K different paths using Yen's algorithm style approach.
        Returns up to K distinct routes from start to goal.
        """
        all_paths = []
        start_str = str(start)
        goal_str = str(goal)
        
        if start_str not in self.graph or goal_str not in self.graph:
            return []
        
        # Step 1: Get the shortest path using current algorithm
        first_path, first_cost, _ = self.find_path(start, goal, hour)
        if not first_path:
            return []
        
        all_paths.append((first_path, first_cost))
        print(f"Path 1 found: {first_path} (cost={first_cost})")
        
        # Step 2: Use Yen's algorithm to find more paths
        potential_paths = []  # heap of (cost, path, deviation_node_index)
        
        for k_idx in range(1, k):
            # For each path found so far, create deviations
            last_path = all_paths[-1][0]
            
            for i in range(len(last_path) - 1):
                # Spur node is the node we deviate from
                spur_node = last_path[i]
                root_path = last_path[:i+1]
                
                # Remove edges that would create duplicate paths
                modified_graph = {}
                for node, neighbors in self.graph.items():
                    modified_graph[node] = neighbors.copy()
                
                # Remove edges used in previous paths that share the root path
                for path in all_paths:
                    if len(path[0]) > i and path[0][:i+1] == root_path:
                        # Remove the next edge in this path
                        if i+1 < len(path[0]):
                            from_node = str(path[0][i])
                            to_node = str(path[0][i+1])
                            # Remove this specific edge
                            modified_graph[from_node] = [(n, d) for n, d in modified_graph[from_node] if n != to_node]
                
                # Store original graph, use modified
                original_graph = self.graph
                self.graph = modified_graph
                
                # Find spur path from spur_node to goal
                spur_path, spur_cost, _ = self.find_path(spur_node, goal, hour)
                
                # Restore original graph
                self.graph = original_graph
                
                if spur_path:
                    # Combine root path + spur path (excluding first node of spur path if duplicate)
                    total_path = root_path[:-1] + spur_path
                    total_cost = self._calculate_path_time(total_path, hour)
                    
                    # Check if this path is already in all_paths
                    if total_path not in [p for p, _ in all_paths]:
                        heappush(potential_paths, (total_cost, total_path))
            
            if not potential_paths:
                break
            
            # Get the best potential path
            best_cost, best_path = heappop(potential_paths)
            all_paths.append((best_path, best_cost))
            print(f"Path {k_idx+1} found: {best_path} (cost={best_cost})")
        
        # Sort by cost
        all_paths.sort(key=lambda x: x[1])
        
        # Ensure we have exactly k paths (or as many as possible)
        result = []
        seen_paths = set()
        for path, cost in all_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                result.append((path, cost))
            if len(result) >= k:
                break
        
        print(f"Total distinct routes found: {len(result)}")
        return result

    def find_unique_paths_all_algorithms(self, start, goal, hour=12, max_paths=5):
        """Run all 6 algorithms once each, return up to max_paths unique paths sorted by travel time.
        Returns list of (path, cost, algo_name) tuples."""
        seen = {}  # path_tuple -> index in results
        results = []

        for algo_name, algo_func in self.algorithms.items():
            path, cost, _ = algo_func(start, goal, hour)
            if path:
                key = tuple(path)
                if key in seen:
                    idx = seen[key]
                    existing_path, existing_cost, existing_algos = results[idx]
                    results[idx] = (existing_path, existing_cost, existing_algos + [algo_name])
                else:
                    seen[key] = len(results)
                    results.append((path, cost, [algo_name]))

        results.sort(key=lambda x: x[1])
        return results[:max_paths]

    def get_graph_info(self):
        total_nodes = len(self.graph)
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'isolated_nodes': sum(1 for n in self.graph.values() if not n)
        }
    
    def get_algorithm_list(self):
        return list(self.algorithms.keys())
    
    def get_algorithm_info(self):
        return {
            'bfs': {'name': 'BFS', 'optimal': False, 'complete': True},
            'dfs': {'name': 'DFS', 'optimal': False, 'complete': False},
            'greedy': {'name': 'Greedy Best-First', 'optimal': False, 'complete': False},
            'astar': {'name': 'A*', 'optimal': True, 'complete': True},
            'dijkstra': {'name': "Dijkstra's", 'optimal': True, 'complete': True},
            'bidirectional': {'name': 'Bidirectional A*', 'optimal': True, 'complete': True}
        }