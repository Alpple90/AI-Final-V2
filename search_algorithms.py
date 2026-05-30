# search_algorithms.py (ADD Bidirectional A*)
"""
Search Algorithms from Assignment 2A
Includes: BFS, DFS, Greedy, A*, Bidirectional A*, Dijkstra (UCS)
"""

from heapq import heappush, heappop
import math


class SearchAlgorithms:
    def __init__(self, graph, travel_time_calculator, traffic_predictor):
        self.graph = graph
        self.tt_calc = travel_time_calculator
        self.traffic_predictor = traffic_predictor
        self.current_model = 'lstm'
        
    def set_model(self, model_name):
        self.current_model = model_name
    
    def _get_edge_cost(self, from_node, to_node, distance, hour):
        predicted_flow = self.traffic_predictor.predict(self.current_model, None, hour)
        return self.tt_calc.calculate_travel_time(distance, predicted_flow)
    
    def _heuristic(self, node, goal, coords=None):
        """Euclidean distance heuristic"""
        if coords and node in coords and goal in coords:
            lat1, lon1 = coords[node]
            lat2, lon2 = coords[goal]
            R = 6371
            lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
            lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
            dlat = lat2_r - lat1_r
            dlon = lon2_r - lon1_r
            a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return (R * c / 60) * 60  # minutes at free-flow speed
        return 0
    
    # ============================================================
    # 1. BFS (Breadth-First Search)
    # ============================================================
    def bfs(self, start, goal, hour=12, coords=None):
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
    
    # ============================================================
    # 2. DFS (Depth-First Search)
    # ============================================================
    def dfs(self, start, goal, hour=12, coords=None, max_depth=50):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        stack = [(start_str, [start_str], 0)]
        visited = set()
        nodes_explored = 0
        best_path, best_cost = None, float('inf')
        
        while stack:
            current, path, depth = stack.pop()
            nodes_explored += 1
            if depth > max_depth:
                continue
            if current == goal_str:
                total_time = self._calculate_path_time([int(n) for n in path], hour)
                if total_time < best_cost:
                    best_cost, best_path = total_time, [int(n) for n in path]
                continue
            if current in visited:
                continue
            visited.add(current)
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor], depth + 1))
        return best_path, best_cost, nodes_explored
    
    # ============================================================
    # 3. Greedy Best-First Search
    # ============================================================
    def greedy(self, start, goal, hour=12, coords=None):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        pq = [(self._heuristic(start_str, goal_str, coords), start_str, [start_str])]
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
                    h = self._heuristic(neighbor, goal_str, coords)
                    heappush(pq, (h, neighbor, path + [neighbor]))
        return None, float('inf'), nodes_explored
    
    # ============================================================
    # 4. A* Search
    # ============================================================
    def a_star(self, start, goal, hour=12, coords=None):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        pq = [(0, start_str, [start_str], 0)]
        visited = {}
        nodes_explored = 0
        counter = 0
        
        while pq:
            est_total, current, path, actual_cost = heappop(pq)
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
                h = self._heuristic(neighbor, goal_str, coords)
                counter += 1
                heappush(pq, (new_cost + h, neighbor, path + [neighbor], new_cost))
        return None, float('inf'), nodes_explored
    
    # ============================================================
    # 5. Dijkstra (Uniform Cost Search)
    # ============================================================
    def dijkstra(self, start, goal, hour=12, coords=None):
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
    
    # ============================================================
    # 6. Bidirectional A* Search (NEW)
    # ============================================================
    def bidirectional_a_star(self, start, goal, hour=12, coords=None):
        """
        Bidirectional A* Search
        Searches from both start and goal simultaneously
        Much faster than standard A* for large graphs
        """
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0
        
        # Forward search from start
        forward_pq = [(self._heuristic(start_str, goal_str, coords), start_str, [start_str], 0)]
        forward_visited = {start_str: (0, [start_str])}  # node -> (cost, path)
        
        # Backward search from goal
        backward_pq = [(self._heuristic(goal_str, start_str, coords), goal_str, [goal_str], 0)]
        backward_visited = {goal_str: (0, [goal_str])}
        
        nodes_explored = 0
        best_path = None
        best_cost = float('inf')
        counter = 0
        
        while forward_pq and backward_pq:
            # Expand forward
            if forward_pq:
                _, current, path, cost = heappop(forward_pq)
                nodes_explored += 1
                
                if current in backward_visited:
                    # Meet in the middle
                    back_cost, back_path = backward_visited[current]
                    total_cost = cost + back_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        # Combine paths (remove duplicate meeting node)
                        combined = path[:-1] + back_path[::-1]
                        best_path = [int(n) for n in combined]
                        # Don't stop - keep searching for potentially better
                
                if current in forward_visited and forward_visited[current][0] <= cost:
                    continue
                forward_visited[current] = (cost, path)
                
                for neighbor, distance in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    edge_cost = self._get_edge_cost(current, neighbor, distance, hour)
                    new_cost = cost + edge_cost
                    h = self._heuristic(neighbor, goal_str, coords)
                    counter += 1
                    heappush(forward_pq, (new_cost + h, neighbor, path + [neighbor], new_cost))
            
            # Expand backward
            if backward_pq:
                _, current, path, cost = heappop(backward_pq)
                nodes_explored += 1
                
                if current in forward_visited:
                    forward_cost, forward_path = forward_visited[current]
                    total_cost = forward_cost + cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        combined = forward_path[:-1] + path[::-1]
                        best_path = [int(n) for n in combined]
                
                if current in backward_visited and backward_visited[current][0] <= cost:
                    continue
                backward_visited[current] = (cost, path)
                
                for neighbor, distance in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    edge_cost = self._get_edge_cost(current, neighbor, distance, hour)
                    new_cost = cost + edge_cost
                    h = self._heuristic(neighbor, start_str, coords)
                    counter += 1
                    heappush(backward_pq, (new_cost + h, neighbor, path + [neighbor], new_cost))
            
            # Early termination if best path found and can't be beaten
            if best_path and forward_pq and backward_pq:
                forward_best = forward_pq[0][0]
                backward_best = backward_pq[0][0]
                if forward_best + backward_best >= best_cost:
                    break
        
        if best_path:
            total_time = self._calculate_path_time(best_path, hour)
            return best_path, total_time, nodes_explored
        return None, float('inf'), nodes_explored
    
    # ============================================================
    # Helper Methods
    # ============================================================
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
    
    def get_algorithm_info(self):
        return {
            'bfs': {'name': 'BFS', 'optimal': False, 'complete': True, 'time': 'O(V+E)', 'space': 'O(V)'},
            'dfs': {'name': 'DFS', 'optimal': False, 'complete': False, 'time': 'O(V+E)', 'space': 'O(depth)'},
            'greedy': {'name': 'Greedy Best-First', 'optimal': False, 'complete': False, 'time': 'O(V log V)', 'space': 'O(V)'},
            'astar': {'name': 'A*', 'optimal': True, 'complete': True, 'time': 'O(E log V)', 'space': 'O(V)'},
            'dijkstra': {'name': 'Dijkstra', 'optimal': True, 'complete': True, 'time': 'O(E log V)', 'space': 'O(V)'},
            'bidirectional': {'name': 'Bidirectional A*', 'optimal': True, 'complete': True, 'time': 'O(b^(d/2))', 'space': 'O(b^(d/2))'}
        }
    
    def recommend_algorithm(self, graph_size, avg_degree, time_pressure='medium'):
        """Automatically recommend best algorithm based on scenario"""
        if graph_size > 500:
            return 'bidirectional', 'Bidirectional A* is fastest for large graphs'
        elif time_pressure == 'high':
            return 'greedy', 'Greedy is fastest for time-critical applications'
        elif time_pressure == 'low' and graph_size < 200:
            return 'astar', 'A* provides optimal paths with good performance'
        elif avg_degree > 8:
            return 'dijkstra', 'Dijkstra handles high connectivity well'
        else:
            return 'astar', 'A* is the best all-purpose algorithm'