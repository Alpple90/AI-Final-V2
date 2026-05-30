# pathfinder.py - 6 search algorithms for finding routes

from heapq import heappush, heappop
import math
from config import DEFAULT_K_ROUTES
from travel_time import calc_travel_time


class PathFinder:
    def __init__(self, graph, traffic_predictor, coords=None):
        self.graph = graph
        self.traffic_predictor = traffic_predictor
        self.coords = coords or {}
        self.current_model = 'lstm'
        self.current_algorithm = 'astar'

        self.algorithms = {
            'bfs': self.bfs,
            'dfs': self.dfs,
            'greedy': self.greedy,
            'astar': self.astar,
            'dijkstra': self.dijkstra,
            'bidirectional': self.bidirectional_astar
        }

    # swap out the ML model used for traffic prediction
    def set_model(self, model_name):
        self.current_model = model_name

    # switch to a different search algorithm, returns False if unknown
    def set_algorithm(self, algo_name):
        if algo_name in self.algorithms:
            self.current_algorithm = algo_name
            return True
        return False

    def get_edge_cost(self, from_node, to_node, distance, hour):
        # grab predicted flow for this edge, then convert to travel time
        predicted_flow = self.traffic_predictor.predict(self.current_model, None, hour)
        return calc_travel_time(distance, predicted_flow)

    # sum up travel time across every edge in the path
    def calc_path_time(self, path, hour):
        if len(path) < 2:
            return 0
        total_time = 0
        for i in range(len(path) - 1):
            from_node = str(path[i])
            to_node = str(path[i+1])
            for neighbor, dist in self.graph.get(from_node, []):
                if neighbor == to_node:
                    total_time += self.get_edge_cost(from_node, to_node, dist, hour)
                    break
        return round(total_time, 2)

    def heuristic(self, node, goal):
        # straight-line distance estimate using haversine
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
        dist_km = R * c
        return (dist_km / 60) * 60

    # run BFS from start to goal, return path + cost + nodes explored
    def bfs(self, start, goal, hour=12):
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
                total_time = self.calc_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], total_time, nodes_explored
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None, float('inf'), nodes_explored

    # run DFS from start to goal with a depth cap to avoid runaway paths
    def dfs(self, start, goal, hour=12, max_depth=50):
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
                total_time = self.calc_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], total_time, nodes_explored
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor], depth + 1))
        return None, float('inf'), nodes_explored

    # greedy best-first search, picks the node that looks closest to the goal
    def greedy(self, start, goal, hour=12):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0

        pq = [(self.heuristic(start_str, goal_str), start_str, [start_str])]
        visited = set()
        nodes_explored = 0

        while pq:
            _, current, path = heappop(pq)
            nodes_explored += 1
            if current in visited:
                continue
            visited.add(current)
            if current == goal_str:
                total_time = self.calc_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], total_time, nodes_explored
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    h = self.heuristic(neighbor, goal_str)
                    heappush(pq, (h, neighbor, path + [neighbor]))
        return None, float('inf'), nodes_explored

    # A* search combining actual cost with haversine heuristic
    def astar(self, start, goal, hour=12):
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
                edge_cost = self.get_edge_cost(current, neighbor, distance, hour)
                new_cost = actual_cost + edge_cost
                h = self.heuristic(neighbor, goal_str)
                counter += 1
                heappush(pq, (new_cost + h, counter, neighbor, path + [neighbor], new_cost))
        return None, float('inf'), nodes_explored

    # dijkstra's shortest path expanding by lowest cost so far
    def dijkstra(self, start, goal, hour=12):
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
                edge_cost = self.get_edge_cost(current, neighbor, distance, hour)
                new_cost = cost + edge_cost
                heappush(pq, (new_cost, neighbor, path + [neighbor]))
        return None, float('inf'), nodes_explored

    # A* searching from both ends simultaneously and merging when they meet
    def bidirectional_astar(self, start, goal, hour=12):
        start_str, goal_str = str(start), str(goal)
        if start_str not in self.graph or goal_str not in self.graph:
            return None, float('inf'), 0

        fwd_pq = [(self.heuristic(start_str, goal_str), 0, start_str, [start_str], 0)]
        fwd_visited = {}
        bwd_pq = [(self.heuristic(goal_str, start_str), 0, goal_str, [goal_str], 0)]
        bwd_visited = {}

        nodes_explored = 0
        best_path = None
        best_cost = float('inf')
        counter = 0

        while fwd_pq and bwd_pq:
            # forward step
            _, _, current, path, cost = heappop(fwd_pq)
            nodes_explored += 1
            if current in fwd_visited and fwd_visited[current][0] <= cost:
                pass
            else:
                fwd_visited[current] = (cost, path)
                if current in bwd_visited:
                    back_cost, back_path = bwd_visited[current]
                    total_cost = cost + back_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = [int(n) for n in (path[:-1] + back_path[::-1])]
                for neighbor, distance in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    edge_cost = self.get_edge_cost(current, neighbor, distance, hour)
                    new_cost = cost + edge_cost
                    h = self.heuristic(neighbor, goal_str)
                    counter += 1
                    heappush(fwd_pq, (new_cost + h, counter, neighbor, path + [neighbor], new_cost))

            # backward step
            _, _, current, path, cost = heappop(bwd_pq)
            nodes_explored += 1
            if current in bwd_visited and bwd_visited[current][0] <= cost:
                pass
            else:
                bwd_visited[current] = (cost, path)
                if current in fwd_visited:
                    fwd_cost, fwd_path = fwd_visited[current]
                    total_cost = fwd_cost + cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = [int(n) for n in (fwd_path[:-1] + path[::-1])]
                for neighbor, distance in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    edge_cost = self.get_edge_cost(current, neighbor, distance, hour)
                    new_cost = cost + edge_cost
                    h = self.heuristic(neighbor, start_str)
                    counter += 1
                    heappush(bwd_pq, (new_cost + h, counter, neighbor, path + [neighbor], new_cost))

            if best_path and fwd_pq and bwd_pq:
                if fwd_pq[0][0] + bwd_pq[0][0] >= best_cost:
                    break

        if best_path:
            return best_path, round(best_cost, 2), nodes_explored
        return None, float('inf'), nodes_explored

    # dispatch to whichever algorithm is currently selected
    def find_path(self, start, goal, hour=12):
        algo_func = self.algorithms.get(self.current_algorithm, self.astar)
        return algo_func(start, goal, hour)

    # find the k cheapest paths using Yen's spur approach
    def find_top_k_paths(self, start, goal, k=DEFAULT_K_ROUTES, hour=12):
        all_paths = []
        start_str = str(start)
        goal_str = str(goal)

        if start_str not in self.graph or goal_str not in self.graph:
            return []

        first_path, first_cost, nodes_exp = self.find_path(start, goal, hour)
        if not first_path:
            return []

        all_paths.append((first_path, first_cost))
        print(f"  Path 1: {first_cost:.2f} min")

        potential_paths = []  # min-heap of (cost, path)

        for k_idx in range(1, k):
            last_path = all_paths[-1][0]
            candidates_added = 0

            for i in range(len(last_path) - 1):
                spur_node = last_path[i]
                root_path = last_path[:i+1]

                # copy graph so we can remove edges without touching the original
                mod_graph = {}
                for node, neighbors in self.graph.items():
                    mod_graph[node] = neighbors.copy()

                # remove edges from previous paths that share the same root
                for path in all_paths:
                    if len(path[0]) > i and path[0][:i+1] == root_path:
                        if i+1 < len(path[0]):
                            from_node = str(path[0][i])
                            to_node = str(path[0][i+1])
                            mod_graph[from_node] = [(n, d) for n, d in mod_graph[from_node] if n != to_node]

                original_graph = self.graph
                self.graph = mod_graph

                spur_path, spur_cost, _ = self.find_path(spur_node, goal, hour)

                self.graph = original_graph

                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    total_cost = self.calc_path_time(total_path, hour)
                    if total_path not in [p for p, _ in all_paths]:
                        heappush(potential_paths, (total_cost, total_path))
                        candidates_added += 1

            if not potential_paths:
                break

            best_cost, best_path = heappop(potential_paths)
            all_paths.append((best_path, best_cost))
            print(f"  Path {k_idx+1}: {best_cost:.2f} min")

        all_paths.sort(key=lambda x: x[1])

        result = []
        seen_paths = set()
        for path, cost in all_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                result.append((path, cost))
            if len(result) >= k:
                break

        return result

    # run every algorithm and pool their routes, then deduplicate and return the best ones
    def find_unique_paths(self, start, goal, hour=12, max_paths=5):
        # each algorithm finds up to max_paths routes via Yen's spur method,
        # then pool everything, deduplicate and return the top max_paths
        seen = {}   # path_tuple -> index in results
        results = []

        print(f"---Finding routes {start} -> {goal}---")
        for algo_name in self.algorithms:
            print(f"\n--- {algo_name.upper()} ---")
            self.set_algorithm(algo_name)
            k_paths = self.find_top_k_paths(start, goal, k=max_paths, hour=hour)

            for path, cost in k_paths:
                key = tuple(path)
                if key in seen:
                    idx = seen[key]
                    existing_path, existing_cost, existing_algos = results[idx]
                    if algo_name not in existing_algos:
                        results[idx] = (existing_path, existing_cost, existing_algos + [algo_name])
                else:
                    seen[key] = len(results)
                    results.append((path, cost, [algo_name]))

        results.sort(key=lambda x: x[1])
        print(f"\n---{len(results[:max_paths])} unique route(s) found---")
        return results[:max_paths]
