# pathfinder.py - 6 search algorithms for finding routes

from heapq import heappush, heappop
import math
from config import DEFAULT_K_ROUTES
from travel_time import calculate_travel_time


class PathFinder:
    def __init__(self, graph, traffic_predictor, coords=None):
        self.graph = graph
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

    def _get_edge_cost(self, from_node, to_node, distance, hour):
        # grab predicted flow for this edge, then convert to travel time
        predictedFlow = self.traffic_predictor.predict(self.current_model, None, hour)
        return calculate_travel_time(distance, predictedFlow)

    def _calculate_path_time(self, path, hour):
        if len(path) < 2:
            return 0
        totalTime = 0
        for i in range(len(path) - 1):
            fromNode = str(path[i])
            toNode = str(path[i+1])
            for neighbor, dist in self.graph.get(fromNode, []):
                if neighbor == toNode:
                    totalTime += self._get_edge_cost(fromNode, toNode, dist, hour)
                    break
        return round(totalTime, 2)

    def _heuristic(self, node, goal):
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
        distKm = R * c
        return (distKm / 60) * 60

    def _bfs(self, start, goal, hour=12):
        from collections import deque
        startStr, goalStr = str(start), str(goal)
        if startStr not in self.graph or goalStr not in self.graph:
            return None, float('inf'), 0

        queue = deque([(startStr, [startStr])])
        visited = {startStr}
        nodesExplored = 0

        while queue:
            current, path = queue.popleft()
            nodesExplored += 1
            if current == goalStr:
                totalTime = self._calculate_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], totalTime, nodesExplored
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None, float('inf'), nodesExplored

    def _dfs(self, start, goal, hour=12, max_depth=50):
        startStr, goalStr = str(start), str(goal)
        if startStr not in self.graph or goalStr not in self.graph:
            return None, float('inf'), 0

        stack = [(startStr, [startStr], 0)]
        visited = set()
        nodesExplored = 0

        while stack:
            current, path, depth = stack.pop()
            nodesExplored += 1
            if current == goalStr:
                totalTime = self._calculate_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], totalTime, nodesExplored
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor], depth + 1))
        return None, float('inf'), nodesExplored

    def _greedy(self, start, goal, hour=12):
        startStr, goalStr = str(start), str(goal)
        if startStr not in self.graph or goalStr not in self.graph:
            return None, float('inf'), 0

        pq = [(self._heuristic(startStr, goalStr), startStr, [startStr])]
        visited = set()
        nodesExplored = 0

        while pq:
            _, current, path = heappop(pq)
            nodesExplored += 1
            if current in visited:
                continue
            visited.add(current)
            if current == goalStr:
                totalTime = self._calculate_path_time([int(n) for n in path], hour)
                return [int(n) for n in path], totalTime, nodesExplored
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    h = self._heuristic(neighbor, goalStr)
                    heappush(pq, (h, neighbor, path + [neighbor]))
        return None, float('inf'), nodesExplored

    def _astar(self, start, goal, hour=12):
        startStr, goalStr = str(start), str(goal)
        if startStr not in self.graph or goalStr not in self.graph:
            return None, float('inf'), 0

        pq = [(0, 0, startStr, [startStr], 0)]
        visited = {}
        nodesExplored = 0
        counter = 0

        while pq:
            estTotal, _, current, path, actualCost = heappop(pq)
            nodesExplored += 1
            if current in visited and visited[current] <= actualCost:
                continue
            visited[current] = actualCost
            if current == goalStr:
                return [int(n) for n in path], round(actualCost, 2), nodesExplored
            for neighbor, distance in self.graph.get(current, []):
                if neighbor in path:
                    continue
                edgeCost = self._get_edge_cost(current, neighbor, distance, hour)
                newCost = actualCost + edgeCost
                h = self._heuristic(neighbor, goalStr)
                counter += 1
                heappush(pq, (newCost + h, counter, neighbor, path + [neighbor], newCost))
        return None, float('inf'), nodesExplored

    def _dijkstra(self, start, goal, hour=12):
        startStr, goalStr = str(start), str(goal)
        if startStr not in self.graph or goalStr not in self.graph:
            return None, float('inf'), 0

        pq = [(0, startStr, [startStr])]
        visited = {}
        nodesExplored = 0

        while pq:
            cost, current, path = heappop(pq)
            nodesExplored += 1
            if current in visited and visited[current] <= cost:
                continue
            visited[current] = cost
            if current == goalStr:
                return [int(n) for n in path], round(cost, 2), nodesExplored
            for neighbor, distance in self.graph.get(current, []):
                if neighbor in path:
                    continue
                edgeCost = self._get_edge_cost(current, neighbor, distance, hour)
                newCost = cost + edgeCost
                heappush(pq, (newCost, neighbor, path + [neighbor]))
        return None, float('inf'), nodesExplored

    def _bidirectional_astar(self, start, goal, hour=12):
        startStr, goalStr = str(start), str(goal)
        if startStr not in self.graph or goalStr not in self.graph:
            return None, float('inf'), 0

        fwdPq = [(self._heuristic(startStr, goalStr), 0, startStr, [startStr], 0)]
        fwdVisited = {}
        bwdPq = [(self._heuristic(goalStr, startStr), 0, goalStr, [goalStr], 0)]
        bwdVisited = {}

        nodesExplored = 0
        bestPath = None
        bestCost = float('inf')
        counter = 0

        while fwdPq and bwdPq:
            # forward step
            _, _, current, path, cost = heappop(fwdPq)
            nodesExplored += 1
            if current in fwdVisited and fwdVisited[current][0] <= cost:
                pass
            else:
                fwdVisited[current] = (cost, path)
                if current in bwdVisited:
                    backCost, backPath = bwdVisited[current]
                    totalCost = cost + backCost
                    if totalCost < bestCost:
                        bestCost = totalCost
                        bestPath = [int(n) for n in (path[:-1] + backPath[::-1])]
                for neighbor, distance in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    edgeCost = self._get_edge_cost(current, neighbor, distance, hour)
                    newCost = cost + edgeCost
                    h = self._heuristic(neighbor, goalStr)
                    counter += 1
                    heappush(fwdPq, (newCost + h, counter, neighbor, path + [neighbor], newCost))

            # backward step
            _, _, current, path, cost = heappop(bwdPq)
            nodesExplored += 1
            if current in bwdVisited and bwdVisited[current][0] <= cost:
                pass
            else:
                bwdVisited[current] = (cost, path)
                if current in fwdVisited:
                    fwdCost, fwdPath = fwdVisited[current]
                    totalCost = fwdCost + cost
                    if totalCost < bestCost:
                        bestCost = totalCost
                        bestPath = [int(n) for n in (fwdPath[:-1] + path[::-1])]
                for neighbor, distance in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    edgeCost = self._get_edge_cost(current, neighbor, distance, hour)
                    newCost = cost + edgeCost
                    h = self._heuristic(neighbor, startStr)
                    counter += 1
                    heappush(bwdPq, (newCost + h, counter, neighbor, path + [neighbor], newCost))

            if bestPath and fwdPq and bwdPq:
                if fwdPq[0][0] + bwdPq[0][0] >= bestCost:
                    break

        if bestPath:
            return bestPath, round(bestCost, 2), nodesExplored
        return None, float('inf'), nodesExplored

    def find_path(self, start, goal, hour=12):
        algoFunc = self.algorithms.get(self.current_algorithm, self._astar)
        return algoFunc(start, goal, hour)

    # Yen's-style k-shortest paths
    def find_top_k_paths(self, start, goal, k=DEFAULT_K_ROUTES, hour=12):
        allPaths = []
        startStr = str(start)
        goalStr = str(goal)

        if startStr not in self.graph or goalStr not in self.graph:
            return []

        firstPath, firstCost, _ = self.find_path(start, goal, hour)
        if not firstPath:
            return []

        allPaths.append((firstPath, firstCost))
        print(f"Path 1 found: {firstPath} (cost={firstCost})")

        potentialPaths = []  # min-heap of (cost, path)

        for kIdx in range(1, k):
            lastPath = allPaths[-1][0]

            for i in range(len(lastPath) - 1):
                spurNode = lastPath[i]
                rootPath = lastPath[:i+1]

                # copy graph so we can remove edges without touching the original
                modGraph = {}
                for node, neighbors in self.graph.items():
                    modGraph[node] = neighbors.copy()

                # remove edges from previous paths that share the same root
                for path in allPaths:
                    if len(path[0]) > i and path[0][:i+1] == rootPath:
                        if i+1 < len(path[0]):
                            fromNode = str(path[0][i])
                            toNode = str(path[0][i+1])
                            modGraph[fromNode] = [(n, d) for n, d in modGraph[fromNode] if n != toNode]

                originalGraph = self.graph
                self.graph = modGraph

                spurPath, spurCost, _ = self.find_path(spurNode, goal, hour)

                self.graph = originalGraph

                if spurPath:
                    totalPath = rootPath[:-1] + spurPath
                    totalCost = self._calculate_path_time(totalPath, hour)
                    if totalPath not in [p for p, _ in allPaths]:
                        heappush(potentialPaths, (totalCost, totalPath))

            if not potentialPaths:
                break

            bestCost, bestPath = heappop(potentialPaths)
            allPaths.append((bestPath, bestCost))
            print(f"Path {kIdx+1} found: {bestPath} (cost={bestCost})")

        allPaths.sort(key=lambda x: x[1])

        result = []
        seenPaths = set()
        for path, cost in allPaths:
            pathTuple = tuple(path)
            if pathTuple not in seenPaths:
                seenPaths.add(pathTuple)
                result.append((path, cost))
            if len(result) >= k:
                break

        print(f"Total distinct routes found: {len(result)}")
        return result

    def find_unique_paths_all_algorithms(self, start, goal, hour=12, max_paths=5):
        # run every algorithm once, deduplicate, sort by time
        seen = {}  # path_tuple -> index in results
        results = []

        for algoName, algoFunc in self.algorithms.items():
            path, cost, _ = algoFunc(start, goal, hour)
            if path:
                key = tuple(path)
                if key in seen:
                    idx = seen[key]
                    existingPath, existingCost, existingAlgos = results[idx]
                    results[idx] = (existingPath, existingCost, existingAlgos + [algoName])
                else:
                    seen[key] = len(results)
                    results.append((path, cost, [algoName]))

        results.sort(key=lambda x: x[1])
        return results[:max_paths]
