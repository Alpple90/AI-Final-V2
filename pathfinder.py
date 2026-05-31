# pathfinder.py - 6 search algorithms for finding routes

from heapq import heappush, heappop
import math
from config import DEFAULT_K_ROUTES
from travel_time import calcTravelTime


class PathFinder:
    def __init__(self, graph, traffic_predictor, coords=None):
        self.graph = graph
        self.traffic_predictor = traffic_predictor
        self.coords = coords or {}
        self.currentModel = 'lstm'
        self.currentAlgorithm = 'astar'

        self.algorithms = {
            'bfs': self.bfs,
            'dfs': self.dfs,
            'greedy': self.greedy,
            'astar': self.astar,
            'dijkstra': self.dijkstra,
            'bidirectional': self.bidirectionalAstar
        }

    # swap out the ML model used for traffic prediction
    def setModel(self, modelName):
        self.currentModel = modelName

    # switch to a different search algorithm, returns False if unknown
    def setAlgorithm(self, algoName):
        if algoName in self.algorithms:
            self.currentAlgorithm = algoName
            return True
        return False

    def getEdgeCost(self, fromNode, toNode, distance, hour, dayOfWeek=2):
        predictedFlow = self.traffic_predictor.predict(self.currentModel, toNode, hour, dayOfWeek)
        return calcTravelTime(distance, predictedFlow)

    # sum up travel time across every edge in the path
    def calcPathTime(self, path, hour, dayOfWeek=2):
        if len(path) < 2:
            return 0
        totalTime = 0
        for i in range(len(path) - 1):
            fromNode = str(path[i])
            toNode = str(path[i+1])
            for neighbor, dist in self.graph.get(fromNode, []):
                if neighbor == toNode:
                    totalTime += self.getEdgeCost(fromNode, toNode, dist, hour, dayOfWeek)
                    break
        return round(totalTime, 2)

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
        return R * c

    # run BFS from start to goal, return path + cost + nodes explored
    def bfs(self, start, goal, hour=12, dayOfWeek=2):
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
                totalTime = self.calcPathTime([int(n) for n in path], hour, dayOfWeek)
                return [int(n) for n in path], totalTime, nodesExplored
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None, float('inf'), nodesExplored

    # run DFS from start to goal with a depth cap to avoid runaway paths
    def dfs(self, start, goal, hour=12, maxDepth=50, dayOfWeek=2):
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
                totalTime = self.calcPathTime([int(n) for n in path], hour, dayOfWeek)
                return [int(n) for n in path], totalTime, nodesExplored
            if current in visited or depth > maxDepth:
                continue
            visited.add(current)
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor], depth + 1))
        return None, float('inf'), nodesExplored

    # greedy best-first search, picks the node that looks closest to the goal
    def greedy(self, start, goal, hour=12, dayOfWeek=2):
        startStr, goalStr = str(start), str(goal)
        if startStr not in self.graph or goalStr not in self.graph:
            return None, float('inf'), 0

        pq = [(self.heuristic(startStr, goalStr), startStr, [startStr])]
        visited = set()
        nodesExplored = 0

        while pq:
            _, current, path = heappop(pq)
            nodesExplored += 1
            if current in visited:
                continue
            visited.add(current)
            if current == goalStr:
                totalTime = self.calcPathTime([int(n) for n in path], hour, dayOfWeek)
                return [int(n) for n in path], totalTime, nodesExplored
            for neighbor, _ in self.graph.get(current, []):
                if neighbor not in visited:
                    h = self.heuristic(neighbor, goalStr)
                    heappush(pq, (h, neighbor, path + [neighbor]))
        return None, float('inf'), nodesExplored

    # A* search combining actual cost with haversine heuristic
    def astar(self, start, goal, hour=12, dayOfWeek=2):
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
                edgeCost = self.getEdgeCost(current, neighbor, distance, hour, dayOfWeek)
                newCost = actualCost + edgeCost
                h = self.heuristic(neighbor, goalStr)
                counter += 1
                heappush(pq, (newCost + h, counter, neighbor, path + [neighbor], newCost))
        return None, float('inf'), nodesExplored

    # dijkstra's shortest path expanding by lowest cost so far
    def dijkstra(self, start, goal, hour=12, dayOfWeek=2):
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
                edgeCost = self.getEdgeCost(current, neighbor, distance, hour, dayOfWeek)
                newCost = cost + edgeCost
                heappush(pq, (newCost, neighbor, path + [neighbor]))
        return None, float('inf'), nodesExplored

    # A* searching from both ends simultaneously and merging when they meet
    def bidirectionalAstar(self, start, goal, hour=12, dayOfWeek=2):
        startStr, goalStr = str(start), str(goal)
        if startStr not in self.graph or goalStr not in self.graph:
            return None, float('inf'), 0

        fwdPq = [(self.heuristic(startStr, goalStr), 0, startStr, [startStr], 0)]
        fwdVisited = {}
        bwdPq = [(self.heuristic(goalStr, startStr), 0, goalStr, [goalStr], 0)]
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
                    edgeCost = self.getEdgeCost(current, neighbor, distance, hour, dayOfWeek)
                    newCost = cost + edgeCost
                    h = self.heuristic(neighbor, goalStr)
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
                    edgeCost = self.getEdgeCost(current, neighbor, distance, hour, dayOfWeek)
                    newCost = cost + edgeCost
                    h = self.heuristic(neighbor, startStr)
                    counter += 1
                    heappush(bwdPq, (newCost + h, counter, neighbor, path + [neighbor], newCost))

            if bestPath and fwdPq and bwdPq:
                if fwdPq[0][0] + bwdPq[0][0] >= bestCost:
                    break

        if bestPath:
            return bestPath, round(bestCost, 2), nodesExplored
        return None, float('inf'), nodesExplored

    # dispatch to whichever algorithm is currently selected
    def findPath(self, start, goal, hour=12, dayOfWeek=2):
        algoFunc = self.algorithms.get(self.currentAlgorithm, self.astar)
        return algoFunc(start, goal, hour, dayOfWeek)

    # find the k cheapest paths using Yen's spur approach
    def findTopKPaths(self, start, goal, k=DEFAULT_K_ROUTES, hour=12, dayOfWeek=2):
        allPaths = []
        startStr = str(start)
        goalStr = str(goal)

        if startStr not in self.graph or goalStr not in self.graph:
            return []

        firstPath, firstCost, nodesExp = self.findPath(start, goal, hour, dayOfWeek)
        if not firstPath:
            return []

        allPaths.append((firstPath, firstCost))
        print(f"  Path 1: {firstCost:.2f} min")

        potentialPaths = []  # min-heap of (cost, path)

        for kIdx in range(1, k):
            lastPath = allPaths[-1][0]
            candidatesAdded = 0

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

                spurPath, spurCost, _ = self.findPath(spurNode, goal, hour, dayOfWeek)

                self.graph = originalGraph

                if spurPath:
                    totalPath = rootPath[:-1] + spurPath
                    totalCost = self.calcPathTime(totalPath, hour, dayOfWeek)
                    if totalPath not in [p for p, _ in allPaths]:
                        heappush(potentialPaths, (totalCost, totalPath))
                        candidatesAdded += 1

            if not potentialPaths:
                break

            bestCost, bestPath = heappop(potentialPaths)
            allPaths.append((bestPath, bestCost))
            print(f"  Path {kIdx+1}: {bestCost:.2f} min")

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

        return result

    # run every algorithm and pool their routes, then deduplicate and return the best ones
    def findUniquePaths(self, start, goal, hour=12, maxPaths=5, dayOfWeek=2):
        # each algorithm finds up to maxPaths routes via Yen's spur method,
        # then pool everything, deduplicate and return the top maxPaths
        seen = {}
        results = []

        print(f"---Finding routes {start} -> {goal}---")
        for algoName in self.algorithms:
            print(f"\n--- {algoName.upper()} ---")
            self.setAlgorithm(algoName)
            kPaths = self.findTopKPaths(start, goal, k=maxPaths, hour=hour, dayOfWeek=dayOfWeek)

            for path, cost in kPaths:
                key = tuple(path)
                if key in seen:
                    idx = seen[key]
                    existingPath, existingCost, existingAlgos = results[idx]
                    if algoName not in existingAlgos:
                        results[idx] = (existingPath, existingCost, existingAlgos + [algoName])
                else:
                    seen[key] = len(results)
                    results.append((path, cost, [algoName]))

        results.sort(key=lambda x: x[1])
        print(f"\n---{len(results[:maxPaths])} unique route(s) found---")
        return results[:maxPaths]
