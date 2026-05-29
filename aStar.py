
import heapq

try:
    from numpy import hypot
    def distance(a, b):
        """The distance between two (x, y) points."""
        xA, yA = a
        xB, yB = b
        return hypot((xA - xB), (yA - yB))
except:
    def distance(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

from custom import parsing

from itertools import count

def aStar(graph, start, goal, heuristic):

    # Priority queue
    open_set = []

    # counter to ensure when priotity tie, the node that enters the queue first will be popped
    # when in priority tie, heapq will conpare the next value after the first and will use that to break the tie
    # see https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes for more detail
    counter = count()

    # Set of explored nodes
    explored = set()

    # Dict to record path to goal node
    path_find = {}

    # Dict of g cost of nodes
    g_cost = {start:0}

    # checks is goal is passed as a single goal or a set of goals
    # if single goal, normalise it as a set to work with the rest of the algorithm:
    if not isinstance(goal, (list, tuple, set)):
        goal = [goal]
    else:
        goal = sorted(goal)

    heapq.heappush(open_set, (0, next(counter), 0, start, None))

    while open_set:
        f, _, g, node, parent = heapq.heappop(open_set)
        
        if node in explored:
            continue

        path_find[node] = parent

        if node in goal:
            return node, path_to_goal(path_find, start, node), explored, len(explored) + 1 

        explored.add(node)
        for neighbour, cost in graph.get(node, []):

            if neighbour in explored:
                continue
            temp_g = g + cost
            
            if neighbour not in g_cost:

                g_cost[neighbour] = temp_g
                
                f_cost = temp_g + min(heuristic(neighbour, dest) for dest in goal)

                heapq.heappush(open_set, (f_cost, next(counter), temp_g, neighbour, node))
            elif temp_g < g_cost[neighbour]:
                g_cost[neighbour] = temp_g
                
                f_cost = temp_g + min(heuristic(neighbour, dest) for dest in goal)
                heapq.heappush(open_set, (f_cost, next(counter), temp_g, neighbour, node))
 
    return None, None, None, None
    
def path_to_goal(explored, start, goal):
    path = []

    node = goal

    while node is not None:
        path.append(node)

        node = explored.get(node)

    path.reverse()

    return path

def implement_aStar_graph(locations, graph, start, destination):

    # Heuristic for Node/Edge graph problem
    #h = lambda n: distance(locations[n], locations[destination], destination)
    h = lambda n, g: distance(locations[n], locations[g])

    # Returns goal node, path, explored set, number of nodes created 
    return aStar(graph, start, destination, h)


def test():

    locations, graph, start, destinations = parsing("PathFinder-test.txt")
    
    print(locations)
    print(graph)
    print(start)
    print(destinations)
    print()
    
    goal_node, path, eSet, numNodes = implement_aStar_graph(locations, graph, start, destinations)
    print(goal_node)
    print(path)

#test()