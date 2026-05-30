# graph_builder.py - build adjacency list from SCATS connections + coords
import math


# calculate the great-circle distance in km between two lat/lng points
def haversineDistance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1R, lon1R = math.radians(lat1), math.radians(lon1)
    lat2R, lon2R = math.radians(lat2), math.radians(lon2)
    dLat = lat2R - lat1R
    dLon = lon2R - lon1R
    a = math.sin(dLat/2)**2 + math.cos(lat1R) * math.cos(lat2R) * math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# build weighted adjacency list from connection map and lat/lng coords
def buildGraph(connections, coords):
    # start with every node having an empty neighbour list
    graph = {node: [] for node in set(connections) | {n for nbrs in connections.values() for n in nbrs}}

    for node, neighbors in connections.items():
        if node not in coords:
            continue
        for neighbor in neighbors:
            if neighbor not in coords:
                continue
            distance = haversineDistance(*coords[node], *coords[neighbor])
            # skip edges that are implausibly short or long
            if 0.1 <= distance <= 10.0:
                graph[node].append((neighbor, round(distance, 3)))

    return graph


# return a quick summary of node count, edge count and isolated nodes
def getGraphInfo(graph):
    return {
        'total_nodes': len(graph),
        'total_edges': sum(len(v) for v in graph.values()) // 2,
        'isolated_nodes': sum(1 for v in graph.values() if not v),
    }
