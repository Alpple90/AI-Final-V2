# graph_builder.py - build adjacency list from SCATS connections + coords
import math


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def build_graph(connections, coords):
    # start with every node having an empty neighbour list
    graph = {node: [] for node in set(connections) | {n for nbrs in connections.values() for n in nbrs}}

    for node, neighbors in connections.items():
        if node not in coords:
            continue
        for neighbor in neighbors:
            if neighbor not in coords:
                continue
            distance = haversine_distance(*coords[node], *coords[neighbor])
            # skip edges that are implausibly short or long
            if 0.1 <= distance <= 10.0:
                graph[node].append((neighbor, round(distance, 3)))

    return graph


def get_graph_info(graph):
    return {
        'total_nodes': len(graph),
        'total_edges': sum(len(v) for v in graph.values()) // 2,
        'isolated_nodes': sum(1 for v in graph.values() if not v),
    }
