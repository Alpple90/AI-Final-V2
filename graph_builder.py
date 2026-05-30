# graph_builder.py
"""
Graph builder for SCATS network
Converts road connections into a graph with distances
"""

import math


class GraphBuilder:
    """Builds graph from SCATS connections and coordinates"""
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km using Haversine formula"""
        R = 6371  # Earth's radius in km
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    @staticmethod
    def build_graph(connections, coords):
        """
        Build graph dictionary from connections and coordinates
        
        Args:
            connections: Dictionary of {node: [neighbors]} from map.py
            coords: Dictionary of {node: (lat, lon)}
            
        Returns:
            Graph dictionary: {node: [(neighbor, distance_km)]}
        """
        graph = {}
        
        # Add all nodes
        all_nodes = set(connections.keys())
        for neighbors in connections.values():
            all_nodes.update(neighbors)
        
        for node in all_nodes:
            graph[node] = []
        
        # Add edges with distances
        for node, neighbors in connections.items():
            if node not in coords:
                continue
                
            for neighbor in neighbors:
                if neighbor not in coords:
                    continue
                    
                lat1, lon1 = coords[node]
                lat2, lon2 = coords[neighbor]
                
                distance = GraphBuilder.haversine_distance(lat1, lon1, lat2, lon2)
                
                if 0.1 <= distance <= 10.0:
                    graph[node].append((neighbor, round(distance, 3)))
        
        # Sort neighbors by distance
        for node in graph:
            graph[node].sort(key=lambda x: x[1])
        
        return graph
    
    @staticmethod
    def get_graph_info(graph):
        """Get statistics about the graph"""
        total_nodes = len(graph)
        total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        isolated_nodes = sum(1 for neighbors in graph.values() if not neighbors)
        max_degree = max(len(neighbors) for neighbors in graph.values()) if graph else 0
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'isolated_nodes': isolated_nodes,
            'max_degree': max_degree
        }