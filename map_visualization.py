# map_visualization.py
"""
Map visualization wrapper for TBRGS
Uses your existing map.py for SCATS network display
"""

import math
from config import MAP_TILE_SERVER, MAP_ZOOM_LEVEL, MAP_LOCATE_ZOOM

# Import from your existing map.py
from map import (
    NODE_CONNECTIONS,
    NODE_COLOURS,
    load_sites,
    draw_edges
)

# Try to import tkintermapview
try:
    import tkintermapview
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


class SCATSMapViewer:
    """
    Wrapper for tkintermapview with SCATS network overlay
    Uses your map.py for road connections and colors
    """
    
    def __init__(self):
        self.map_widget = None
        self.coords = {}          # SCATS number -> (lat, lon)
        self.markers = {}         # SCATS number -> marker object
        self.current_route_items = []  # Temporary route elements
        self.network_paths = []   # Network edge path objects
        self.is_initialized = False
        
    def load_coordinates(self):
        """Load SCATS coordinates using your map.py function"""
        sites_df = load_sites()
        self.coords = {
            row['SCATS Number']: (row['LAT'], row['LNG']) 
            for _, row in sites_df.iterrows()
        }
        print(f"Loaded {len(self.coords)} SCATS sites with coordinates")
        return self.coords
    
    def get_available_sites(self):
        """Return sorted list of available SCATS site numbers"""
        return sorted([s for s in self.coords.keys() if s.isdigit()], key=int)
    
    def create_map(self, parent_frame):
        """Create the map widget"""
        if not MAP_AVAILABLE:
            return None
        
        self.map_widget = tkintermapview.TkinterMapView(parent_frame, corner_radius=0)
        self.map_widget.pack(fill='both', expand=True)
        self.map_widget.set_tile_server(MAP_TILE_SERVER)
        
        # Center map on Borondara
        if self.coords:
            lats = [c[0] for c in self.coords.values()]
            lngs = [c[1] for c in self.coords.values()]
            self.map_widget.set_position(sum(lats)/len(lats), sum(lngs)/len(lngs))
            self.map_widget.set_zoom(MAP_ZOOM_LEVEL)
        
        return self.map_widget
    
    def draw_network(self):
        """Draw all roads and markers on the map"""
        if not self.map_widget or not self.coords:
            return
        
        # Draw roads using YOUR draw_edges function
        self.network_paths = draw_edges(self.map_widget, self.coords)
        self.network_visible = True
        
        # Draw markers using YOUR colors
        for sid, (lat, lng) in self.coords.items():
            colour = NODE_COLOURS.get(sid, '#1a1a2e')
            marker = self.map_widget.set_marker(
                lat, lng, 
                text=sid,
                marker_color_circle=colour,
                marker_color_outside='#ffffff',
                font=('Arial', 10, 'bold')
            )
            self.markers[sid] = marker
        
        self.is_initialized = True
    
    def locate_site(self, site_str):
        """Center map on a specific SCATS site"""
        if not self.map_widget or site_str not in self.coords:
            return False
        
        lat, lng = self.coords[site_str]
        self.map_widget.set_position(lat, lng)
        self.map_widget.set_zoom(MAP_LOCATE_ZOOM)
        return True
    
    def draw_route(self, path, color='#ff6f00', is_best=False):
        """Draw a single route on the map in the given color."""
        if not self.map_widget or len(path) < 2:
            return

        for i in range(len(path) - 1):
            node1, node2 = str(path[i]), str(path[i+1])
            if node1 in self.coords and node2 in self.coords:
                lat1, lng1 = self.coords[node1]
                lat2, lng2 = self.coords[node2]
                line = self.map_widget.set_path(
                    [(lat1, lng1), (lat2, lng2)],
                    color=color,
                    width=5 if is_best else 3
                )
                self.current_route_items.append(line)

        # Update existing origin/destination markers to highlight them
        if is_best:
            for node_str, marker_colour, label in [
                (str(path[0]),  '#2e7d32', f"{path[0]} [START]"),
                (str(path[-1]), '#c62828', f"{path[-1]} [END]"),
            ]:
                if node_str not in self.coords or node_str not in self.markers:
                    continue
                try:
                    self.markers[node_str].delete()
                except:
                    pass
                lat, lng = self.coords[node_str]
                m = self.map_widget.set_marker(
                    lat, lng,
                    text=label,
                    marker_color_circle=marker_colour,
                    marker_color_outside='#ffffff',
                    font=('Arial', 12, 'bold')
                )
                self.markers[node_str] = m
                self.current_route_items.append((node_str, m))

    def clear_route(self):
        """Clear the currently displayed route"""
        highlighted_nodes = set()
        for item in self.current_route_items:
            if isinstance(item, tuple):
                node_str, marker = item
                highlighted_nodes.add(node_str)
                try:
                    marker.delete()
                except:
                    pass
            else:
                try:
                    item.delete()
                except:
                    pass
        self.current_route_items = []

        # Restore original markers for any highlighted nodes
        if self.is_initialized:
            for sid in highlighted_nodes:
                if sid not in self.coords:
                    continue
                lat, lng = self.coords[sid]
                colour = NODE_COLOURS.get(sid, '#1a1a2e')
                self.markers[sid] = self.map_widget.set_marker(
                    lat, lng,
                    text=sid,
                    marker_color_circle=colour,
                    marker_color_outside='#ffffff',
                    font=('Arial', 10, 'bold')
                )
    
    def is_map_available(self):
        """Check if map functionality is available"""
        return MAP_AVAILABLE
    
    def get_node_connections(self):
        """Return the road connections for graph building"""
        return NODE_CONNECTIONS
    
    def get_node_colours(self):
        """Return node colors for map display"""
        return NODE_COLOURS
    
    def get_coords(self):
        """Return coordinates dictionary"""
        return self.coords