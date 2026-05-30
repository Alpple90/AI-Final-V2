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
        draw_edges(self.map_widget, self.coords)
        
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
    
    def draw_route(self, path):
        """
        Draw a route on the map with orange line
        
        Args:
            path: List of SCATS site numbers in order
        """
        if not self.map_widget or len(path) < 2:
            return
        
        self.clear_route()
        
        # Draw route lines between consecutive nodes
        for i in range(len(path) - 1):
            node1 = str(path[i])
            node2 = str(path[i+1])
            
            if node1 in self.coords and node2 in self.coords:
                lat1, lng1 = self.coords[node1]
                lat2, lng2 = self.coords[node2]
                
                route_line = self.map_widget.set_path(
                    [(lat1, lng1), (lat2, lng2)],
                    color='#ff6f00',  # Orange
                    width=5
                )
                self.current_route_items.append(route_line)
        
        # Highlight nodes on the route
        for i, node in enumerate(path):
            node_str = str(node)
            if node_str not in self.coords:
                continue
                
            # Delete old marker
            if node_str in self.markers:
                try:
                    self.markers[node_str].delete()
                except:
                    pass
            
            # Choose color based on position
            if i == 0:  # Origin
                colour, text = '#2e7d32', f"🚗 {node_str}"
            elif i == len(path) - 1:  # Destination
                colour, text = '#c62828', f"🏁 {node_str}"
            else:  # Waypoint
                colour, text = '#ff6f00', node_str
            
            lat, lng = self.coords[node_str]
            new_marker = self.map_widget.set_marker(
                lat, lng, 
                text=text,
                marker_color_circle=colour,
                marker_color_outside='#ffffff',
                font=('Arial', 11, 'bold')
            )
            self.current_route_items.append(new_marker)
            self.markers[node_str] = new_marker
    
    def clear_route(self):
        """Clear the currently displayed route"""
        for item in self.current_route_items:
            try:
                item.delete()
            except:
                pass
        self.current_route_items = []
        
        # Redraw original markers
        if self.is_initialized:
            for sid, (lat, lng) in self.coords.items():
                if sid in self.markers:
                    try:
                        self.markers[sid].delete()
                    except:
                        pass
                
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