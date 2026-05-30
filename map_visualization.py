# map_visualization.py - tkintermapview wrapper for the SCATS network

import math
from config import MAP_TILE_SERVER, MAP_ZOOM_LEVEL, MAP_LOCATE_ZOOM

from map import (
    NODE_CONNECTIONS,
    NODE_COLOURS,
    load_sites,
    draw_edges
)

try:
    import tkintermapview
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


class SCATSMapViewer:

    def __init__(self):
        self.map_widget = None
        self.coords = {}
        self.markers = {}
        self.current_route_items = []
        self.network_paths = []
        self.is_initialized = False

    def load_coordinates(self):
        sitesDF = load_sites()
        self.coords = {
            row['SCATS Number']: (row['LAT'], row['LNG'])
            for _, row in sitesDF.iterrows()
        }
        print(f"Loaded {len(self.coords)} SCATS sites with coordinates")
        return self.coords

    def get_available_sites(self):
        return sorted([s for s in self.coords.keys() if s.isdigit()], key=int)

    def create_map(self, parent_frame):
        if not MAP_AVAILABLE:
            return None

        self.map_widget = tkintermapview.TkinterMapView(parent_frame, corner_radius=0)
        self.map_widget.pack(fill='both', expand=True)
        self.map_widget.set_tile_server(MAP_TILE_SERVER)

        # centre on Boroondara
        if self.coords:
            lats = [c[0] for c in self.coords.values()]
            lngs = [c[1] for c in self.coords.values()]
            self.map_widget.set_position(sum(lats)/len(lats), sum(lngs)/len(lngs))
            self.map_widget.set_zoom(MAP_ZOOM_LEVEL)

        return self.map_widget

    def draw_network(self):
        if not self.map_widget or not self.coords:
            return

        self.network_paths = draw_edges(self.map_widget, self.coords)
        self.network_visible = True

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
        if not self.map_widget or site_str not in self.coords:
            return False
        lat, lng = self.coords[site_str]
        self.map_widget.set_position(lat, lng)
        self.map_widget.set_zoom(MAP_LOCATE_ZOOM)
        return True

    def draw_route(self, path, color='#ff6f00', is_best=False):
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

        # highlight start/end markers
        if is_best:
            for nodeStr, markerColour, label in [
                (str(path[0]),  '#2e7d32', f"{path[0]} [START]"),
                (str(path[-1]), '#c62828', f"{path[-1]} [END]"),
            ]:
                if nodeStr not in self.coords or nodeStr not in self.markers:
                    continue
                try:
                    self.markers[nodeStr].delete()
                except:
                    pass
                lat, lng = self.coords[nodeStr]
                m = self.map_widget.set_marker(
                    lat, lng,
                    text=label,
                    marker_color_circle=markerColour,
                    marker_color_outside='#ffffff',
                    font=('Arial', 12, 'bold')
                )
                self.markers[nodeStr] = m
                self.current_route_items.append((nodeStr, m))

    def clear_route(self):
        highlightedNodes = set()
        for item in self.current_route_items:
            if isinstance(item, tuple):
                nodeStr, marker = item
                highlightedNodes.add(nodeStr)
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

        # put original markers back for any nodes we highlighted
        if self.is_initialized:
            for sid in highlightedNodes:
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
        return MAP_AVAILABLE

    def get_node_connections(self):
        return NODE_CONNECTIONS

    def get_node_colours(self):
        return NODE_COLOURS

    def get_coords(self):
        return self.coords
