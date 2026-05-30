# map_visualization.py - tkintermapview wrapper for the SCATS network

import math
from config import MAP_TILE_SERVER, MAP_ZOOM_LEVEL, MAP_LOCATE_ZOOM

from map import (
    NODE_CONNECTIONS,
    NODE_COLOURS,
    loadSites,
    drawEdges
)

try:
    import tkintermapview
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


class SCATSMapViewer:

    def __init__(self):
        self.mapWidget = None
        self.coords = {}
        self.markers = {}
        self.currentRouteItems = []
        self.networkPaths = []
        self.isInitialized = False

    # read lat/lng for every SCATS site from the data source and store them
    def loadCoords(self):
        sitesDF = loadSites()
        self.coords = {
            row['SCATS Number']: (row['LAT'], row['LNG'])
            for _, row in sitesDF.iterrows()
        }
        print(f"Loaded {len(self.coords)} SCATS sites with coordinates")
        return self.coords

    # return a sorted list of numeric SCATS site IDs
    def getSites(self):
        return sorted([s for s in self.coords.keys() if s.isdigit()], key=int)

    # spin up the map widget inside the given frame and centre it on Boroondara
    def createMap(self, parentFrame):
        if not MAP_AVAILABLE:
            return None

        self.mapWidget = tkintermapview.TkinterMapView(parentFrame, corner_radius=0)
        self.mapWidget.pack(fill='both', expand=True)
        self.mapWidget.set_tile_server(MAP_TILE_SERVER)

        # centre on Boroondara
        if self.coords:
            lats = [c[0] for c in self.coords.values()]
            lngs = [c[1] for c in self.coords.values()]
            self.mapWidget.set_position(sum(lats)/len(lats), sum(lngs)/len(lngs))
            self.mapWidget.set_zoom(MAP_ZOOM_LEVEL)

        return self.mapWidget

    # draw all road edges and place a marker for every SCATS site
    def drawNetwork(self):
        if not self.mapWidget or not self.coords:
            return

        self.networkPaths = drawEdges(self.mapWidget, self.coords)
        self.networkVisible = True

        for sid, (lat, lng) in self.coords.items():
            colour = NODE_COLOURS.get(sid, '#1a1a2e')
            marker = self.mapWidget.set_marker(
                lat, lng,
                text=sid,
                marker_color_circle=colour,
                marker_color_outside='#ffffff',
                font=('Arial', 10, 'bold')
            )
            self.markers[sid] = marker

        self.isInitialized = True

    # pan and zoom the map to a specific SCATS site
    def locateSite(self, siteStr):
        if not self.mapWidget or siteStr not in self.coords:
            return False
        lat, lng = self.coords[siteStr]
        self.mapWidget.set_position(lat, lng)
        self.mapWidget.set_zoom(MAP_LOCATE_ZOOM)
        return True

    # draw the route as coloured lines and optionally highlight start/end markers
    def drawRoute(self, path, color='#ff6f00', isBest=False):
        if not self.mapWidget or len(path) < 2:
            return

        for i in range(len(path) - 1):
            node1, node2 = str(path[i]), str(path[i+1])
            if node1 in self.coords and node2 in self.coords:
                lat1, lng1 = self.coords[node1]
                lat2, lng2 = self.coords[node2]
                line = self.mapWidget.set_path(
                    [(lat1, lng1), (lat2, lng2)],
                    color=color,
                    width=5 if isBest else 3
                )
                self.currentRouteItems.append(line)

        # highlight start/end markers
        if isBest:
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
                m = self.mapWidget.set_marker(
                    lat, lng,
                    text=label,
                    marker_color_circle=markerColour,
                    marker_color_outside='#ffffff',
                    font=('Arial', 12, 'bold')
                )
                self.markers[nodeStr] = m
                self.currentRouteItems.append((nodeStr, m))

    # remove all drawn route lines and restore any markers we changed
    def clearRoute(self):
        highlightedNodes = set()
        for item in self.currentRouteItems:
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
        self.currentRouteItems = []

        # put original markers back for any nodes we highlighted
        if self.isInitialized:
            for sid in highlightedNodes:
                if sid not in self.coords:
                    continue
                lat, lng = self.coords[sid]
                colour = NODE_COLOURS.get(sid, '#1a1a2e')
                self.markers[sid] = self.mapWidget.set_marker(
                    lat, lng,
                    text=sid,
                    marker_color_circle=colour,
                    marker_color_outside='#ffffff',
                    font=('Arial', 10, 'bold')
                )

    # check whether the tkintermapview package is installed
    def mapAvailable(self):
        return MAP_AVAILABLE

    # return the raw node connection dict from the map module
    def getNodeConnections(self):
        return NODE_CONNECTIONS

    # return the colour mapping for each node
    def getNodeColours(self):
        return NODE_COLOURS

    # return the lat/lng coordinate dict for all loaded sites
    def getCoords(self):
        return self.coords
