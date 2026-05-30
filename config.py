# config.py - all app constants in one place

# travel time settings (from PDF v1.0)
SPEED_LIMIT_KPH = 60
INTERSECTION_DELAY_SEC = 30
CAPACITY_FLOW = 1500         # vehicles per hour at capacity
FREE_FLOW_THRESHOLD = 351    # below this, speed = speed limit

# quadratic coefficients for flow-speed relationship
FLOW_TO_SPEED_A = -1.4648375
FLOW_TO_SPEED_B = 93.75

# pathfinding settings
DEFAULT_K_ROUTES = 5         # number of routes to find
MAX_VISITS_PER_NODE = 5      # prevent infinite loops

# GUI settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
LEFT_PANEL_WIDTH = 450

MODELS_SAVE_FOLDER = 'saved_models'

# map settings
MAP_TILE_SERVER = 'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png'
MAP_ZOOM_LEVEL = 13
MAP_LOCATE_ZOOM = 15