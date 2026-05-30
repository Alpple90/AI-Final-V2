# config.py
"""
Configuration settings for TBRGS
All constants in one place for easy modification
"""

# ============================================================
# TRAFFIC PREDICTION SETTINGS
# ============================================================
SEQ_LENGTH = 12              # Number of past 15-min intervals (12 = 3 hours)
BATCH_SIZE = 32              # Batch size for deep learning
LEARNING_RATE = 0.001        # Learning rate for LSTM/GRU
LSTM_EPOCHS = 50             # Number of training epochs for LSTM
GRU_EPOCHS = 50              # Number of training epochs for GRU

# ============================================================
# TRAVEL TIME SETTINGS (from PDF v1.0)
# ============================================================
SPEED_LIMIT_KPH = 60
INTERSECTION_DELAY_SEC = 30
CAPACITY_FLOW = 1500         # Vehicles per hour at capacity
FREE_FLOW_THRESHOLD = 351    # Below this, speed = speed limit

# Quadratic coefficients for flow-speed relationship
FLOW_TO_SPEED_A = -1.4648375
FLOW_TO_SPEED_B = 93.75

# ============================================================
# PATHFINDING SETTINGS
# ============================================================
DEFAULT_K_ROUTES = 5         # Number of routes to find
MAX_VISITS_PER_NODE = 5      # Prevent infinite loops

# ============================================================
# GUI SETTINGS
# ============================================================
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
LEFT_PANEL_WIDTH = 450

# ============================================================
# FILE PATHS
# ============================================================
TRAFFIC_DATA_FILE = 'Scats Data October 2006.xls'
COORDINATES_FILE = 'scatsTrueLongLat.xlsx'
MODELS_SAVE_FOLDER = 'saved_models'

# ============================================================
# MAP SETTINGS
# ============================================================
MAP_TILE_SERVER = 'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png'
MAP_ZOOM_LEVEL = 13
MAP_LOCATE_ZOOM = 15