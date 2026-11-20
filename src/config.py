import os

SEED = 42

LAT_MIN, LAT_MAX = 5.22, 5.80
LON_MIN, LON_MAX = -72.80, -72.36
CELL_RESOLUTION_M = 450

TOWN_CENTERS = {
    "Pisba":          (5.722, -72.486, 420, 1000),
    "Paya":           (5.625, -72.424, 820, 2000),
    "Labranzagrande": (5.563, -72.578, 1500, 3700),
    "Pajarito":       (5.293, -72.703, 995, 1419),
}
VILLAGES_PER_MUNICIPALITY = 16
MIN_VILLAGE_SEPARATION_KM = 2.0

ALTITUDE_PLAINS = 600
ALTITUDE_PARAMO = 3900
MOUNTAIN_AMPLITUDE_M = (680, 360, 100)
MOUNTAIN_SCALE_CELLS = (11, 5, 2)

MAIN_VALLEY_WIDTH_KM = 2.6
MAIN_VALLEY_DEPTH_M = 650
NUM_STREAMS = 5
STREAM_VALLEY_WIDTH_KM = 1.2
STREAM_VALLEY_DEPTH_M = 280

LANDSLIDE_BASE_RATE = 0.035

CLASSES = ["Apto", "Condicionado", "No apto"]
SLOPE_CONDITIONAL = 15.0
SLOPE_UNSUITABLE = 30.0
ALTITUDE_CONDITIONAL = 3000
ALTITUDE_UNSUITABLE = 3300
RIVER_UNSUITABLE_M = 120
RIVER_CONDITIONAL_M = 300
PEAK_CONDITIONAL_M = 45
PEAK_UNSUITABLE_M = 90

TRAINING_FRACTION = 0.30
LABEL_NOISE = 0.05
NUM_TREES = 120
TREE_DEPTH = 12
CV_BLOCK_SIZE_KM = 8
CV_FOLDS = 5

TOBLER_VMAX_KMH = 6.0
OFF_ROAD_FACTOR = 0.6
STREAM_FACTOR = 0.35
VEHICLE_SPEED_TERTIARY_KMH = 22
VEHICLE_SPEED_SECONDARY_KMH = 35
VEHICLE_SLOPE_PENALTY = 4.0

TRAVEL_MODES = ["A pie", "Vehículo por vía + a pie"]
CRITERIA = ["Tiempo de viaje (considera pendientes)", "Distancia 3D por camino (km)"]

SINUOSITY_MIN, SINUOSITY_MAX = 1.15, 1.60

REMOVED_ROADS_FRACTION = 0.22
BRIDGE_COST = 40.0

CENTER_TYPES = {
    "Centro de salud": {"k": 3, "allowed_classes": [0], "requires_road": False,
                        "emoji": "🏥", "color": "#1f6feb", "symbol": "cross"},
    "Colegio":         {"k": 3, "allowed_classes": [0, 1], "requires_road": False,
                        "emoji": "🏫", "color": "#8250df", "symbol": "square"},
    "Hospital":        {"k": 1, "allowed_classes": [0], "requires_road": True,
                        "emoji": "🚑", "color": "#cf222e", "symbol": "diamond"},
}
K_MAX_PER_KIND = 8

FAR_THRESHOLD_KM = 25
FAR_THRESHOLD_MIN = 120
THRESHOLD_1H_MIN = 60
NEAR_THRESHOLD_KM = 8

TARGETS = {
    "N1": 20.0,
    "N2": 30.0,
    "N3": 10.0,
    "N4": 0.0,
    "N6": 1.0,
    "T1": 2000.0,
    "T2_slope": 1.2, "T2_mb": 50.0,
    "T3": 1000.0,
    "T4_max": 2.0, "T4_mean": 1.5,
    "T5": 90, "T6": 90, "T7": 70,
    "T8_auc": 0.85, "T8_recall": 0.80,
    "T9": 90.0,
    "T10": 5.0,
    "T11": 100.0,
}

TEST_SIZES = {"quick": [500, 1000, 2000, 5000, 10000], "full": [500, 1000, 2000, 5000, 10000, 20000, 50000]}
MAX_V_MATRIX = {"quick": 1000, "full": 3000}
APPROXIMATION_INSTANCES = {"quick": 100, "full": 300}
TABLE_III_REGIONS = {"quick": 10, "full": 30}

LM_STUDIO_URL = os.environ.get("ACCESO_LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = os.environ.get("ACCESO_LM_STUDIO_MODEL", "")

CLOUD_URL = os.environ.get("ACCESO_CLOUD_URL", "https://api.openai.com/v1")
CLOUD_API_KEY = os.environ.get("ACCESO_CLOUD_API_KEY", "")
CLOUD_MODEL = os.environ.get("ACCESO_CLOUD_MODEL", "")

CONNECT_TIMEOUT_S = 3
RESPONSE_TIMEOUT_S = 10
MAX_TOOL_ROUNDS = 4
TEMPERATURE = 0.2
