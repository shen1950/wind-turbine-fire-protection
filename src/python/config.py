SEED = 2026

SAMPLE_RATE_HZ = 1
SEQ_LEN = 60
N_PER_CLASS = 300
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

FEATURE_NAMES = [
    "temperature",
    "smoke",
    "gas",
    "infrared_hotspot",
    "current_fluctuation",
]

CLASS_NAMES = {
    0: "normal",
    1: "overheat",
    2: "smoldering",
    3: "open_fire",
}

BINARY_CLASS_NAMES = {
    0: "normal",
    1: "fire_risk",
}

CRITICAL_TIME = 55
ALARM_THRESHOLD = 0.50

