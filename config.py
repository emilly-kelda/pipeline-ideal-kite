"""
config.py
─────────
Central place for all constants used across the pipeline.
Instead of magic numbers scattered through multiple files,
every threshold and path lives here and is imported where needed.
"""

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR     = "data/raw"
CLEAN_DIR   = "data/clean"
OUTPUTS_DIR = "outputs"

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Wind validation thresholds (knots) ───────────────────────────────────────
WIND_MIN_KITE  = 5    # below this → sensor error or calm, not kiteable
WIND_MAX_KITE  = 60   # above this → sensor spike

# ── Rider validation ─────────────────────────────────────────────────────────
WEIGHT_MIN_KG  = 40
WEIGHT_MAX_KG  = 150

# ── Kite size validation (m²) ────────────────────────────────────────────────
KITE_SIZE_MIN  = 4
KITE_SIZE_MAX  = 21

# ── Unit conversion factors ──────────────────────────────────────────────────
KMH_TO_KNOTS   = 1 / 1.852
MS_TO_KNOTS    = 1 / 0.5144
MPH_TO_KNOTS   = 0.868976

BEAUFORT_TO_KNOTS = {
    0: 0,  1: 2,  2: 5,  3: 9,  4: 13, 5: 17,
    6: 22, 7: 27, 8: 31, 9: 36, 10: 41, 11: 47, 12: 55,
}

# ── Quiver optimizer ─────────────────────────────────────────────────────────
COVERAGE_WEIGHT   = 0.6   # how much wind coverage matters vs price
PRICE_WEIGHT      = 0.4   # how much price/value matters

# ── Safe wind directions per location ────────────────────────────────────────
# Each location has a list of (min_deg, max_deg) tuples defining safe ranges.
# Wind coming from outside these ranges is offshore or dangerous.
# Based on real kitespot research — see project README for sources.

SAFE_DIRECTIONS = {
    "LOC01": [(90,  180)],          # Cumbuco — SE trade winds
    "LOC02": [(90,  180)],          # Jericoacoara — SE trade winds
    "LOC03": [(225, 315)],          # Tarifa — Poniente (W/SW) safe for all levels
    "LOC04": [(45,  135)],          # Maui — NE/E trade winds
    "LOC05": [(90,  180)],          # Cape Town — SE Cape Doctor
    "LOC06": [(0,   90)],           # Cabarete — NE trade winds
    "LOC07": [(0,   90)],           # Dakhla — N/NE trade winds
    "LOC08": [(90,  180)],          # Guajiru — SE trade winds
    "LOC09": [(0,   45), (135, 225)], # Lake Garda — N morning + S afternoon thermals
    "LOC10": [(0,   90), (180, 270)], # Boracay — NE Amihan + SW Habagat seasons
}