"""
data_generator.py
─────────────────
Generates three synthetic messy CSV files that simulate
real-world data collection problems.

Run this first — everything else in the pipeline depends on these files.

Output:
    data/raw/kite_catalog_raw.csv
    data/raw/wind_observations_raw.csv
    data/raw/rider_profiles_raw.csv
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

from config import (
    RAW_DIR, SEED,
    KITE_SIZE_MIN, KITE_SIZE_MAX,
    BEAUFORT_TO_KNOTS,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

# ── Output folder ─────────────────────────────────────────────────────────────
os.makedirs(RAW_DIR, exist_ok=True)

# ── Brand variants ────────────────────────────────────────────────────────────
# Each brand has a list of messy spellings someone might enter
BRAND_VARIANTS = {
    "Duotone":   ["Duotone", "duotone", "DUOTONE", "Duo-tone"],
    "Cabrinha":  ["Cabrinha", "cabrinha", "CABRINHA", "Cabrinha Kites"],
    "North":     ["North", "north", "NORTH", "North Kiteboarding"],
    "F-One":     ["F-One", "f-one", "F One", "Fone"],
    "Core":      ["Core", "CORE", "Core Kiteboarding"],
    "Ozone":     ["Ozone", "ozone", "OZONE"],
    "Naish":     ["Naish", "naish", "NAISH"],
    "Slingshot": ["Slingshot", "slingshot", "SLINGSHOT"],
    "Airush":    ["Airush", "airush", "AIRUSH"],
    "Reedin":    ["Reedin", "reedin", "REEDIN"],
}

BRAND_MODELS = {
    "Duotone":   ["Rebel SLS", "Neo SLS", "Dice", "Juice", "Evo"],
    "Cabrinha":  ["Switchblade", "Drifter", "Moto", "Nitro", "Contra"],
    "North":     ["Orbit", "Carve", "Pulse", "Reach", "Neo"],
    "F-One":     ["Bandit", "Breeze", "Strike", "Diablo", "Revolt"],
    "Core":      ["XR7", "Section", "Nexus", "Riot", "Free"],
    "Ozone":     ["Enduro", "Edge", "Catalyst", "Reo", "Chrono"],
    "Naish":     ["Pivot", "Triad", "Boxer", "Slash", "Hero"],
    "Slingshot": ["Code", "Rally", "Ghost", "Crisis", "RPX"],
    "Airush":    ["Lift", "Ultra", "Diamond", "Origin", "Razor"],
    "Reedin":    ["Super Model", "Expert Model", "Hyper Model"],
}

KITE_SIZES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]

WIND_RANGES = {
    5:  (20, 40), 6:  (18, 38), 7:  (16, 35),
    8:  (14, 32), 9:  (12, 30), 10: (11, 28),
    11: (10, 26), 12: (9,  25), 13: (8,  23),
    14: (7,  22), 15: (6,  20), 17: (5,  18),
}

def messy_size(size):
    """Return a kite size in one of several inconsistent string formats."""
    options = [
        str(size),          # "12"
        f"{size}m",         # "12m"
        f"{size} m",        # "12 m"
        f"{size}m²",        # "12m²"
        f"{float(size)}",   # "12.0"
        f"{size} sqm",      # "12 sqm"
    ]
    return random.choice(options)


def messy_price(price_usd):
    """Return a price in one of several currency formats."""
    eur = round(price_usd * 0.92, 2)
    brl = round(price_usd * 5.05, 2)
    options = [
        f"${price_usd}",
        f"USD {price_usd}",
        f"€{eur}",
        f"R${brl}",
        str(price_usd),
    ]
    return random.choice(options)


def messy_wind_range(min_kn, max_kn):
    """Return wind range in inconsistent formats, sometimes missing the minimum."""
    min_kmh = round(min_kn * 1.852)
    max_kmh = round(max_kn * 1.852)
    options = [
        (str(min_kn),       str(max_kn)),
        (f"{min_kn}kn",     f"{max_kn}kn"),
        (f"{min_kn} knots", f"{max_kn} knots"),
        (f"{min_kmh} km/h", f"{max_kmh} km/h"),
        (None,              str(max_kn)),   # missing minimum — planted gap
    ]
    return random.choice(options)

# ── Build kite catalog ────────────────────────────────────────────────────────
rows = []
kite_id = 1

for brand_clean, brand_variants in BRAND_VARIANTS.items():
    models = BRAND_MODELS[brand_clean]

    for model in models:
        model_sizes = random.sample(KITE_SIZES, k=random.randint(4, 8))
        year = random.choice([2023, 2024, 2025])

        for size in model_sizes:
            base_price = random.randint(900, 2200)
            min_kn, max_kn = WIND_RANGES[size]
            wind_min_str, wind_max_str = messy_wind_range(min_kn, max_kn)

            rows.append({
                "kite_id":        kite_id,
                "brand":          random.choice(brand_variants),
                "model":          model,
                "year":           year,
                "size_m2":        messy_size(size),
                "wind_range_min": wind_min_str,
                "wind_range_max": wind_max_str,
                "price":          messy_price(base_price),
            })
            kite_id += 1

# Build DataFrame and inject duplicates
kite_df = pd.DataFrame(rows)

n_dupes = int(len(kite_df) * 0.05)
dupes = kite_df.sample(n=n_dupes, random_state=SEED)
kite_df = pd.concat([kite_df, dupes], ignore_index=True)
kite_df = kite_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

kite_df.to_csv(f"{RAW_DIR}/kite_catalog_raw.csv", index=False)
print(f"[✓] kite_catalog_raw.csv — {len(kite_df)} rows")

# ── Locations ─────────────────────────────────────────────────────────────────
LOCATIONS = [
    {"id": "LOC01", "name": "Cumbuco, Brazil",          "avg_kn": 18, "std": 4},
    {"id": "LOC02", "name": "Jericoacoara, Brazil",      "avg_kn": 20, "std": 3},
    {"id": "LOC03", "name": "Tarifa, Spain",             "avg_kn": 22, "std": 7},
    {"id": "LOC04", "name": "Maui, Hawaii",              "avg_kn": 16, "std": 5},
    {"id": "LOC05", "name": "Cape Town, South Africa",   "avg_kn": 24, "std": 8},
    {"id": "LOC06", "name": "Cabarete, Dominican Rep.",  "avg_kn": 17, "std": 4},
    {"id": "LOC07", "name": "Dakhla, Morocco",           "avg_kn": 22, "std": 5},
    {"id": "LOC08", "name": "Ilha do Guajiru, Brazil",   "avg_kn": 19, "std": 3},
    {"id": "LOC09", "name": "Lake Garda, Italy",         "avg_kn": 12, "std": 6},
    {"id": "LOC10", "name": "Boracay, Philippines",      "avg_kn": 15, "std": 4},
]

LOCATION_NAME_VARIANTS = {
    "Cumbuco, Brazil":          ["Cumbuco, Brazil", "Cumbuco", "cumbuco", " Cumbuco"],
    "Jericoacoara, Brazil":     ["Jericoacoara, Brazil", "Jeri", "jericoacoara"],
    "Tarifa, Spain":            ["Tarifa, Spain", "Tarifa", "TARIFA"],
    "Maui, Hawaii":             ["Maui, Hawaii", "Maui", "maui"],
    "Cape Town, South Africa":  ["Cape Town, South Africa", "Cape Town", "capetown"],
    "Cabarete, Dominican Rep.": ["Cabarete, Dominican Rep.", "Cabarete", "cabarete"],
    "Dakhla, Morocco":          ["Dakhla, Morocco", "Dakhla", "DAKHLA"],
    "Ilha do Guajiru, Brazil":  ["Ilha do Guajiru, Brazil", "Guajiru", "Ilha Guajiru"],
    "Lake Garda, Italy":        ["Lake Garda, Italy", "Garda", "Lago di Garda"],
    "Boracay, Philippines":     ["Boracay, Philippines", "Boracay", "boracay"],
}

DATE_FORMATS = [
    "%Y-%m-%d",   # 2024-03-15
    "%d/%m/%Y",   # 15/03/2024
    "%m/%d/%Y",   # 03/15/2024
    "%d %b %Y",   # 15 Mar 2024
    "%b %d %Y",   # Mar 15 2024
]


def messy_date(d):
    """Return a date in one of five inconsistent string formats."""
    options = [d.strftime(fmt) for fmt in DATE_FORMATS]
    return random.choice(options)


def messy_wind_speed(knots):
    """Return wind speed as a string in a random unit."""
    kmh      = round(knots * 1.852, 1)
    ms       = round(knots * 0.5144, 1)
    beaufort = min(12, max(0, int(knots / 3.5)))

    options = [
        (f"{knots} knots", "knots"),
        (f"{kmh} km/h",    "km/h"),
        (f"{ms} m/s",      "m/s"),
        (str(beaufort),    "beaufort"),
        (str(knots),       "knots"),   # bare number, unit unknown
    ]
    return random.choice(options)


def messy_direction(deg):
    """Return wind direction as cardinal, full name, or degrees."""
    CARDINAL = {
        0: "N", 45: "NE", 90: "E", 135: "SE",
        180: "S", 225: "SW", 270: "W", 315: "NW"
    }
    full = {
        "N": "north", "NE": "northeast", "E": "east", "SE": "southeast",
        "S": "south", "SW": "southwest", "W": "west", "NW": "northwest"
    }
    nearest = min(CARDINAL, key=lambda k: abs(k - deg))
    abbr = CARDINAL[nearest]

    options = [abbr, full[abbr], str(deg), f"{deg}°"]
    return random.choice(options)

# ── Build wind observations ───────────────────────────────────────────────────
obs_rows = []

start_date = date(2023, 1, 1)
end_date   = date(2024, 12, 31)

all_dates = [start_date + timedelta(days=i)
             for i in range((end_date - start_date).days + 1)]

BLACKOUT_LOCATIONS = {"LOC03", "LOC07"}
BLACKOUT_DATES     = set(start_date + timedelta(days=i) for i in range(60, 74))

for loc in LOCATIONS:
    manual_block_dates = set(random.sample(all_dates, k=20))

    for d in all_dates:
        if loc["id"] in BLACKOUT_LOCATIONS and d in BLACKOUT_DATES:
            continue

        wind_kn = max(0.0, round(float(np.random.normal(loc["avg_kn"], loc["std"])), 1))

        if d in manual_block_dates:
            wind_kn = float(round(loc["avg_kn"]))

        deg     = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        gust_kn = round(wind_kn * random.uniform(1.1, 1.4), 1)

        speed_str, unit = messy_wind_speed(wind_kn)

        if random.random() < 0.08:
            gust_kn = round(wind_kn * random.uniform(0.4, 0.9), 1)

        gust_out = None if random.random() < 0.05 else gust_kn

        if random.random() < 0.03:
            wind_kn   = random.choice([0.0, 150.0, -5.0])
            speed_str = str(wind_kn)
            unit      = "knots"

        obs_rows.append({
            "location_id":    loc["id"],
            "location_name":  random.choice(LOCATION_NAME_VARIANTS[loc["name"]]),
            "date":           messy_date(d),
            "wind_speed":     speed_str,
            "wind_unit":      unit,
            "wind_direction": messy_direction(deg),
            "gust_speed":     gust_out,
        })

wind_df = pd.DataFrame(obs_rows)

n_near_dupes = int(len(wind_df) * 0.02)
near_dupes   = wind_df.sample(n=n_near_dupes, random_state=SEED).copy()
wind_df      = pd.concat([wind_df, near_dupes], ignore_index=True)
wind_df      = wind_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

wind_df.to_csv(f"{RAW_DIR}/wind_observations_raw.csv", index=False)
print(f"[✓] wind_observations_raw.csv — {len(wind_df)} rows")

# ── Rider profiles ────────────────────────────────────────────────────────────
SKILL_VARIANTS = {
    "beginner":     ["beginner", "Beginner", "beg", "novice", "newbie"],
    "intermediate": ["intermediate", "Intermediate", "inter", "mid-level"],
    "advanced":     ["advanced", "Advanced", "adv", "expert", "pro"],
}

STYLE_VARIANTS = {
    "freeride":  ["freeride", "Freeride", "free ride", "cruising"],
    "big_air":   ["big_air", "Big Air", "bigair", "jumping"],
    "wave":      ["wave", "Wave", "surf", "wave riding"],
    "freestyle": ["freestyle", "Freestyle", "tricks", "unhook"],
}

RIDER_NAMES = [
    "Ana Lima", "Bruno Costa", "Carla Mendes", "Diego Souza", "Elena Faria",
    "Felipe Rocha", "Gabriela Nunes", "Henrique Alves", "Isabel Martins",
    "João Pires", "Karina Silva", "Lucas Ferreira", "Mariana Gomes",
    "Nicolas Teixeira", "Olivia Santos", "Pedro Dias", "Rafaela Cruz",
    "Samuel Barros", "Thais Oliveira", "Victor Cardoso",
]


def messy_weight(kg):
    """Return weight as kg string, lbs string, or bare number."""
    lbs = round(kg * 2.205, 1)
    options = [f"{kg} kg", f"{lbs} lbs", str(kg), str(lbs)]
    return random.choice(options)


def messy_budget(usd):
    """Return budget in USD, EUR, or BRL."""
    eur = round(usd * 0.92)
    brl = round(usd * 5.05)
    options = [f"${usd}", f"USD {usd}", f"€{eur}", f"R${brl}", str(usd)]
    return random.choice(options)


def messy_existing_kites(sizes):
    """Return existing kite sizes in inconsistent string formats."""
    if not sizes:
        return None
    size_strs = [f"{s}m" if random.random() > 0.3 else str(s) for s in sizes]
    fmt = random.choice(["comma", "plus", "slash"])
    if fmt == "comma":  return ", ".join(size_strs)
    if fmt == "plus":   return "+".join(size_strs)
    return "/".join(size_strs)


# ── Build rider profiles ──────────────────────────────────────────────────────
rider_rows = []
location_ids = [loc["id"] for loc in LOCATIONS]

for i, name in enumerate(RIDER_NAMES):
    weight_kg  = round(random.uniform(55, 100), 1)
    budget_usd = random.choice([1500, 2000, 2500, 3000, 3500, 4000, 5000])
    skill      = random.choice(list(SKILL_VARIANTS.keys()))
    style      = random.choice(list(STYLE_VARIANTS.keys()))
    home_loc   = random.choice(location_ids)

    n_existing     = random.choice([0, 0, 1, 2])
    existing_sizes = random.sample(KITE_SIZES, k=n_existing) if n_existing else []

    # Plant one orphaned location ID — rider 8 points to a location
    # that doesn't exist in the wind data. Tests FK validation in cleaner.py.
    assigned_loc = "LOC99" if i == 7 else home_loc

    rider_rows.append({
        "rider_id":            f"RDR{i+1:02d}",
        "name":                name,
        "weight":              messy_weight(weight_kg),
        "skill_level":         random.choice(SKILL_VARIANTS[skill]),
        "preferred_style":     random.choice(STYLE_VARIANTS[style]),
        "location_id":         assigned_loc,
        "budget":              messy_budget(budget_usd),
        "existing_kite_sizes": messy_existing_kites(existing_sizes),
    })

rider_df = pd.DataFrame(rider_rows)
rider_df.to_csv(f"{RAW_DIR}/rider_profiles_raw.csv", index=False)
print(f"[✓] rider_profiles_raw.csv — {len(rider_df)} rows")

print("\nAll raw files written to data/raw/")
print("Next step → run cleaner.py")