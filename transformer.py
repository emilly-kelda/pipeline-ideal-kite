"""
transformer.py
──────────────
Takes the three clean CSVs and engineers six scores for every
rider × kite combination.

Output:
    data/clean/scored_combinations.csv

Each row = one rider + one kite, with:
    coverage_score    % of wind days inside kite's range
    skill_score       % of days inside rider's skill+style window
    weight_score      how well kite size matches rider weight
    direction_score   % of days with safe wind direction
    gust_score        % of days where gusts are safe
    overall_score     weighted combination of all five
"""

import numpy as np
import pandas as pd

from config import (
    RAW_DIR, CLEAN_DIR,
    SAFE_DIRECTIONS,
    COVERAGE_WEIGHT, PRICE_WEIGHT,
)
import os

os.makedirs(CLEAN_DIR, exist_ok=True)

# ── Skill + style comfort windows (knots) ────────────────────────────────────
# Research-backed wind ranges per skill level and riding style.
# A day counts toward skill_score only if wind falls inside
# BOTH the kite's range AND the rider's comfort window.

SKILL_WINDOWS = {
    ("beginner",     "freeride"):   (12, 18),
    ("beginner",     "wave"):       (12, 18),
    ("beginner",     "big_air"):    (12, 18),
    ("beginner",     "freestyle"):  (12, 18),
    ("intermediate", "freeride"):   (10, 25),
    ("intermediate", "wave"):       (14, 25),
    ("intermediate", "big_air"):    (18, 28),
    ("intermediate", "freestyle"):  (14, 25),
    ("advanced",     "freeride"):   (8,  30),
    ("advanced",     "wave"):       (12, 28),
    ("advanced",     "big_air"):    (20, 35),
    ("advanced",     "freestyle"):  (14, 30),
}

# ── Scoring functions ─────────────────────────────────────────────────────────

def coverage_score(wind_df, kite):
    """% of wind days where speed falls inside kite's wind range.

    NaN wind_min is filled using size-based estimate from WIND_RANGES
    in config.py — transformer.py handles missing minimums here.
    """
    w_min = kite["wind_min_kn"]
    w_max = kite["wind_max_kn"]

    if pd.isna(w_min) or pd.isna(w_max):
        return np.nan

    valid = wind_df["wind_speed_kn"].dropna()
    if len(valid) == 0:
        return 0.0

    inside = valid.between(w_min, w_max)
    return round(inside.sum() / len(valid), 3)


def skill_score(wind_df, kite, rider):
    """% of wind days inside BOTH kite range AND rider's comfort window."""
    w_min = kite["wind_min_kn"]
    w_max = kite["wind_max_kn"]

    if pd.isna(w_min) or pd.isna(w_max):
        return np.nan

    skill  = rider["skill_level"]
    style  = rider["preferred_style"]
    window = SKILL_WINDOWS.get((skill, style), (12, 25))  # safe default

    s_min, s_max = window
    effective_min = max(w_min, s_min)
    effective_max = min(w_max, s_max)

    if effective_min >= effective_max:
        return 0.0  # no overlap between kite range and rider window

    valid = wind_df["wind_speed_kn"].dropna()
    if len(valid) == 0:
        return 0.0

    inside = valid.between(effective_min, effective_max)
    return round(inside.sum() / len(valid), 3)


def weight_score(wind_df, kite, rider):
    """How well kite size matches rider weight at local average wind.

    Formula: ideal_size = (weight_kg / avg_wind_kn) * 2.2
    Score = 1 - abs(actual_size - ideal_size) / ideal_size
    Clamped to 0-1 range.
    """
    weight  = rider["weight_kg"]
    size    = kite["size_m2"]
    avg_wind = wind_df["wind_speed_kn"].dropna().mean()

    if pd.isna(weight) or pd.isna(size) or pd.isna(avg_wind) or avg_wind == 0:
        return np.nan

    ideal = (weight / avg_wind) * 2.2
    score = 1 - abs(size - ideal) / ideal
    return round(float(np.clip(score, 0, 1)), 3)


def direction_score(wind_df, location_id):
    """% of days where wind direction is safe for this location.

    Safe ranges come from SAFE_DIRECTIONS in config.py —
    research-backed per location based on coastline orientation.
    """
    safe_ranges = SAFE_DIRECTIONS.get(location_id, [])
    if not safe_ranges:
        return np.nan

    dirs = wind_df["wind_direction_deg"].dropna()
    if len(dirs) == 0:
        return 0.0

    def is_safe(deg):
        return any(lo <= deg <= hi for lo, hi in safe_ranges)

    safe = dirs.apply(is_safe)
    return round(safe.sum() / len(dirs), 3)


def gust_score(wind_df):
    """% of days where gust ratio is within safe limits.

    Gust ratio = gust_kn / wind_speed_kn
    Ratio > 1.5 means dangerously gusty conditions.
    """
    df = wind_df.dropna(subset=["wind_speed_kn", "gust_kn"])
    if len(df) == 0:
        return np.nan

    df = df[df["wind_speed_kn"] > 0].copy()
    df["gust_ratio"] = df["gust_kn"] / df["wind_speed_kn"]
    safe = df["gust_ratio"].le(1.5)
    return round(safe.sum() / len(df), 3)

# ── Main scoring loop ─────────────────────────────────────────────────────────

def run_transformer():
    print("\n── Transformer ──────────────────────────────────────")

    # Load all three clean files
    kites  = pd.read_csv(f"{CLEAN_DIR}/kite_catalog_clean.csv")
    wind   = pd.read_csv(f"{CLEAN_DIR}/wind_observations_clean.csv")
    riders = pd.read_csv(f"{CLEAN_DIR}/rider_profiles_clean.csv")

    print(f"   Kites:   {len(kites)} rows")
    print(f"   Wind:    {len(wind)} rows")
    print(f"   Riders:  {len(riders)} rows")

    results = []

    for _, rider in riders.iterrows():
        loc_id = rider["location_id"]

        if pd.isna(loc_id) or str(loc_id).strip() == "":
            print(f"   [!] Skipping {rider['name']} — location_id is missing or invalid")
            continue

        # Filter wind to this rider's home location only
        loc_wind = wind[wind["location_id"] == loc_id].copy()

        # Calculate location-level scores once per rider
        # (direction and gust don't depend on the kite)
        dir_score  = direction_score(loc_wind, loc_id)
        gust_score_val = gust_score(loc_wind)

        # Score every kite against this rider
        for _, kite in kites.iterrows():

            # Skip kites the rider already owns
            existing = rider["existing_sizes"]
            if isinstance(existing, str):
                existing = eval(existing)
            if kite["size_m2"] in existing:
                continue

            # Skip kites outside rider's budget
            if pd.notna(rider["budget_usd"]) and pd.notna(kite["price_usd"]):
                if kite["price_usd"] > rider["budget_usd"]:
                    continue

            cov   = coverage_score(loc_wind, kite)
            skill = skill_score(loc_wind, kite, rider)
            wgt   = weight_score(loc_wind, kite, rider)

            # Overall score — weighted combination
            # Coverage and skill matter most, weight and direction secondary
            scores = [cov, skill, wgt, dir_score, gust_score_val]
            if all(pd.isna(s) for s in scores):
                overall = np.nan
            else:
                weights = [0.30, 0.25, 0.20, 0.15, 0.10]
                weighted = sum(s * w for s, w in zip(scores, weights)
                               if pd.notna(s))
                weight_sum = sum(w for s, w in zip(scores, weights)
                                 if pd.notna(s))
                overall = round(weighted / weight_sum, 3)

            results.append({
                "rider_id":        rider["rider_id"],
                "rider_name":      rider["name"],
                "location_id":     loc_id,
                "kite_id":         kite["kite_id"],
                "brand":           kite["brand"],
                "model":           kite["model"],
                "size_m2":         kite["size_m2"],
                "price_usd":       kite["price_usd"],
                "coverage_score":  cov,
                "skill_score":     skill,
                "weight_score":    wgt,
                "direction_score": dir_score,
                "gust_score":      gust_score_val,
                "overall_score":   overall,
            })

    df = pd.DataFrame(results)
    df.sort_values(["rider_id", "overall_score"],
                   ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(f"{CLEAN_DIR}/scored_combinations.csv", index=False)
    print(f"   Scored:  {len(df)} combinations")
    print("   Saved → data/clean/scored_combinations.csv")
    return df


if __name__ == "__main__":
    scored_df = run_transformer()
    print("\nNext step → run analyzer.py")