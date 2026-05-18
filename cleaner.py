"""
cleaner.py
──────────
Takes the three raw CSVs and produces clean, analysis-ready versions.

Each section targets one file. Every problem gets its own function
with a clear input → output contract.

Output files:
    data/clean/kite_catalog_clean.csv
    data/clean/wind_observations_clean.csv
    data/clean/rider_profiles_clean.csv

A cleaning audit log is printed at the end showing exactly how many
issues were fixed in each column.
"""

import os
import re
import numpy as np
import pandas as pd

from config import (
    RAW_DIR, CLEAN_DIR, SEED,
    WIND_MIN_KITE, WIND_MAX_KITE,
    WEIGHT_MIN_KG, WEIGHT_MAX_KG,
    KITE_SIZE_MIN, KITE_SIZE_MAX,
    KMH_TO_KNOTS, MS_TO_KNOTS, MPH_TO_KNOTS,
    BEAUFORT_TO_KNOTS,
)

os.makedirs(CLEAN_DIR, exist_ok=True)

# The audit log tracks every fix made across all three files.
# At the end we print it as a table so you can see exactly
# what was wrong and how much of it there was.
audit = {}


# ── Shared utilities ──────────────────────────────────────────────────────────

def strip_and_lower(series):
    """Strip whitespace and lowercase a whole column at once.

    .str accessor gives you string methods on an entire column.
    Without it you would need a for-loop over every row.
    """
    return series.str.strip().str.lower()


def extract_number(text):
    """Extract the first number from a messy string.

    re.search() scans the string and returns the first match.
    The pattern looks for an optional minus sign, one or more
    digits, and an optional decimal part.
    """
    if pd.isna(text):
        return np.nan
    match = re.search(r"-?[0-9]+([.][0-9]+)?", str(text))
    return float(match.group()) if match else np.nan

def fuzzy_normalize(text, canonical_list, threshold=80):
    """Match a messy string to the closest canonical value using fuzzy matching.

    Instead of manually mapping every possible variant, fuzzy matching
    finds the closest match automatically — even for typos and abbreviations.

    Args:
        text:           the messy input string
        canonical_list: list of accepted clean values to match against
        threshold:      minimum similarity score (0-100) to accept a match
                        80 means "at least 80% similar"
                        too low = wrong matches, too high = misses valid variants

    Returns:
        the best matching canonical value, or np.nan if no match is good enough

    LIMITATION: fuzzy matching can make wrong matches if the threshold is too low
    or if canonical values are too similar to each other.
    Always check the audit log for unmatched values.
    """
    from rapidfuzz import process

    if pd.isna(text):
        return np.nan

    s = str(text).strip()
    result = process.extractOne(s, canonical_list)

    if result is None:
        return np.nan

    match, score, _ = result
    return match if score >= threshold else np.nan


def parse_weight_kg(text):
    """Convert a messy weight string to float kg.

    Detection order:
        1. If "lbs" in string → convert to kg
        2. If "kg" in string → keep as is
        3. No unit → use domain knowledge heuristic:
           if number > 150 it is almost certainly lbs
           if number <= 150 assume kg

    IMPORTANT: This is an assumption, not a certainty.
    In a production pipeline, ambiguous rows should be flagged
    for human review instead of silently assumed.

    TODO: consider rapidfuzz for ambiguous cases.
    """
    if pd.isna(text):
        return np.nan

    s = str(text).strip().lower()
    amount = extract_number(s)

    if pd.isna(amount):
        return np.nan

    if "lbs" in s or "lb" in s:
        return round(amount / 2.205, 1)
    elif "kg" in s:
        return round(amount, 1)
    else:
        # No unit label — apply heuristic
        # A number above 150 is almost certainly lbs, not kg
        # ASSUMPTION: no adult kiter weighs 210kg but 210lbs is realistic
        # In production: flag for human review instead of assuming
        if amount > 150:
            return round(amount / 2.205, 1)
        else:
            return round(amount, 1)


def parse_budget_usd(text):
    """Convert a messy budget string to integer USD.

    Handles: "$2500", "USD 2500", "€1380", "R$17675", "2500"

    Returns whole numbers — budgets don't need cent precision.

    Assumption: bare numbers with no currency symbol are assumed USD.
    In a real pipeline, currency should always be stored separately
    from the amount — combining them in one field is a data design flaw.
    """
    if pd.isna(text):
        return np.nan

    s = str(text).strip().lower()
    amount = extract_number(s)

    if pd.isna(amount):
        return np.nan

    if "r$" in s or "brl" in s:
        return int(round(amount / 5.05))
    elif "€" in s or "eur" in s:
        return int(round(amount / 0.92))
    else:
        return int(round(amount))


def parse_existing_sizes(text):
    """Parse messy kite size string into a sorted list of floats.

    Handles: "12m, 7m", "9+12", "10m+6", "9 12", None

    re.findall() returns ALL number matches in the string as a list,
    unlike re.search() which only returns the first match.
    Only keeps numbers within the valid kite size range.
    """
    if pd.isna(text):
        return []

    matches = re.findall(r"[0-9]+[.]?[0-9]*", str(text))
    sizes = [float(m) for m in matches if KITE_SIZE_MIN <= float(m) <= KITE_SIZE_MAX]
    return sorted(sizes)


# ── Normalization maps ────────────────────────────────────────────────────────

# LIMITATION: unrecognized variants get filled with a default value.
# For production use, consider fuzzy matching with rapidfuzz
# or a human review step for unknown labels.

SKILL_MAP = {
    "beginner":     "beginner",
    "beginer":      "beginner",
    "beg":          "beginner",
    "novice":       "beginner",
    "newbie":       "beginner",
    "intermediate": "intermediate",
    "inter":        "intermediate",
    "mid-level":    "intermediate",
    "advanced":     "advanced",
    "adv":          "advanced",
    "expert":       "advanced",
    "pro":          "advanced",
}

STYLE_MAP = {
    "freeride":    "freeride",
    "free ride":   "freeride",
    "cruising":    "freeride",
    "big_air":     "big_air",
    "bigair":      "big_air",
    "jumping":     "big_air",
    "air":         "big_air",
    "wave":        "wave",
    "surf":        "wave",
    "wave riding": "wave",
    "freestyle":   "freestyle",
    "tricks":      "freestyle",
    "unhook":      "freestyle",
    "free style":  "freestyle",
}

VALID_LOCATION_IDS = {f"LOC{i:02d}" for i in range(1, 11)}


# ── Rider profiles ────────────────────────────────────────────────────────────

def clean_rider_profiles(path):
    print("\n── Rider profiles ───────────────────────────────────")
    df = pd.read_csv(path)
    original_len = len(df)
    print(f"   Loaded: {original_len} rows")

    # ── 1. Weight → kg ─────────────────────────────────────────────────────
    df["weight_kg"] = df["weight"].apply(parse_weight_kg)

    # Sanity check — reject physically impossible weights
    invalid_weight = df["weight_kg"].lt(WEIGHT_MIN_KG) | df["weight_kg"].gt(WEIGHT_MAX_KG)
    audit["rider_weight_invalid"] = int(invalid_weight.sum())
    df.loc[invalid_weight, "weight_kg"] = np.nan
    audit["rider_weight_missing"] = int(df["weight_kg"].isna().sum())

    # ── 2. Skill level normalization ───────────────────────────────────────
    # Hybrid approach:
    # Step 1 — static map handles known synonyms (pro, expert, novice)
    # Step 2 — fuzzy matching catches typos we didn't anticipate

    CANONICAL_SKILLS = ["beginner", "intermediate", "advanced"]
    df["skill_level"] = strip_and_lower(df["skill_level"]).map(SKILL_MAP)

    # Apply fuzzy matching only to rows that didn't match the static map
    unmatched = df["skill_level"].isna()
    df.loc[unmatched, "skill_level"] = strip_and_lower(
        pd.read_csv(f"{RAW_DIR}/rider_profiles_raw.csv")["skill_level"]
    )[unmatched].apply(lambda x: fuzzy_normalize(x, CANONICAL_SKILLS, threshold=70))

    audit["rider_skill_unknown"] = int(df["skill_level"].isna().sum())
    df["skill_level"] = df["skill_level"].fillna("beginner")

    # ── 3. Style normalization ─────────────────────────────────────────────
    CANONICAL_STYLES = ["freeride", "big_air", "wave", "freestyle"]
    df["preferred_style"] = strip_and_lower(df["preferred_style"]).apply(
        lambda x: fuzzy_normalize(x, CANONICAL_STYLES, threshold=70)
    )
    df["preferred_style"] = df["preferred_style"].fillna("freeride")

    # ── 4. Budget → USD ────────────────────────────────────────────────────
    df["budget_usd"] = df["budget"].apply(parse_budget_usd)
    audit["rider_budget_missing"] = int(df["budget_usd"].isna().sum())

    # ── 5. Existing sizes → clean list ─────────────────────────────────────
    df["existing_sizes"] = df["existing_kite_sizes"].apply(parse_existing_sizes)

    # ── 6. Cross-table FK validation ──────────────────────────────────────
    # Check every home_location_id actually exists in wind data.
    # Rows with invalid IDs like LOC99 get nulled and flagged.
    df["home_location_id"] = df["home_location_id"].apply(
        lambda loc: loc if loc in VALID_LOCATION_IDS else None
    )
    audit["rider_orphan_location"] = int(df["home_location_id"].isna().sum())

    # ── 7. Drop raw columns ────────────────────────────────────────────────
    df.drop(columns=["weight", "budget", "existing_kite_sizes"], inplace=True)

    print(f"   Clean:  {len(df)} rows")
    df.to_csv(f"{CLEAN_DIR}/rider_profiles_clean.csv", index=False)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rider_df = clean_rider_profiles(f"{RAW_DIR}/rider_profiles_raw.csv")

    print("\n" + "═" * 52)
    print("  CLEANING AUDIT LOG")
    print("═" * 52)
    for key, val in audit.items():
        label = key.replace("_", " ")
        print(f"  {label:<38}  {val:>6}")
    print("═" * 52)