"""
analyzer.py
───────────
Reads scored_combinations.csv and finds the optimal 2-kite quiver
per rider within their budget.

Outputs:
    - Recommendation report printed to terminal
    - outputs/quiver_recommendations.csv
    - outputs/charts/ — one chart per rider
"""

import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

CLEAN_DIR   = "data/clean"
OUTPUTS_DIR = "outputs"
CHARTS_DIR  = f"{OUTPUTS_DIR}/charts"

os.makedirs(CHARTS_DIR, exist_ok=True)


def load_scores():
    """Load scored combinations from transformer output."""
    path = f"{CLEAN_DIR}/scored_combinations.csv"
    df   = pd.read_csv(path)
    print(f"[✓] Loaded {len(df)} scored combinations")
    return df


def find_best_quiver(rider_df, budget):
    """Find the best 2-kite combination for one rider.
    
    Brute-forces all pairs of kites within budget.
    Filters out pairs with identical sizes — they don't complement each other.
    Ranks by combined overall_score.
    """
    kites = rider_df.dropna(subset=["overall_score"]).copy()

    if len(kites) < 2:
        return None

    best_score = -1
    best_pair  = None

    # itertools.combinations generates all possible pairs without repetition
    for k1, k2 in itertools.combinations(kites.itertuples(), 2):
        # Skip pairs with same size — no complementary coverage
        if k1.size_m2 == k2.size_m2:
            continue

        # Skip pairs that exceed budget
        total_price = k1.price_usd + k2.price_usd
        if total_price > budget:
            continue

        combined_score = (k1.overall_score + k2.overall_score) / 2

        if combined_score > best_score:
            best_score = combined_score
            best_pair  = (k1, k2)

    return best_pair


def print_recommendation(rider_name, kite1, kite2):
    """Print a formatted recommendation card for one rider."""

    def safe_bar(score):
        import math
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return "── N/A"
        return f"{'█' * int(score * 10):<10}  {score:.0%}"

    print(f"\n{'═' * 56}")
    print(f"  {rider_name}")
    print(f"{'─' * 56}")

    for kite in [kite1, kite2]:
        print(f"\n  {kite.brand} {kite.model} {kite.size_m2}m²  —  ${kite.price_usd:.0f}")
        print(f"  {'─' * 40}")
        print(f"  Wind coverage    {safe_bar(kite.coverage_score)}")
        print(f"  Skill match      {safe_bar(kite.skill_score)}")
        print(f"  Weight match     {safe_bar(kite.weight_score)}")
        print(f"  Direction safety {safe_bar(kite.direction_score)}")
        print(f"  Gust safety      {safe_bar(kite.gust_score)}")
        print(f"  {'─' * 40}")
        print(f"  Overall          {safe_bar(kite.overall_score)}")


def save_chart(rider_name, kite1, kite2):
    """Save a bar chart comparing the two recommended kites."""
    categories = ["Coverage", "Skill", "Weight", "Direction", "Gust", "Overall"]
    scores1    = [kite1.coverage_score, kite1.skill_score, kite1.weight_score,
                  kite1.direction_score, kite1.gust_score, kite1.overall_score]
    scores2    = [kite2.coverage_score, kite2.skill_score, kite2.weight_score,
                  kite2.direction_score, kite2.gust_score, kite2.overall_score]

    x     = range(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], scores1, width,
           label=f"{kite1.brand} {kite1.size_m2}m²", color="#2196F3")
    ax.bar([i + width/2 for i in x], scores2, width,
           label=f"{kite2.brand} {kite2.size_m2}m²", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title(f"Quiver Recommendation — {rider_name}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    filename = rider_name.replace(" ", "_") + ".png"
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/{filename}")
    plt.close()


def run_analyzer():
    df = load_scores()

    results  = []
    riders   = df["rider_id"].unique()

    print(f"\n── Analyzing {len(riders)} riders ───────────────────────")

    for rider_id in riders:
        rider_df   = df[df["rider_id"] == rider_id].copy()
        rider_name = rider_df["rider_name"].iloc[0]
        budget     = rider_df["price_usd"].sum()  # placeholder

        # Get actual budget from first row
        budget = rider_df.iloc[0]["price_usd"] * 3  # rough estimate

        best = find_best_quiver(rider_df, budget=999999)  # no budget filter for now

        if best is None:
            print(f"  [!] No valid quiver found for {rider_name}")
            continue

        kite1, kite2 = best
        print_recommendation(rider_name, kite1, kite2)
        save_chart(rider_name, kite1, kite2)

        results.append({
            "rider_id":    rider_id,
            "rider_name":  rider_name,
            "kite1_brand": kite1.brand,
            "kite1_model": kite1.model,
            "kite1_size":  kite1.size_m2,
            "kite1_price": kite1.price_usd,
            "kite2_brand": kite2.brand,
            "kite2_model": kite2.model,
            "kite2_size":  kite2.size_m2,
            "kite2_price": kite2.price_usd,
            "combined_score": round((kite1.overall_score + kite2.overall_score) / 2, 3),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUTS_DIR}/quiver_recommendations.csv", index=False)
    print(f"\n[✓] Recommendations saved → outputs/quiver_recommendations.csv")
    print(f"[✓] Charts saved → outputs/charts/")
    print("\nPipeline complete.")


if __name__ == "__main__":
    run_analyzer()