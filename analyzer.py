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
        if abs(k1.size_m2 - k2.size_m2) < 2:
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


def save_chart(rider_name, kite1, kite2=None):
    """Save a bar chart for one or two kites.

    Single-kite path (kite2=None): one bar group, no combined column.
    Two-kite path: three bar groups with a Quiver Combined column.
    """
    import math

    def _v(score):
        return 0.0 if score is None or (isinstance(score, float) and math.isnan(score)) else score

    categories = ["Coverage", "Skill", "Weight", "Gust", "Overall"]
    scores1    = [_v(kite1.coverage_score), _v(kite1.skill_score), _v(kite1.weight_score),
                  _v(kite1.gust_score), _v(kite1.overall_score)]

    x   = range(len(categories))
    fig, ax = plt.subplots(figsize=(11, 5))

    if kite2 is None:
        width = 0.5
        ax.bar(list(x), scores1, width,
               label=f"{kite1.brand} {kite1.model} {kite1.size_m2}m²", color="#2196F3")
        ax.set_title(f"Single-Kite Recommendation — {rider_name}", fontsize=11)
        ncol = 1
    else:
        scores2 = [_v(kite2.coverage_score), _v(kite2.skill_score), _v(kite2.weight_score),
                   _v(kite2.gust_score), _v(kite2.overall_score)]

        def _combined(s1, s2, additive):
            return 1 - (1 - s1) * (1 - s2) if additive else max(s1, s2)

        scores_combined = [
            _combined(scores1[0], scores2[0], True),   # coverage
            _combined(scores1[1], scores2[1], True),   # skill
            _combined(scores1[2], scores2[2], False),  # weight
            _combined(scores1[3], scores2[3], False),  # gust
            _combined(scores1[4], scores2[4], False),  # overall
        ]

        width = 0.25
        ax.bar([i - width for i in x], scores1,         width,
               label=f"{kite1.brand} {kite1.size_m2}m²", color="#2196F3")
        ax.bar([i         for i in x], scores2,         width,
               label=f"{kite2.brand} {kite2.size_m2}m²", color="#FF9800")
        ax.bar([i + width for i in x], scores_combined, width,
               label="Quiver Combined", color="#4CAF50")
        ax.set_title(
            f"Quiver Recommendation — {rider_name}\n"
            "Combined = what the pair covers together",
            fontsize=11,
        )
        ncol = 3

    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncol)
    ax.grid(axis="y", alpha=0.3)

    filename = rider_name.replace(" ", "_") + ".png"
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/{filename}")
    plt.close()


def find_best_single(rider_df):
    """Return the single kite with the highest overall_score."""
    kites = rider_df.dropna(subset=["overall_score"])
    if kites.empty:
        return None
    return kites.loc[kites["overall_score"].idxmax()]


def run_analyzer():
    df = load_scores()

    results  = []
    riders   = df["rider_id"].unique()

    print(f"\n── Analyzing {len(riders)} riders ───────────────────────")

    for rider_id in riders:
        rider_df   = df[df["rider_id"] == rider_id].copy()
        rider_name = rider_df["rider_name"].iloc[0]

        best_single = find_best_single(rider_df)

        if best_single is None:
            print(f"  [!] No valid kite found for {rider_name}")
            continue

        if best_single["coverage_score"] >= 0.80:
            kite = best_single
            print(
                f"\n  One kite is enough — "
                f"{kite['brand']} {kite['model']} {kite['size_m2']}m² "
                f"covers {kite['coverage_score']:.0%} of wind days at this location"
            )
            save_chart(rider_name, kite)
            results.append({
                "rider_id":    rider_id,
                "rider_name":  rider_name,
                "quiver_size": 1,
                "kite1_brand": kite["brand"],
                "kite1_model": kite["model"],
                "kite1_size":  kite["size_m2"],
                "kite1_price": kite["price_usd"],
                "kite2_brand": None,
                "kite2_model": None,
                "kite2_size":  None,
                "kite2_price": None,
                "combined_score": round(kite["overall_score"], 3),
            })
            continue

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
            "quiver_size": 2,
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