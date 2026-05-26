"""
app.py
──────
Streamlit app for kite-ideal.

Two pages:
    1. Get My Recommendation — personal quiver recommendation
    2. Compare Kites         — side by side score comparison

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="kite-ideal",
    page_icon="🪁",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    scores  = pd.read_csv("data/clean/scored_combinations.csv")
    kites   = pd.read_csv("data/clean/kite_catalog_clean.csv")
    riders  = pd.read_csv("data/clean/rider_profiles_clean.csv")
    recs    = pd.read_csv("outputs/quiver_recommendations.csv")
    wind    = pd.read_csv("data/clean/wind_observations_clean.csv")
    return scores, kites, riders, recs, wind

scores_df, kites_df, riders_df, recs_df, wind_df = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/kite--ideal-🪁-blue", width=200)
st.sidebar.title("kite-ideal")
st.sidebar.caption("Data-driven kite recommendations")

page = st.sidebar.radio(
    "Navigate",
    ["🏄 Get My Recommendation", "📊 Compare Kites"]
)

# ── Page 1: Get My Recommendation ────────────────────────────────────────────
if page == "🏄 Get My Recommendation":
    st.title("🪁 Find Your Ideal Kite")
    st.caption("Enter your profile and get a data-driven quiver recommendation.")

    col1, col2 = st.columns(2)

    with col1:
        weight = st.slider("Your weight (kg)", 40, 150, 75)
        budget = st.number_input("Budget (USD)", 500, 10000, 3000, step=100)
        skill  = st.selectbox("Skill level", ["beginner", "intermediate", "advanced"])

    with col2:
        style    = st.selectbox("Riding style", ["freeride", "wave", "big_air", "freestyle"])
        location = st.selectbox("Home location", [
            "LOC01 — Cumbuco, Brazil",
            "LOC02 — Jericoacoara, Brazil",
            "LOC03 — Tarifa, Spain",
            "LOC04 — Maui, Hawaii",
            "LOC05 — Cape Town, South Africa",
            "LOC06 — Cabarete, Dominican Rep.",
            "LOC07 — Dakhla, Morocco",
            "LOC08 — Ilha do Guajiru, Brazil",
            "LOC09 — Lake Garda, Italy",
            "LOC10 — Boracay, Philippines",
        ])
        loc_id = location.split(" — ")[0]

    st.divider()

    if st.button("🔍 Get My Recommendation", type="primary"):

        # ── Wind stats for this location ───────────────────────────────────────
        loc_wind_series = wind_df[wind_df["location_id"] == loc_id]["wind_speed_kn"].dropna()
        loc_scores      = scores_df[scores_df["location_id"] == loc_id].copy()

        if loc_wind_series.empty or loc_scores.empty:
            st.error("No wind data available for this location.")
        else:
            avg_wind = loc_wind_series.mean()
            wind_std = loc_wind_series.std()
            cv       = round(wind_std / avg_wind, 2) if avg_wind > 0 else 0.0

            def ideal_kite(w, wind):
                return int(round(max(5.0, min(17.0, (w / wind) * 2.2)))) if wind > 0 else None

            light_wind  = max(avg_wind - wind_std, 1.0)
            strong_wind = avg_wind + wind_std
            size_light  = ideal_kite(weight, light_wind)
            size_avg    = ideal_kite(weight, avg_wind)
            size_strong = ideal_kite(weight, strong_wind)

            size_range = (
                f"{size_light} – {size_strong} m²"
                if size_light and size_strong else "N/A"
            )

            consistency_label = (
                "Consistent — one kite size covers most of your sessions"
                if cv < 0.20 else
                "Variable — consider having a backup kite for strong wind days"
                if cv <= 0.30 else
                "Unpredictable — wind changes a lot here, you'll need kites for both light and strong days"
            )

            # ── TOP SECTION: profile summary bar ──────────────────────────────
            st.markdown("#### Your Profile Summary")
            p1, p2, p3 = st.columns(3)
            p1.metric("Your weight",     f"{weight} kg")
            p2.metric("Ideal kite size", size_range if size_avg else "N/A")
            p2.caption(f"Light {size_light}m · Avg {size_avg}m · Strong {size_strong}m")
            p3.metric("Avg wind",        f"{avg_wind:.1f} kn")
            st.info(consistency_label)

            st.divider()

            # ── MIDDLE SECTION: top 3 kites ────────────────────────────────────
            # Pick kites with meaningfully different sizes (≥ 2m² apart each).
            # Iterate ranked rows and greedily add a kite only if it is at
            # least 2m² away from every already-selected kite.
            ranked = (
                loc_scores
                .dropna(subset=["overall_score"])
                .sort_values("overall_score", ascending=False)
                .drop_duplicates(subset=["kite_id"])
            )
            selected = []
            for _, row in ranked.iterrows():
                if all(abs(row["size_m2"] - s["size_m2"]) >= 2 for s in selected):
                    selected.append(row)
                if len(selected) == 3:
                    break
            top3 = pd.DataFrame(selected).reset_index(drop=True)

            if top3.empty:
                st.warning("No scored kites found for this location.")
            else:
                medals = ["🥇 Best Pick", "🥈 Runner Up", "🥉 Third Option"]
                card_cols = st.columns(3)

                for col, (_, kite), medal in zip(card_cols, top3.iterrows(), medals):
                    with col:
                        st.markdown(f"**{medal}**")
                        st.markdown(
                            f"### {kite['brand']} {kite['model']} {kite['size_m2']}m²"
                        )
                        price = kite.get("price_usd")
                        if pd.notna(price):
                            st.markdown(f"**${price:,.0f}**")
                        else:
                            st.markdown("Price N/A")

                        def pct(val):
                            return f"{val:.0%}" if pd.notna(val) else "N/A"

                        st.metric("Wind coverage", pct(kite["coverage_score"]))
                        st.metric("Skill match",   pct(kite["skill_score"]))
                        st.metric("Weight match",  pct(kite["weight_score"]))
                        st.metric("Gust safety",   pct(kite["gust_score"]))
                        st.metric("Overall score", pct(kite["overall_score"]))

            st.divider()

            # ── BOTTOM SECTION: location context panel ─────────────────────────
            st.markdown("#### 📍 Location Context")

            MONTH_NAMES = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
            }
            loc_wind_df   = wind_df[wind_df["location_id"] == loc_id].copy()
            monthly_avg   = loc_wind_df.groupby("month")["wind_speed_kn"].mean()
            best_months   = ", ".join(
                MONTH_NAMES[m] for m in monthly_avg.nlargest(3).index
            )

            cv_label = "Low" if cv < 0.20 else "Medium" if cv <= 0.30 else "High"
            cv_descriptions = {
                "Low":    "Very consistent — one kite covers most sessions",
                "Medium": "Moderate variability — two sizes recommended",
                "High":   "High variability — wider quiver recommended",
            }
            cv_description = cv_descriptions[cv_label]

            WIND_DIRECTION_DESCRIPTIONS = {
                "LOC01": "SE trade winds — side-onshore, safe most days",
                "LOC02": "SE trade winds — side-onshore, safe most days",
                "LOC03": "Poniente (W/SW) safe — avoid Levante (E) days",
                "LOC04": "NE trade winds — side-onshore, safe most days",
                "LOC05": "SE Cape Doctor — side-onshore, but highly variable",
                "LOC06": "NE trade winds — side-onshore, safe most days",
                "LOC07": "N/NE trade winds — lagoon protects, very safe",
                "LOC08": "SE trade winds — side-onshore, safe most days",
                "LOC09": "Thermal winds — N morning, S afternoon, both safe",
                "LOC10": "NE or SW seasonal — both side-onshore, safe",
            }
            dir_description = WIND_DIRECTION_DESCRIPTIONS.get(loc_id, "No description available")

            l1, l2, l3 = st.columns(3)
            l1.metric("Avg wind speed",      f"{avg_wind:.1f} kn")
            with l2:
                st.markdown("**Wind variability**")
                st.markdown(cv_description)
            l3.metric("Best months to kite", best_months)
            st.markdown(f"**Wind direction:** {dir_description}")


# ── Page 2: Compare Kites ─────────────────────────────────────────────────────
elif page == "📊 Compare Kites":
    st.title("📊 Compare Kites")
    st.caption("Select two kites and a location to see a side-by-side score comparison.")

    # Build kite label list
    kites_df["label"] = kites_df["brand"] + " " + kites_df["model"] + " " + kites_df["size_m2"].astype(str) + "m²"
    kite_labels = sorted(kites_df["label"].unique())

    col1, col2, col3 = st.columns(3)

    with col1:
        kite1_label = st.selectbox("Kite 1", kite_labels, index=0)
    with col2:
        kite2_label = st.selectbox("Kite 2", kite_labels, index=10)
    with col3:
        location = st.selectbox("Location", [
            "LOC01 — Cumbuco, Brazil",
            "LOC02 — Jericoacoara, Brazil",
            "LOC03 — Tarifa, Spain",
            "LOC04 — Maui, Hawaii",
            "LOC05 — Cape Town, South Africa",
            "LOC06 — Cabarete, Dominican Rep.",
            "LOC07 — Dakhla, Morocco",
            "LOC08 — Ilha do Guajiru, Brazil",
            "LOC09 — Lake Garda, Italy",
            "LOC10 — Boracay, Philippines",
        ])
        loc_id = location.split(" — ")[0]

    st.divider()

    if st.button("📊 Compare", type="primary"):
        # Get kite IDs
        kite1_id = kites_df[kites_df["label"] == kite1_label]["kite_id"].iloc[0]
        kite2_id = kites_df[kites_df["label"] == kite2_label]["kite_id"].iloc[0]

        # Get scores for these kites at this location
        s1 = scores_df[
            (scores_df["kite_id"] == kite1_id) &
            (scores_df["location_id"] == loc_id)
        ]
        s2 = scores_df[
            (scores_df["kite_id"] == kite2_id) &
            (scores_df["location_id"] == loc_id)
        ]

        if s1.empty or s2.empty:
            st.error("Score data not available for this combination.")
        else:
            s1 = s1.iloc[0]
            s2 = s2.iloc[0]

            categories = ["Coverage", "Skill", "Weight", "Gust", "Overall"]
            scores1 = [s1["coverage_score"], s1["skill_score"],
                       s1["weight_score"], s1["gust_score"], s1["overall_score"]]
            scores2 = [s2["coverage_score"], s2["skill_score"],
                       s2["weight_score"], s2["gust_score"], s2["overall_score"]]

            # Metrics side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(kite1_label)
                for cat, score in zip(categories, scores1):
                    if pd.notna(score):
                        st.metric(cat, f"{score:.0%}")
                    else:
                        st.metric(cat, "N/A")

            with col2:
                st.subheader(kite2_label)
                for cat, score in zip(categories, scores2):
                    if pd.notna(score):
                        st.metric(cat, f"{score:.0%}")
                    else:
                        st.metric(cat, "N/A")

            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(10, 4))
            x = range(len(categories))
            width = 0.35

            ax.bar([i - width/2 for i in x],
                   [s if pd.notna(s) else 0 for s in scores1],
                   width, label=kite1_label, color="#2196F3")
            ax.bar([i + width/2 for i in x],
                   [s if pd.notna(s) else 0 for s in scores2],
                   width, label=kite2_label, color="#FF9800")

            ax.set_xticks(list(x))
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close()