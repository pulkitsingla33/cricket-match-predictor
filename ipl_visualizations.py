"""
IPL Data Visualization
=======================
Reads matches.csv + deliveries.csv and produces a suite of
publication-ready plots saved as PNGs in --output_dir.

Usage:
    python ipl_visualizations.py \
        --matches matches.csv \
        --deliveries deliveries.csv \
        --output_dir ./plots
"""

import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# ─────── Colour palette ───────────────────────────────
C_MAIN   = "#E8533F"  # IPL red-orange
C_ACC    = "#1A2E4A"  # deep navy
C_GOLD   = "#F4C542"
C_GREEN  = "#3BAD6B"
C_GREY   = "#8A9BB0"
TEAM_PALETTE = [
    "#E8533F","#1A2E4A","#F4C542","#3BAD6B","#9B59B6",
    "#E67E22","#2980B9","#16A085","#8E44AD","#C0392B",
    "#27AE60","#D35400","#2C3E50","#F39C12","#1ABC9C",
]


def load_data(matches_path, deliveries_path):
    matches    = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    for col in ["innings1_runs","innings2_runs","innings1_wickets","innings2_wickets",
                "win_by_runs","win_by_wickets"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")
    matches["season_year"] = matches["season"].str.extract(r"(\d{4})").astype(float)
    print(f"Loaded {len(matches)} matches, {len(deliveries)} deliveries.")
    return matches, deliveries


def save(fig, path, title=""):
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════
# PLOT 1 — Matches per Season
# ═══════════════════════════════════════════════════════
def plot_matches_per_season(matches, out):
    season_counts = (matches.groupby("season_year").size()
                             .reset_index(name="matches")
                             .sort_values("season_year"))

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.bar(season_counts["season_year"].astype(int),
                  season_counts["matches"],
                  color=C_MAIN, edgecolor="white", linewidth=0.8, width=0.7)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=8, color=C_ACC, fontweight="bold")

    ax.set_title("IPL Matches Played per Season", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of Matches")
    ax.set_xticks(season_counts["season_year"].astype(int))
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    save(fig, out / "01_matches_per_season.png")


# ═══════════════════════════════════════════════════════
# PLOT 2 — Win distribution: Toss & Batting decisions
# ═══════════════════════════════════════════════════════
def plot_toss_analysis(matches, out):
    valid = matches.dropna(subset=["winner","toss_winner"]).copy()
    valid["toss_winner_won"] = (valid["toss_winner"] == valid["winner"]).astype(int)

    # Toss win → match win rate by decision
    decision_win = (valid.groupby("toss_decision")["toss_winner_won"]
                         .agg(["mean","count"])
                         .reset_index())
    decision_win.columns = ["decision","win_rate","count"]
    decision_win["win_pct"] = decision_win["win_rate"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#F8F9FA")

    # Left: toss win → match win
    overall_toss_win = valid["toss_winner_won"].mean() * 100
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")
    colors = [C_GREEN if r > 50 else C_GREY for r in decision_win["win_pct"]]
    bars = ax.bar(decision_win["decision"].str.capitalize(),
                  decision_win["win_pct"], color=colors,
                  edgecolor="white", linewidth=0.8, width=0.5)
    ax.axhline(50, linestyle="--", color=C_MAIN, linewidth=1.2, label="50% baseline")
    for bar, (_, row) in zip(bars, decision_win.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{row['win_pct']:.1f}%\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=9, color=C_ACC)
    ax.set_title("Toss Winner Match Win Rate\nby Toss Decision", fontweight="bold", color=C_ACC)
    ax.set_ylabel("Match Win %")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Right: stacked outcomes
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    overall = [overall_toss_win, 100 - overall_toss_win]
    labels  = ["Toss winner won", "Toss winner lost"]
    wedge_props = {"edgecolor":"white","linewidth":2}
    wedges, _, autotexts = ax2.pie(
        overall, labels=labels, autopct="%1.1f%%",
        colors=[C_GREEN, C_GREY], startangle=90,
        wedgeprops=wedge_props, textprops={"fontsize":10}
    )
    ax2.set_title(f"Overall Toss-Win Effect\n(n={len(valid)} matches)", fontweight="bold", color=C_ACC)

    plt.tight_layout()
    save(fig, out / "02_toss_analysis.png")


# ═══════════════════════════════════════════════════════
# PLOT 3 — Top Run-Scorers (career aggregates)
# ═══════════════════════════════════════════════════════
def plot_top_batters(deliveries, out, top_n=15):
    legal = deliveries[deliveries["wides"] == 0].copy()
    batter_agg = legal.groupby("batter").agg(
        runs        =("batter_runs", "sum"),
        balls       =("batter_runs", "count"),
        fours       =("batter_runs", lambda x: (x == 4).sum()),
        sixes       =("batter_runs", lambda x: (x == 6).sum()),
    ).reset_index()
    batter_agg["strike_rate"] = (batter_agg["runs"] / batter_agg["balls"] * 100).round(1)
    top = batter_agg.nlargest(top_n, "runs")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    y_pos = range(len(top))
    colors_bar = [TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(top))]
    bars = ax.barh(list(y_pos), top["runs"], color=colors_bar,
                   edgecolor="white", linewidth=0.8, height=0.75)

    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f"{int(row['runs'])} runs  |  SR {row['strike_rate']}",
                va="center", fontsize=8, color=C_ACC)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top["batter"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Run Scorers (all IPL matches)", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Total Runs")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(200))
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, top["runs"].max() * 1.18)

    plt.tight_layout()
    save(fig, out / "03_top_run_scorers.png")


# ═══════════════════════════════════════════════════════
# PLOT 4 — Top Wicket Takers
# ═══════════════════════════════════════════════════════
def plot_top_bowlers(deliveries, out, top_n=15):
    bowler_agg = deliveries.groupby("bowler").agg(
        wickets       =("is_wicket",   "sum"),
        balls_bowled  =("total_runs",  "count"),
        runs_conceded =("total_runs",  "sum"),
    ).reset_index()
    bowler_agg["economy"] = (bowler_agg["runs_conceded"] /
                              (bowler_agg["balls_bowled"] / 6)).round(2)
    top = bowler_agg.nlargest(top_n, "wickets")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    y_pos   = range(len(top))
    colors_bar = [TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(top))]
    bars = ax.barh(list(y_pos), top["wickets"], color=colors_bar,
                   edgecolor="white", height=0.75)

    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{int(row['wickets'])} wkts  |  Eco {row['economy']}",
                va="center", fontsize=8, color=C_ACC)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top["bowler"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Wicket Takers", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Total Wickets")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, top["wickets"].max() * 1.18)

    plt.tight_layout()
    save(fig, out / "04_top_wicket_takers.png")


# ═══════════════════════════════════════════════════════
# PLOT 5 — Run Distribution by Over (scoring pattern)
# ═══════════════════════════════════════════════════════
def plot_runs_by_over(deliveries, out):
    inn1 = deliveries[deliveries["innings"] == 1]
    inn2 = deliveries[deliveries["innings"] == 2]

    def over_agg(df):
        return (df.groupby("over")["total_runs"]
                  .mean()
                  .reset_index(name="avg_runs"))

    o1 = over_agg(inn1)
    o2 = over_agg(inn2)
    overs = range(1, 21)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    ax.plot(o1["over"], o1["avg_runs"], color=C_MAIN,  linewidth=2.5,
            marker="o", markersize=5, label="Innings 1")
    ax.plot(o2["over"], o2["avg_runs"], color=C_ACC,   linewidth=2.5,
            marker="s", markersize=5, label="Innings 2")
    ax.fill_between(o1["over"], o1["avg_runs"], alpha=0.12, color=C_MAIN)
    ax.fill_between(o2["over"], o2["avg_runs"], alpha=0.12, color=C_ACC)

    # Phase bands
    ax.axvspan(1,  6,  alpha=0.06, color=C_GREEN, label="Powerplay (1-6)")
    ax.axvspan(7,  15, alpha=0.04, color=C_GREY,  label="Middle (7-15)")
    ax.axvspan(16, 20, alpha=0.06, color=C_GOLD,  label="Death (16-20)")

    ax.set_title("Average Runs per Over — Innings 1 vs Innings 2",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Over Number")
    ax.set_ylabel("Avg Runs")
    ax.set_xticks(list(overs))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    save(fig, out / "05_runs_by_over.png")


# ═══════════════════════════════════════════════════════
# PLOT 6 — Wickets by Over (when do teams fall?)
# ═══════════════════════════════════════════════════════
def plot_wickets_by_over(deliveries, out):
    wkts = deliveries[deliveries["is_wicket"] == 1]

    inn1 = wkts[wkts["innings"] == 1].groupby("over").size().reset_index(name="wickets")
    inn2 = wkts[wkts["innings"] == 2].groupby("over").size().reset_index(name="wickets")

    # Normalize by number of matches
    n_matches = deliveries["match_id"].nunique()
    inn1["wkts_per_match"] = inn1["wickets"] / n_matches
    inn2["wkts_per_match"] = inn2["wickets"] / n_matches

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    x = np.arange(1, 21)
    width = 0.4

    def get_wkts(df):
        full = pd.DataFrame({"over": range(1,21)})
        merged = full.merge(df[["over","wkts_per_match"]], on="over", how="left").fillna(0)
        return merged["wkts_per_match"].values

    ax.bar(x - width/2, get_wkts(inn1), width=width, color=C_MAIN,
           label="Innings 1", edgecolor="white")
    ax.bar(x + width/2, get_wkts(inn2), width=width, color=C_ACC,
           label="Innings 2", edgecolor="white")

    ax.axvspan(0.5, 6.5,  alpha=0.05, color=C_GREEN)
    ax.axvspan(15.5, 20.5,alpha=0.06, color=C_GOLD)

    ax.set_title("Average Wickets per Over per Match",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Over Number")
    ax.set_ylabel("Avg Wickets")
    ax.set_xticks(list(x))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    save(fig, out / "06_wickets_by_over.png")


# ═══════════════════════════════════════════════════════
# PLOT 7 — Score Distribution (histogram)
# ═══════════════════════════════════════════════════════
def plot_score_distribution(matches, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.patch.set_facecolor("#F8F9FA")

    for ax, col, label, color in zip(
        axes,
        ["innings1_runs", "innings2_runs"],
        ["Innings 1 Final Score", "Innings 2 Final Score"],
        [C_MAIN, C_ACC]
    ):
        data = matches[col].dropna()
        ax.set_facecolor("#F8F9FA")
        ax.hist(data, bins=25, color=color, edgecolor="white", linewidth=0.6, alpha=0.9)
        ax.axvline(data.mean(), color=C_GOLD, linewidth=2,
                   linestyle="--", label=f"Mean: {data.mean():.0f}")
        ax.axvline(data.median(), color=C_GREY, linewidth=2,
                   linestyle=":", label=f"Median: {data.median():.0f}")
        ax.set_title(label, fontweight="bold", color=C_ACC)
        ax.set_xlabel("Total Runs")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.suptitle("Distribution of Team Scores", fontweight="bold", fontsize=14,
                 color=C_ACC, y=1.02)
    plt.tight_layout()
    save(fig, out / "07_score_distribution.png")


# ═══════════════════════════════════════════════════════
# PLOT 8 — Team Win Counts
# ═══════════════════════════════════════════════════════
def plot_team_wins(matches, out):
    valid = matches.dropna(subset=["winner"])
    # exclude "no result" / "tie"
    teams = matches[["team1","team2"]].values.flatten()
    known_teams = set(teams)
    win_counts = (valid[valid["winner"].isin(known_teams)]
                  .groupby("winner").size()
                  .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(12, max(5, len(win_counts) * 0.45)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    colors_bar = [TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(win_counts))]
    bars = ax.barh(win_counts.index, win_counts.values,
                   color=colors_bar, edgecolor="white", height=0.7)

    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(int(bar.get_width())), va="center", fontsize=8, color=C_ACC)

    ax.set_title("Total Wins by Team", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Wins")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, win_counts.max() * 1.12)

    plt.tight_layout()
    save(fig, out / "08_team_wins.png")


# ═══════════════════════════════════════════════════════
# PLOT 9 — Dismissal Types
# ═══════════════════════════════════════════════════════
def plot_dismissal_types(deliveries, out):
    wkts = deliveries[deliveries["is_wicket"] == 1].dropna(subset=["wicket_kind"])
    counts = wkts["wicket_kind"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    wedge_colors = [TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(counts))]
    wedge_props  = {"edgecolor":"white","linewidth":2}
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct=lambda p: f"{p:.1f}%\n({int(p/100*sum(counts.values))})",
        colors=wedge_colors,
        startangle=140,
        wedgeprops=wedge_props,
        textprops={"fontsize": 9},
        pctdistance=0.78,
    )
    ax.set_title("Wicket Dismissal Types", fontweight="bold", color=C_ACC, fontsize=13, pad=15)

    plt.tight_layout()
    save(fig, out / "09_dismissal_types.png")


# ═══════════════════════════════════════════════════════
# PLOT 10 — Powerplay vs Total Score scatter
# ═══════════════════════════════════════════════════════
def plot_powerplay_vs_total(matches, deliveries, out):
    pp = (deliveries[(deliveries["in_powerplay"] == 1) & (deliveries["innings"] == 1)]
          .groupby("match_id")["total_runs"].sum()
          .reset_index(name="pp_runs"))

    merged = matches[["match_id","innings1_runs","season_year"]].merge(pp, on="match_id", how="inner")
    merged = merged.dropna(subset=["innings1_runs","pp_runs"])

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    seasons = sorted(merged["season_year"].dropna().unique())
    cmap    = plt.cm.get_cmap("tab20", len(seasons))

    for i, season in enumerate(seasons):
        sub = merged[merged["season_year"] == season]
        ax.scatter(sub["pp_runs"], sub["innings1_runs"],
                   color=cmap(i), label=str(int(season)),
                   alpha=0.7, s=40, edgecolors="white", linewidths=0.4)

    # trend line
    m, b = np.polyfit(merged["pp_runs"], merged["innings1_runs"], 1)
    x_line = np.linspace(merged["pp_runs"].min(), merged["pp_runs"].max(), 100)
    ax.plot(x_line, m * x_line + b, color=C_MAIN, linewidth=2, linestyle="--",
            label=f"Trend (slope={m:.2f})")

    ax.set_title("Powerplay Score vs Final Innings 1 Score",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Powerplay Runs (Overs 1–6)")
    ax.set_ylabel("Innings 1 Total Runs")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    save(fig, out / "10_powerplay_vs_total.png")


# ═══════════════════════════════════════════════════════
# PLOT 11 — Correlation heatmap of match features
# ═══════════════════════════════════════════════════════
def plot_correlation_heatmap(matches, deliveries, out):
    try:
        import matplotlib.colors as mcolors

        pp_inn1 = (deliveries[(deliveries["in_powerplay"] == 1) & (deliveries["innings"] == 1)]
                   .groupby("match_id")["total_runs"].sum().reset_index(name="pp_runs_inn1"))
        death_inn1 = (deliveries[(deliveries["over"] >= 17) & (deliveries["innings"] == 1)]
                      .groupby("match_id")["total_runs"].sum().reset_index(name="death_runs_inn1"))

        df = matches[["match_id","innings1_runs","innings2_runs","innings1_wickets",
                       "innings2_wickets","win_by_runs","win_by_wickets"]].copy()
        df = df.merge(pp_inn1, on="match_id", how="left")
        df = df.merge(death_inn1, on="match_id", how="left")

        df["toss_bat_first"] = (matches["toss_decision"] == "bat").astype(int)
        df["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)

        corr_cols = ["innings1_runs","innings2_runs","innings1_wickets","innings2_wickets",
                     "pp_runs_inn1","death_runs_inn1","toss_bat_first","team1_won"]
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr = df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#F8F9FA")
        ax.set_facecolor("#F8F9FA")

        cmap = plt.cm.RdBu_r
        im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

        labels = [c.replace("_"," ").title() for c in corr_cols]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(val) > 0.5 else C_ACC)

        ax.set_title("Feature Correlation Matrix", fontweight="bold", color=C_ACC, pad=12)
        plt.tight_layout()
        save(fig, out / "11_correlation_heatmap.png")
    except Exception as e:
        print(f"  Skipped correlation heatmap: {e}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IPL Visualizations")
    parser.add_argument("--matches",    required=True)
    parser.add_argument("--deliveries", required=True)
    parser.add_argument("--output_dir", default="./plots")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    matches, deliveries = load_data(args.matches, args.deliveries)

    print(f"\nGenerating plots → {out}/\n")
    plot_matches_per_season(matches, out)
    plot_toss_analysis(matches, out)
    plot_top_batters(deliveries, out)
    plot_top_bowlers(deliveries, out)
    plot_runs_by_over(deliveries, out)
    plot_wickets_by_over(deliveries, out)
    plot_score_distribution(matches, out)
    plot_team_wins(matches, out)
    plot_dismissal_types(deliveries, out)
    plot_powerplay_vs_total(matches, deliveries, out)
    plot_correlation_heatmap(matches, deliveries, out)

    print(f"\nAll plots saved to: {out}")


if __name__ == "__main__":
    main()