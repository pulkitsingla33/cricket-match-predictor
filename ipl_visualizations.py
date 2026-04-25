"""
IPL Data Visualization — Extended
===================================
Produces 19 publication-ready plots split into two categories:

  EXISTING (Plots 1–11):
    01  Matches per season
    02  Toss analysis
    03  Top run scorers
    04  Top wicket takers
    05  Runs by over (innings comparison)
    06  Wickets by over
    07  Score distribution
    08  Team wins
    09  Dismissal types
    10  Powerplay vs total score scatter
    11  Correlation heatmap

  NEW — Cricket Pattern Analysis (Plots 12–19):
    12  Chase success rate by target range
    13  Run rate progression heatmap (over × wickets-lost pressure map)
    14  Bowling economy by phase (powerplay / middle / death)
    15  Average partnership runs by wicket number
    16  Extras leakage per team (bowling side)
    17  Score acceleration curve (rolling run rate, innings 1 vs 2)
    18  Batter scoring zone breakdown by over phase
    19  Win margin distribution (runs vs wickets)

Usage:
    python ipl_visualizations.py \\
        --matches   matches.csv \\
        --deliveries deliveries.csv \\
        --output_dir ./plots
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
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

# ── Palette ───────────────────────────────────────────
C_MAIN  = "#E8533F"
C_ACC   = "#1A2E4A"
C_GOLD  = "#F4C542"
C_GREEN = "#3BAD6B"
C_GREY  = "#8A9BB0"
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


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path.name}")


def bg(fig, *axes):
    """Apply consistent background colour."""
    fig.patch.set_facecolor("#F8F9FA")
    for ax in axes:
        ax.set_facecolor("#F8F9FA")


# ═══════════════════════════════════════════════════════════════════
# ── EXISTING PLOTS (1–11) ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

def plot_matches_per_season(matches, out):
    sc = matches.groupby("season_year").size().reset_index(name="matches").sort_values("season_year")
    fig, ax = plt.subplots(figsize=(12, 5)); bg(fig, ax)
    bars = ax.bar(sc["season_year"].astype(int), sc["matches"],
                  color=C_MAIN, edgecolor="white", linewidth=0.8, width=0.7)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
                str(int(b.get_height())), ha="center", va="bottom", fontsize=8, color=C_ACC, fontweight="bold")
    ax.set_title("IPL Matches Played per Season", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Season"); ax.set_ylabel("Matches")
    ax.set_xticks(sc["season_year"].astype(int)); ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(); save(fig, out / "01_matches_per_season.png")


def plot_toss_analysis(matches, out):
    valid = matches.dropna(subset=["winner","toss_winner"]).copy()
    valid["toss_winner_won"] = (valid["toss_winner"] == valid["winner"]).astype(int)
    dw = valid.groupby("toss_decision")["toss_winner_won"].agg(["mean","count"]).reset_index()
    dw.columns = ["decision","win_rate","count"]; dw["win_pct"] = dw["win_rate"]*100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); bg(fig, axes[0], axes[1])
    colors = [C_GREEN if r > 50 else C_GREY for r in dw["win_pct"]]
    bars = axes[0].bar(dw["decision"].str.capitalize(), dw["win_pct"],
                       color=colors, edgecolor="white", width=0.5)
    axes[0].axhline(50, linestyle="--", color=C_MAIN, linewidth=1.2, label="50% baseline")
    for b, (_, r) in zip(bars, dw.iterrows()):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f"{r['win_pct']:.1f}%\n(n={int(r['count'])})",
                     ha="center", va="bottom", fontsize=9, color=C_ACC)
    axes[0].set_title("Toss Winner Match Win Rate\nby Decision", fontweight="bold", color=C_ACC)
    axes[0].set_ylabel("Match Win %"); axes[0].set_ylim(0, 75)
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    overall = valid["toss_winner_won"].mean()*100
    axes[1].pie([overall, 100-overall], labels=["Toss winner won","Toss winner lost"],
                autopct="%1.1f%%", colors=[C_GREEN, C_GREY], startangle=90,
                wedgeprops={"edgecolor":"white","linewidth":2}, textprops={"fontsize":10})
    axes[1].set_title(f"Overall Toss-Win Effect\n(n={len(valid)} matches)", fontweight="bold", color=C_ACC)
    plt.tight_layout(); save(fig, out / "02_toss_analysis.png")


def plot_top_batters(deliveries, out, top_n=15):
    legal = deliveries[deliveries["wides"]==0].copy()
    agg = legal.groupby("batter").agg(
        runs=("batter_runs","sum"), balls=("batter_runs","count"),
        fours=("batter_runs", lambda x: (x==4).sum()),
        sixes=("batter_runs", lambda x: (x==6).sum()),
    ).reset_index()
    agg["strike_rate"] = (agg["runs"]/agg["balls"]*100).round(1)
    top = agg.nlargest(top_n, "runs")
    fig, ax = plt.subplots(figsize=(12, 6)); bg(fig, ax)
    y_pos = range(len(top))
    bars = ax.barh(list(y_pos), top["runs"],
                   color=[TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(top))],
                   edgecolor="white", height=0.75)
    for b, (_, r) in zip(bars, top.iterrows()):
        ax.text(b.get_width()+20, b.get_y()+b.get_height()/2,
                f"{int(r['runs'])} runs  |  SR {r['strike_rate']}", va="center", fontsize=8, color=C_ACC)
    ax.set_yticks(list(y_pos)); ax.set_yticklabels(top["batter"], fontsize=9); ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Run Scorers", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Total Runs"); ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, top["runs"].max()*1.18)
    plt.tight_layout(); save(fig, out / "03_top_run_scorers.png")


def plot_top_bowlers(deliveries, out, top_n=15):
    agg = deliveries.groupby("bowler").agg(
        wickets=("is_wicket","sum"), balls=("total_runs","count"), runs=("total_runs","sum"),
    ).reset_index()
    agg["economy"] = (agg["runs"]/(agg["balls"]/6)).round(2)
    top = agg.nlargest(top_n, "wickets")
    fig, ax = plt.subplots(figsize=(12, 6)); bg(fig, ax)
    y_pos = range(len(top))
    bars = ax.barh(list(y_pos), top["wickets"],
                   color=[TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(top))],
                   edgecolor="white", height=0.75)
    for b, (_, r) in zip(bars, top.iterrows()):
        ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
                f"{int(r['wickets'])} wkts  |  Eco {r['economy']}", va="center", fontsize=8, color=C_ACC)
    ax.set_yticks(list(y_pos)); ax.set_yticklabels(top["bowler"], fontsize=9); ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Wicket Takers", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Total Wickets"); ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, top["wickets"].max()*1.18)
    plt.tight_layout(); save(fig, out / "04_top_wicket_takers.png")


def plot_runs_by_over(deliveries, out):
    def over_agg(df):
        return df.groupby("over")["total_runs"].mean().reset_index(name="avg_runs")
    o1 = over_agg(deliveries[deliveries["innings"]==1])
    o2 = over_agg(deliveries[deliveries["innings"]==2])
    fig, ax = plt.subplots(figsize=(13, 5)); bg(fig, ax)
    ax.plot(o1["over"], o1["avg_runs"], color=C_MAIN, linewidth=2.5, marker="o", markersize=5, label="Innings 1")
    ax.plot(o2["over"], o2["avg_runs"], color=C_ACC,  linewidth=2.5, marker="s", markersize=5, label="Innings 2")
    ax.fill_between(o1["over"], o1["avg_runs"], alpha=0.12, color=C_MAIN)
    ax.fill_between(o2["over"], o2["avg_runs"], alpha=0.12, color=C_ACC)
    ax.axvspan(1, 6,  alpha=0.06, color=C_GREEN, label="Powerplay (1–6)")
    ax.axvspan(7, 15, alpha=0.04, color=C_GREY,  label="Middle (7–15)")
    ax.axvspan(16,20, alpha=0.06, color=C_GOLD,  label="Death (16–20)")
    ax.set_title("Average Runs per Over — Innings 1 vs 2", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Over"); ax.set_ylabel("Avg Runs"); ax.set_xticks(range(1,21))
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout(); save(fig, out / "05_runs_by_over.png")


def plot_wickets_by_over(deliveries, out):
    n = deliveries["match_id"].nunique()
    def wkt_agg(inn):
        w = deliveries[(deliveries["innings"]==inn) & (deliveries["is_wicket"]==1)]
        agg = w.groupby("over").size().reset_index(name="wickets")
        full = pd.DataFrame({"over": range(1,21)})
        merged = full.merge(agg, on="over", how="left").fillna(0)
        merged["wpm"] = merged["wickets"] / n
        return merged["wpm"].values
    x = np.arange(1,21); w = 0.4
    fig, ax = plt.subplots(figsize=(13, 5)); bg(fig, ax)
    ax.bar(x-w/2, wkt_agg(1), width=w, color=C_MAIN, label="Innings 1", edgecolor="white")
    ax.bar(x+w/2, wkt_agg(2), width=w, color=C_ACC,  label="Innings 2", edgecolor="white")
    ax.axvspan(0.5, 6.5,  alpha=0.05, color=C_GREEN)
    ax.axvspan(15.5,20.5, alpha=0.06, color=C_GOLD)
    ax.set_title("Average Wickets per Over per Match", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Over"); ax.set_ylabel("Avg Wickets"); ax.set_xticks(list(x))
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout(); save(fig, out / "06_wickets_by_over.png")


def plot_score_distribution(matches, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); bg(fig, axes[0], axes[1])
    for ax, col, label, color in zip(axes,
        ["innings1_runs","innings2_runs"],
        ["Innings 1 Final Score","Innings 2 Final Score"],
        [C_MAIN, C_ACC]):
        data = matches[col].dropna()
        ax.set_facecolor("#F8F9FA")
        ax.hist(data, bins=25, color=color, edgecolor="white", linewidth=0.6, alpha=0.9)
        ax.axvline(data.mean(),   color=C_GOLD, linewidth=2, linestyle="--", label=f"Mean: {data.mean():.0f}")
        ax.axvline(data.median(), color=C_GREY, linewidth=2, linestyle=":",  label=f"Median: {data.median():.0f}")
        ax.set_title(label, fontweight="bold", color=C_ACC)
        ax.set_xlabel("Total Runs"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.suptitle("Distribution of Team Scores", fontweight="bold", fontsize=14, color=C_ACC, y=1.02)
    plt.tight_layout(); save(fig, out / "07_score_distribution.png")


def plot_team_wins(matches, out):
    valid = matches.dropna(subset=["winner"])
    known = set(matches[["team1","team2"]].values.flatten())
    wc = valid[valid["winner"].isin(known)].groupby("winner").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5, len(wc)*0.45))); bg(fig, ax)
    bars = ax.barh(wc.index, wc.values,
                   color=[TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(wc))],
                   edgecolor="white", height=0.7)
    for b in bars:
        ax.text(b.get_width()+0.5, b.get_y()+b.get_height()/2,
                str(int(b.get_width())), va="center", fontsize=8, color=C_ACC)
    ax.set_title("Total Wins by Team", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Wins"); ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, wc.max()*1.12)
    plt.tight_layout(); save(fig, out / "08_team_wins.png")


def plot_dismissal_types(deliveries, out):
    wkts = deliveries[deliveries["is_wicket"]==1].dropna(subset=["wicket_kind"])
    counts = wkts["wicket_kind"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 6)); bg(fig, ax)
    ax.pie(counts.values, labels=counts.index,
           autopct=lambda p: f"{p:.1f}%\n({int(p/100*sum(counts.values))})",
           colors=[TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(counts))],
           startangle=140, wedgeprops={"edgecolor":"white","linewidth":2},
           textprops={"fontsize":9}, pctdistance=0.78)
    ax.set_title("Wicket Dismissal Types", fontweight="bold", color=C_ACC, fontsize=13, pad=15)
    plt.tight_layout(); save(fig, out / "09_dismissal_types.png")


def plot_powerplay_vs_total(matches, deliveries, out):
    pp = (deliveries[(deliveries["in_powerplay"]==1) & (deliveries["innings"]==1)]
          .groupby("match_id")["total_runs"].sum().reset_index(name="pp_runs"))
    merged = matches[["match_id","innings1_runs","season_year"]].merge(pp, on="match_id", how="inner").dropna()
    fig, ax = plt.subplots(figsize=(10, 6)); bg(fig, ax)
    seasons = sorted(merged["season_year"].dropna().unique())
    cmap = plt.cm.get_cmap("tab20", len(seasons))
    for i, s in enumerate(seasons):
        sub = merged[merged["season_year"]==s]
        ax.scatter(sub["pp_runs"], sub["innings1_runs"],
                   color=cmap(i), label=str(int(s)), alpha=0.7, s=40, edgecolors="white", linewidths=0.4)
    m, b = np.polyfit(merged["pp_runs"], merged["innings1_runs"], 1)
    xl = np.linspace(merged["pp_runs"].min(), merged["pp_runs"].max(), 100)
    ax.plot(xl, m*xl+b, color=C_MAIN, linewidth=2, linestyle="--", label=f"Trend (slope={m:.2f})")
    ax.set_title("Powerplay Score vs Final Innings 1 Score", fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Powerplay Runs (Overs 1–6)"); ax.set_ylabel("Innings 1 Total Runs")
    ax.legend(fontsize=7, ncol=3, loc="upper left"); ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout(); save(fig, out / "10_powerplay_vs_total.png")


def plot_correlation_heatmap(matches, deliveries, out):
    pp  = (deliveries[(deliveries["in_powerplay"]==1) & (deliveries["innings"]==1)]
           .groupby("match_id")["total_runs"].sum().reset_index(name="pp_runs_inn1"))
    death = (deliveries[(deliveries["over"]>=17) & (deliveries["innings"]==1)]
             .groupby("match_id")["total_runs"].sum().reset_index(name="death_runs_inn1"))
    df = matches[["match_id","innings1_runs","innings2_runs","innings1_wickets",
                  "innings2_wickets","win_by_runs","win_by_wickets"]].copy()
    df = df.merge(pp, on="match_id", how="left").merge(death, on="match_id", how="left")
    df["toss_bat_first"] = (matches["toss_decision"]=="bat").astype(int)
    df["team1_won"] = (matches["winner"]==matches["team1"]).astype(int)
    corr_cols = [c for c in ["innings1_runs","innings2_runs","innings1_wickets","innings2_wickets",
                              "pp_runs_inn1","death_runs_inn1","toss_bat_first","team1_won"] if c in df.columns]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8)); bg(fig, ax)
    im = ax.imshow(corr.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    labels = [c.replace("_"," ").title() for c in corr_cols]
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(v)>0.5 else C_ACC)
    ax.set_title("Feature Correlation Matrix", fontweight="bold", color=C_ACC, pad=12)
    plt.tight_layout(); save(fig, out / "11_correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════
# ── NEW PLOTS (12–19) ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

# ── Plot 12 — Chase Success Rate by Target Range ──────────────────
def plot_chase_success_by_target(matches, out):
    """
    Bins all chasing games by target (innings1 score + 1) and shows
    win % for the chasing team.  Reveals the 'safe' vs 'dangerous' target zones.
    """
    df = matches.dropna(subset=["winner","innings1_runs","innings2_team"]).copy()
    df["target"]      = df["innings1_runs"] + 1
    df["chase_won"]   = (df["winner"] == df["innings2_team"]).astype(int)

    bins   = [100, 120, 140, 160, 180, 200, 220, 300]
    labels = ["100–119","120–139","140–159","160–179","180–199","200–219","220+"]
    df["target_band"] = pd.cut(df["target"], bins=bins, labels=labels, right=False)

    agg = (df.groupby("target_band", observed=True)["chase_won"]
             .agg(["mean","count"]).reset_index())
    agg.columns = ["band","win_rate","count"]
    agg["win_pct"] = agg["win_rate"] * 100

    fig, ax = plt.subplots(figsize=(12, 5)); bg(fig, ax)
    colors = [C_GREEN if w >= 50 else C_MAIN for w in agg["win_pct"]]
    bars = ax.bar(agg["band"], agg["win_pct"], color=colors, edgecolor="white",
                  linewidth=0.8, width=0.65)

    for b, (_, row) in zip(bars, agg.iterrows()):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.8,
                f"{row['win_pct']:.1f}%\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=8.5, color=C_ACC, fontweight="bold")

    ax.axhline(50, linestyle="--", color=C_GOLD, linewidth=1.5, label="50% (coin flip)")
    ax.set_title("Chase Success Rate by Target Range",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Target Score Band")
    ax.set_ylabel("Chasing Team Win %")
    ax.set_ylim(0, 85)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Annotation
    ax.text(0.98, 0.92, "Green = chasing team favoured",
            transform=ax.transAxes, ha="right", fontsize=8,
            color=C_GREEN, style="italic")

    plt.tight_layout()
    save(fig, out / "12_chase_success_by_target.png")


# ── Plot 13 — Run Rate Pressure Heatmap (Over × Wickets Lost) ─────
def plot_run_rate_pressure_heatmap(deliveries, out):
    """
    2-D heatmap: rows = wickets lost so far, cols = over number.
    Cell value = average runs scored on THAT delivery.
    Reveals how run-scoring changes with both time and wickets in hand.
    """
    df = deliveries[deliveries["innings"] == 1].copy()

    # cumulative_wickets at start of delivery = wickets before this ball
    df["wkts_before"] = df["cumulative_wickets"] - df["is_wicket"]
    df["wkts_before"] = df["wkts_before"].clip(0, 9)

    pivot = (df.groupby(["wkts_before","over"])["total_runs"]
               .mean()
               .unstack(level="over")
               .reindex(index=range(10), columns=range(1,21)))

    fig, ax = plt.subplots(figsize=(14, 6)); bg(fig, ax)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=pivot.values[~np.isnan(pivot.values)].max())

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Avg Runs / Ball", fontsize=9)

    ax.set_xticks(range(20)); ax.set_xticklabels(range(1,21), fontsize=8)
    ax.set_yticks(range(10));  ax.set_yticklabels(
        [f"{w} wkt{'s' if w!=1 else ''} lost" for w in range(10)], fontsize=8)

    # Annotate cells
    for i in range(10):
        for j in range(20):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if 0.2 < v < 1.4 else "white")

    ax.set_title("Run Rate Pressure Map — Innings 1\n(Avg Runs per Ball by Over & Wickets Lost)",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Over Number")
    ax.set_ylabel("Wickets Lost Before This Ball")

    plt.tight_layout()
    save(fig, out / "13_run_rate_pressure_heatmap.png")


# ── Plot 14 — Bowling Economy by Phase ───────────────────────────
def plot_economy_by_phase(deliveries, out, top_n=20):
    """
    For the top N bowlers by balls bowled, shows their economy broken
    into three phases: powerplay, middle, death.
    Reveals phase specialists vs all-rounders.
    """
    def phase(row):
        if row["over"] <= 6:   return "Powerplay (1–6)"
        if row["over"] <= 15:  return "Middle (7–15)"
        return "Death (16–20)"

    df = deliveries.copy()
    df["phase"] = df.apply(phase, axis=1)

    # top bowlers by total balls
    top_bowlers = (df.groupby("bowler")["total_runs"].count()
                     .nlargest(top_n).index.tolist())

    df2 = df[df["bowler"].isin(top_bowlers)]
    eco = (df2.groupby(["bowler","phase"])
              .agg(runs=("total_runs","sum"), balls=("total_runs","count"))
              .reset_index())
    eco["economy"] = eco["runs"] / eco["balls"] * 6

    phases    = ["Powerplay (1–6)", "Middle (7–15)", "Death (16–20)"]
    colors_ph = [C_GREEN, C_GREY, C_MAIN]

    # Sort bowlers by overall economy
    overall_eco = (df2.groupby("bowler")
                      .apply(lambda x: x["total_runs"].sum() / len(x) * 6)
                      .sort_values()
                      .index.tolist())
    # keep only top_n that exist in overall_eco
    overall_eco = [b for b in overall_eco if b in top_bowlers]

    fig, ax = plt.subplots(figsize=(13, max(6, top_n * 0.42))); bg(fig, ax)
    y_pos  = np.arange(len(overall_eco))
    height = 0.25

    for i, (phase_name, color) in enumerate(zip(phases, colors_ph)):
        sub = eco[eco["phase"] == phase_name].set_index("bowler")
        vals = [sub.loc[b, "economy"] if b in sub.index else np.nan
                for b in overall_eco]
        ax.barh(y_pos + (i - 1) * height, vals, height=height,
                color=color, edgecolor="white", label=phase_name, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(overall_eco, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(8, linestyle="--", color=C_GOLD, linewidth=1.2, label="Economy = 8")
    ax.set_title(f"Bowling Economy by Phase — Top {top_n} Bowlers\n(sorted by overall economy)",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Economy Rate (runs per over)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    save(fig, out / "14_economy_by_phase.png")


# ── Plot 15 — Average Partnership Runs by Wicket Number ──────────
def plot_partnership_by_wicket(deliveries, out):
    """
    For each partnership (1st wicket = openers, 2nd = after 1st wicket falls, etc.),
    shows the average runs added together.
    """
    df = deliveries[deliveries["innings"] == 1].copy()
    df = df.sort_values(["match_id","innings","over","ball"])

    records = []
    for match_id, mdf in df.groupby("match_id"):
        wicket_num   = 0
        partner_runs = 0
        for _, row in mdf.iterrows():
            partner_runs += row["total_runs"]
            if row["is_wicket"] == 1:
                records.append({"wicket_num": wicket_num + 1,
                                 "partnership_runs": partner_runs})
                wicket_num   += 1
                partner_runs  = 0
        # last (unfinished) partnership
        if partner_runs > 0:
            records.append({"wicket_num": wicket_num + 1,
                             "partnership_runs": partner_runs})

    pdf = pd.DataFrame(records)
    pdf = pdf[pdf["wicket_num"] <= 10]

    agg = pdf.groupby("wicket_num")["partnership_runs"].agg(["mean","median","count"]).reset_index()
    agg.columns = ["wicket_num","mean","median","count"]

    fig, ax = plt.subplots(figsize=(12, 5)); bg(fig, ax)
    x = agg["wicket_num"]
    ax.bar(x - 0.2, agg["mean"],   width=0.35, color=C_MAIN,  label="Mean",   edgecolor="white")
    ax.bar(x + 0.2, agg["median"], width=0.35, color=C_ACC,   label="Median", edgecolor="white")

    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"{n}{'st' if n==1 else 'nd' if n==2 else 'rd' if n==3 else 'th'}\n(n={int(c)})"
                        for n, c in zip(agg["wicket_num"], agg["count"])], fontsize=8)
    ax.set_title("Average Partnership Runs by Wicket Number — Innings 1",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Partnership (Wicket Number)")
    ax.set_ylabel("Runs Added")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    save(fig, out / "15_partnership_by_wicket.png")


# ── Plot 16 — Extras Leakage per Team (Bowling Side) ─────────────
def plot_extras_leakage(deliveries, matches, out):
    """
    Per bowling team: total wides + no-balls conceded per match.
    High leakage = costly discipline problems.
    """
    # Map bowling_team per delivery (it's already in deliveries)
    df = deliveries.copy()

    extras_agg = (df.groupby(["match_id","bowling_team"])
                    .agg(wides=("wides","sum"), noballs=("noballs","sum"))
                    .reset_index())
    extras_agg["total_extras"] = extras_agg["wides"] + extras_agg["noballs"]

    team_agg = (extras_agg.groupby("bowling_team")["total_extras"]
                            .agg(["mean","sum","count"])
                            .reset_index()
                            .sort_values("mean", ascending=True))
    team_agg.columns = ["team","avg_per_match","total","matches"]

    # Only keep teams that appear as actual franchises
    known = set(matches[["team1","team2"]].values.flatten())
    team_agg = team_agg[team_agg["team"].isin(known)]

    fig, ax = plt.subplots(figsize=(12, max(5, len(team_agg)*0.45))); bg(fig, ax)
    y_pos = range(len(team_agg))
    colors_bar = [TEAM_PALETTE[i % len(TEAM_PALETTE)] for i in range(len(team_agg))]
    bars = ax.barh(list(y_pos), team_agg["avg_per_match"],
                   color=colors_bar, edgecolor="white", height=0.7)

    for b, (_, row) in zip(bars, team_agg.iterrows()):
        ax.text(b.get_width()+0.05, b.get_y()+b.get_height()/2,
                f"{row['avg_per_match']:.2f}  (total {int(row['total'])}, {int(row['matches'])} matches)",
                va="center", fontsize=7.5, color=C_ACC)

    ax.set_yticks(list(y_pos)); ax.set_yticklabels(team_agg["team"], fontsize=8)
    ax.set_title("Extras Leakage (Wides + No-Balls) per Match — Bowling Side",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Avg Extras per Match")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, team_agg["avg_per_match"].max() * 1.3)

    plt.tight_layout()
    save(fig, out / "16_extras_leakage_per_team.png")


# ── Plot 17 — Score Acceleration Curve ───────────────────────────
def plot_score_acceleration(deliveries, out):
    """
    Rolling 2-over average run rate throughout both innings.
    Shows momentum shape: explosive starts, middle collapse, death surge.
    """
    def rolling_rr(inn):
        df = (deliveries[deliveries["innings"]==inn]
              .groupby("over")["total_runs"].sum()
              .reset_index(name="runs"))
        full = pd.DataFrame({"over": range(1,21)})
        df   = full.merge(df, on="over", how="left").fillna(0)
        df["rr"]         = df["runs"] / 1  # per over
        df["rolling_rr"] = df["rr"].rolling(2, min_periods=1).mean()
        return df

    r1 = rolling_rr(1)
    r2 = rolling_rr(2)

    # Normalise by number of matches to get true average
    n = deliveries["match_id"].nunique()
    r1["rolling_rr"] /= n
    r2["rolling_rr"] /= n

    fig, ax = plt.subplots(figsize=(13, 5)); bg(fig, ax)
    ax.plot(r1["over"], r1["rolling_rr"], color=C_MAIN, linewidth=2.5,
            label="Innings 1", marker="o", markersize=4)
    ax.plot(r2["over"], r2["rolling_rr"], color=C_ACC,  linewidth=2.5,
            label="Innings 2", marker="s", markersize=4)
    ax.fill_between(r1["over"], r1["rolling_rr"], alpha=0.10, color=C_MAIN)
    ax.fill_between(r2["over"], r2["rolling_rr"], alpha=0.10, color=C_ACC)

    # Phase shading
    ax.axvspan(1,  6,  alpha=0.07, color=C_GREEN, label="Powerplay")
    ax.axvspan(7,  15, alpha=0.04, color=C_GREY,  label="Middle")
    ax.axvspan(16, 20, alpha=0.07, color=C_GOLD,  label="Death")

    ax.set_title("Score Acceleration Curve — Rolling 2-Over Avg Run Rate",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_xlabel("Over"); ax.set_ylabel("Avg Run Rate (per over)")
    ax.set_xticks(range(1,21)); ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    save(fig, out / "17_score_acceleration_curve.png")


# ── Plot 18 — Batter Scoring Zone Breakdown ───────────────────────
def plot_scoring_zone_breakdown(deliveries, out):
    """
    Stacked 100% bar chart: for each over phase, what fraction of
    legal balls are dots / singles (1–3) / boundaries (4 or 6)?
    Shows how batting intent shifts through an innings.
    """
    legal = deliveries[deliveries["wides"] == 0].copy()

    def phase_label(over):
        if over <= 6:  return "Powerplay\n(1–6)"
        if over <= 15: return "Middle\n(7–15)"
        return "Death\n(16–20)"

    legal["phase"] = legal["over"].apply(phase_label)
    legal["zone"]  = pd.cut(legal["batter_runs"],
                             bins=[-1, 0, 3, 6],
                             labels=["Dot (0)", "Single/2/3 (1–3)", "Boundary (4/6)"])

    agg = (legal.groupby(["phase","zone"], observed=True)
                .size().unstack(fill_value=0))
    agg_pct = agg.div(agg.sum(axis=1), axis=0) * 100

    phase_order = ["Powerplay\n(1–6)", "Middle\n(7–15)", "Death\n(16–20)"]
    agg_pct = agg_pct.reindex(phase_order)

    fig, ax = plt.subplots(figsize=(10, 5)); bg(fig, ax)
    colors_z = [C_GREY, C_GOLD, C_GREEN]
    bottom = np.zeros(len(phase_order))

    for col, color in zip(agg_pct.columns, colors_z):
        vals = agg_pct[col].values
        bars = ax.bar(range(len(phase_order)), vals, bottom=bottom,
                      color=color, label=str(col), edgecolor="white", linewidth=0.6)
        for i, (v, bot) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(i, bot + v/2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if color in [C_GREY, C_GREEN] else C_ACC)
        bottom += vals

    ax.set_xticks(range(len(phase_order)))
    ax.set_xticklabels(phase_order, fontsize=10)
    ax.set_title("Batter Scoring Zone Breakdown by Phase — Innings 1\n"
                 "(% of legal balls that are dots / 1–3 runs / boundaries)",
                 fontweight="bold", color=C_ACC, pad=12)
    ax.set_ylabel("% of Legal Deliveries")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    save(fig, out / "18_scoring_zone_breakdown.png")


# ── Plot 19 — Win Margin Distribution ────────────────────────────
def plot_win_margin_distribution(matches, out):
    """
    Two histograms side by side:
      Left  — wins by RUNS (batting first wins)
      Right — wins by WICKETS (chasing wins)
    Shows how comfortable victories tend to be.
    """
    runs_wins    = matches["win_by_runs"].dropna()
    wickets_wins = matches["win_by_wickets"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); bg(fig, axes[0], axes[1])

    for ax, data, title, color, xlabel in zip(
        axes,
        [runs_wins, wickets_wins],
        ["Wins by Runs (Batting First)", "Wins by Wickets (Chasing)"],
        [C_MAIN, C_ACC],
        ["Margin (runs)", "Margin (wickets remaining)"]
    ):
        ax.set_facecolor("#F8F9FA")
        ax.hist(data, bins=20, color=color, edgecolor="white", linewidth=0.6, alpha=0.9)
        ax.axvline(data.mean(),   color=C_GOLD, linewidth=2, linestyle="--",
                   label=f"Mean: {data.mean():.1f}")
        ax.axvline(data.median(), color=C_GREY, linewidth=2, linestyle=":",
                   label=f"Median: {data.median():.0f}")
        ax.set_title(title, fontweight="bold", color=C_ACC)
        ax.set_xlabel(xlabel); ax.set_ylabel("Frequency")
        ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.35)

        # Annotate total count
        ax.text(0.97, 0.93, f"n = {len(data)}", transform=ax.transAxes,
                ha="right", fontsize=9, color=C_ACC)

    plt.suptitle("Win Margin Distribution", fontweight="bold", fontsize=14,
                 color=C_ACC, y=1.02)
    plt.tight_layout()
    save(fig, out / "19_win_margin_distribution.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IPL Visualizations — Extended")
    parser.add_argument("--matches",     required=True)
    parser.add_argument("--deliveries",  required=True)
    parser.add_argument("--output_dir",  default="./plots")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    matches, deliveries = load_data(args.matches, args.deliveries)
    print(f"\nGenerating 19 plots → {out}/\n")

    # ── Existing ──────────────────────────────────────
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

    # ── New ───────────────────────────────────────────
    plot_chase_success_by_target(matches, out)
    plot_run_rate_pressure_heatmap(deliveries, out)
    plot_economy_by_phase(deliveries, out)
    plot_partnership_by_wicket(deliveries, out)
    plot_extras_leakage(deliveries, matches, out)
    plot_score_acceleration(deliveries, out)
    plot_scoring_zone_breakdown(deliveries, out)
    plot_win_margin_distribution(matches, out)

    print(f"\nAll 19 plots saved to: {out}/")


if __name__ == "__main__":
    main()