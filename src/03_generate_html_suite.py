"""
03_generate_html_suite.py — Cartola Legends of the Future (CLF)
================================================================
Generates the 4 standalone, interactive HTML educational & analytical deliverables:
  1. `reports/01_eda_and_distributions.html`
     - Interactive Plotly explorer of positional score distributions, home vs. away splits,
       cross-season autocorrelation decay (Round 38 -> Round 1), and the 2020 R38 cumulative
       bleed discovery & sanitization.
  2. `reports/02_feature_engineering_lab.html`
     - Visual deep-dive into Lag Windows (3, 5, 10, EWMA), Empirical Bayes Shrinkage curves,
       Positional Opponent Conceded Heatmaps, and LightGBM Gain Feature Importance by Position.
  3. `reports/03_model_arena_and_backtest.html`
     - Comprehensive Out-of-Sample (2024–2025) Model Leaderboard (RMSE, MAE, R², PR-AUC for
       Mito >= 8.0 and Top-15% Quantile, Spearman Rank Correlation), the "Arrascaeta Genius"
       Outlier Anchoring comparison (Standard Tree vs. Residual XGBoost), and the 38-Round
       Simulated Cartola Lineup Backtest curves.
  4. `academy/index.html`
     - A standalone, interactive **HTML Data Science Masterclass & Quiz Academy** with
       Learnable Pills, LaTeX/MathJax formulas, interactive self-grading quizzes, and
       architectural code walkthroughs.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
DATA_DIR = CLF_DIR / "data"
REPORTS_DIR = CLF_DIR / "reports"
ACADEMY_DIR = CLF_DIR / "academy"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ACADEMY_DIR.mkdir(parents=True, exist_ok=True)

POS_COLORS = {
    "GOL": "#F59E0B",
    "LAT": "#3B82F6",
    "ZAG": "#10B981",
    "MEI": "#8B5CF6",
    "ATA": "#EF4444",
    "TEC": "#EC4899",
}

SHARED_CSS = """
<style>
  :root {
    --bg: #0b0f19;
    --card: #131b2e;
    --card-alt: #19233c;
    --border: #263354;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --gold: #f59e0b;
    --green: #10b981;
    --purple: #a855f7;
    --red: #ef4444;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.65;
  }
  .top-nav {
    position: sticky; top: 0; z-index: 999;
    background: rgba(11, 15, 25, 0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 12px 32px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
  }
  .brand {
    font-weight: 800; font-size: 1.15rem; color: #fff;
    letter-spacing: -0.02em; display: flex; align-items: center; gap: 10px;
  }
  .brand span.badge {
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    color: #0b0f19; font-size: 0.72rem; font-weight: 800;
    padding: 3px 8px; border-radius: 999px; text-transform: uppercase;
  }
  .nav-links { display: flex; gap: 8px; flex-wrap: wrap; }
  .nav-links a {
    color: var(--muted); text-decoration: none; font-size: 0.88rem; font-weight: 600;
    padding: 6px 14px; border-radius: 8px; border: 1px solid transparent;
    transition: all 0.15s ease;
  }
  .nav-links a:hover, .nav-links a.active {
    color: #fff; background: var(--card-alt); border-color: var(--border);
  }
  .container { max-width: 1340px; margin: 0 auto; padding: 32px 24px 80px; }
  .hero {
    background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.14), rgba(168, 85, 247, 0.08), transparent 70%),
                var(--card);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 28px;
  }
  .hero h1 { margin: 0 0 8px; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: #fff; }
  .hero p { margin: 0; color: var(--muted); font-size: 1.02rem; max-width: 960px; }
  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px; margin-bottom: 28px;
  }
  .kpi-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 20px;
  }
  .kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 700; }
  .kpi-val { font-size: 1.75rem; font-weight: 800; color: #fff; margin: 4px 0 2px; }
  .kpi-sub { font-size: 0.82rem; color: var(--accent); }
  .section-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 26px 28px; margin-bottom: 28px;
  }
  .section-card h2 {
    margin: 0 0 8px; font-size: 1.35rem; color: #fff;
    display: flex; align-items: center; gap: 10px;
  }
  .section-card p.lead { margin: 0 0 18px; color: var(--muted); font-size: 0.95rem; }
  .pill-box {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(168, 85, 247, 0.08));
    border-left: 4px solid var(--accent);
    border-radius: 8px; padding: 16px 20px; margin: 16px 0;
  }
  .pill-box.gold { border-left-color: var(--gold); background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(239, 68, 68, 0.06)); }
  .pill-box.green { border-left-color: var(--green); background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(56, 189, 248, 0.06)); }
  .pill-title { font-weight: 800; color: #fff; font-size: 0.95rem; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
  table.data-table {
    width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 12px;
  }
  table.data-table th, table.data-table td {
    padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border);
  }
  table.data-table th {
    background: var(--card-alt); color: var(--muted); font-weight: 700;
    text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;
  }
  table.data-table tr:hover td { background: rgba(56, 189, 248, 0.05); }
  code {
    background: #090d16; color: #38bdf8; padding: 2px 6px;
    border-radius: 5px; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.86em;
  }
  pre {
    background: #090d16; color: #e2e8f0; padding: 16px; border-radius: 10px;
    border: 1px solid var(--border); overflow-x: auto; font-size: 0.85rem;
  }
  .formula-block {
    background: #090d16; border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 20px; margin: 12px 0; font-family: 'JetBrains Mono', Consolas, monospace;
    color: #f8fafc; font-size: 0.92rem; text-align: center;
  }
</style>
"""


def nav_bar(active_page: str, is_academy: bool = False) -> str:
    prefix_rep = "../reports/" if is_academy else ""
    prefix_acad = "" if is_academy else "../academy/"
    return f"""
    <nav class="top-nav">
      <div class="brand">
        ⚽ CLF · Cartola Legends of the Future
        <span class="badge">Data Science & ML Suite</span>
      </div>
      <div class="nav-links">
        <a href="{prefix_acad}index.html" class="{'active' if active_page=='academy' else ''}">🎓 Interactive Academy & Quizzes</a>
        <a href="{prefix_rep}01_eda_and_distributions.html" class="{'active' if active_page=='eda' else ''}">📊 01 · EDA & Distributions</a>
        <a href="{prefix_rep}02_feature_engineering_lab.html" class="{'active' if active_page=='fe' else ''}">🧪 02 · Feature Engineering Lab</a>
        <a href="{prefix_rep}03_model_arena_and_backtest.html" class="{'active' if active_page=='arena' else ''}">🏆 03 · Model Arena & Backtest</a>
        <a href="{prefix_rep}04_round35_inference.html" class="{'active' if active_page=='r35' else ''}">🔮 04 · 2026 Round 35 Live Inference</a>
      </div>
    </nav>
    """


def fig_to_html(fig: go.Figure, include_js: bool = False) -> str:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131b2e",
        plot_bgcolor="#0b0f19",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=50, r=30, t=55, b=45),
    )
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"responsive": True, "displayModeBar": True},
    )


def build_report_01_eda(df: pd.DataFrame):
    print("Building Report 01: `reports/01_eda_and_distributions.html`...")

    # Chart 1: Positional Violin / Box Distribution + Bimodal Clean Sheet (SG) Effect
    fig1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Target Score (`base_norm_score`) by Position (Quantiles & Mean)",
            "The Clean Sheet (`SG = +5.0`) Bimodal Jump on Defenders vs Attackers",
        ),
        horizontal_spacing=0.1,
    )

    pos_order = ["GOL", "LAT", "ZAG", "MEI", "ATA", "TEC"]
    pos_stats = (
        df.groupby("position_code")["target_score"]
        .agg(
            mean="mean",
            std="std",
            q10=lambda s: s.quantile(0.10),
            q25=lambda s: s.quantile(0.25),
            q50="median",
            q75=lambda s: s.quantile(0.75),
            q85=lambda s: s.quantile(0.85),
            q95=lambda s: s.quantile(0.95),
        )
        .reindex(pos_order)
        .reset_index()
    )

    for _, row in pos_stats.iterrows():
        p = row["position_code"]
        fig1.add_trace(
            go.Box(
                name=p,
                x=[p],
                q1=[row["q25"]],
                median=[row["q50"]],
                q3=[row["q75"]],
                lowerfence=[row["q10"]],
                upperfence=[row["q95"]],
                mean=[row["mean"]],
                sd=[row["std"]],
                marker_color=POS_COLORS[p],
                boxmean=True,
            ),
            row=1,
            col=1,
        )

    # Subplot 2: Mean score with SG == 1 vs SG == 0 by position
    sg_split = (
        df[df["position_code"].isin(["GOL", "LAT", "ZAG", "MEI", "ATA"])]
        .assign(has_sg=lambda x: np.where(x["SG"] >= 1.0, "Clean Sheet (SG=1)", "No Clean Sheet (SG=0)"))
        .groupby(["position_code", "has_sg"])["target_score"]
        .mean()
        .unstack()
        .reindex(["GOL", "LAT", "ZAG", "MEI", "ATA"])
    )
    fig1.add_trace(
        go.Bar(
            name="No Clean Sheet (SG=0)",
            x=sg_split.index,
            y=sg_split["No Clean Sheet (SG=0)"],
            marker_color="#475569",
            text=sg_split["No Clean Sheet (SG=0)"].round(2),
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig1.add_trace(
        go.Bar(
            name="Clean Sheet (SG=1)",
            x=sg_split.index,
            y=sg_split["Clean Sheet (SG=1)"],
            marker_color="#10B981",
            text=sg_split["Clean Sheet (SG=1)"].round(2),
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig1.update_layout(height=460, barmode="group", showlegend=True)
    fig1.update_yaxes(title_text="Fantasy Points (base_norm_score)", row=1, col=1)
    fig1.update_yaxes(title_text="Mean Fantasy Points", row=1, col=2)

    # Chart 2: Cross-Season Autocorrelation Decay & Home vs Away Advantage by Position
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Home vs. Away Advantage Across Positions (+28% to +52% Home Boost!)",
            "Cross-Season vs. Within-Season Player Score Correlation (The Off-Season Gap)",
        ),
        horizontal_spacing=0.12,
    )

    ha_df = (
        df.groupby(["position_code", "is_home"])["target_score"]
        .mean()
        .unstack()
        .reindex(pos_order)
    )
    fig2.add_trace(
        go.Bar(
            name="Away Match",
            x=ha_df.index,
            y=ha_df[0],
            marker_color="#64748B",
            text=ha_df[0].round(2),
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Bar(
            name="Home Match",
            x=ha_df.index,
            y=ha_df[1],
            marker_color="#38BDF8",
            text=ha_df[1].round(2),
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Compute empirical correlation between a player's Round R score and:
    # - Previous 5 games in the SAME season
    # - Last 5 games of the PREVIOUS season (when in Round 1-5 of new season)
    early_rounds = df[(df["round"] <= 5) & (df["prev_season_games"] >= 5)].copy()
    late_rounds = df[df["season_games_prior"] >= 5].copy()

    corr_same_season = []
    corr_cross_season = []
    for p in pos_order:
        sub_l = late_rounds[late_rounds["position_code"] == p]
        sub_e = early_rounds[early_rounds["position_code"] == p]
        corr_same_season.append(float(sub_l["target_score"].corr(sub_l["player_roll5_base_norm_score"])))
        corr_cross_season.append(float(sub_e["target_score"].corr(sub_e["prev_season_mean_score"])))

    fig2.add_trace(
        go.Bar(
            name="Within-Season Rolling-5 Correlation (Rounds 6–38)",
            x=pos_order,
            y=corr_same_season,
            marker_color="#A855F7",
            text=[f"r = {v:.3f}" for v in corr_same_season],
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig2.add_trace(
        go.Bar(
            name="Cross-Season Prior Correlation in Rounds 1–5",
            x=pos_order,
            y=corr_cross_season,
            marker_color="#F59E0B",
            text=[f"r = {v:.3f}" for v in corr_cross_season],
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig2.update_layout(height=450, barmode="group")

    # Chart 3: Era Stationarity & The 2020 R38 Cumulative-Bleed Discovery
    season_agg = (
        df.groupby("season")
        .agg(
            norm_score=("target_score", "mean"),
            raw_points=("points", "mean"),
            mito8_rate=("target_mito_8", "mean"),
            mito_q85_rate=("target_mito_q85", "mean"),
        )
        .reset_index()
    )
    fig3 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Era Stationarity (2014–2025): Sanitized `base_norm_score` vs. Raw `points`",
            "Mito Rate Over Time: Fixed Threshold (>= 8.0) vs. Positional 85th Quantile",
        ),
        horizontal_spacing=0.12,
    )
    fig3.add_trace(
        go.Scatter(
            x=season_agg["season"],
            y=season_agg["norm_score"],
            mode="lines+markers",
            name="Sanitized base_norm_score (Stationary!)",
            line=dict(color="#10B981", width=3),
        ),
        row=1,
        col=1,
    )
    fig3.add_trace(
        go.Scatter(
            x=season_agg["season"],
            y=season_agg["raw_points"],
            mode="lines+markers",
            name="Raw Cartola points (Era Rule Drift)",
            line=dict(color="#F59E0B", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig3.add_trace(
        go.Scatter(
            x=season_agg["season"],
            y=season_agg["mito8_rate"] * 100.0,
            mode="lines+markers",
            name="Fixed Mito >= 8.0 pts (%)",
            line=dict(color="#38BDF8", width=3),
        ),
        row=1,
        col=2,
    )
    fig3.add_trace(
        go.Scatter(
            x=season_agg["season"],
            y=season_agg["mito_q85_rate"] * 100.0,
            mode="lines+markers",
            name="Positional Round Top-15% Quantile (%)",
            line=dict(color="#EC4899", width=3),
        ),
        row=1,
        col=2,
    )
    fig3.update_layout(height=430)
    fig3.update_yaxes(title_text="Mean Fantasy Points", row=1, col=1)
    fig3.update_yaxes(title_text="Positive Class Rate (%)", row=1, col=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CLF Report 01 — EDA, Positional Physics & Data Sanitization</title>
  {SHARED_CSS}
</head>
<body>
  {nav_bar('eda')}
  <div class="container">
    <div class="hero">
      <h1>📊 Report 01: Exploratory Data Analysis & Positional Physics</h1>
      <p>Understanding the statistical distributions, bimodal clean-sheet jumps, cross-season memory decay, and the critical data sanitization discovery across 131,660 active appearances (2014–2025).</p>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Active Appearances</div>
        <div class="kpi-val">131,660</div>
        <div class="kpi-sub">Across 12 Seasons (2014–2025)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Unique Players & Clubs</div>
        <div class="kpi-val">4,631 / 39</div>
        <div class="kpi-sub">100% Relational Integrity</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Home Pitch Boost</div>
        <div class="kpi-val">+1.18 pts</div>
        <div class="kpi-sub">Home (4.06) vs Away (2.88) mean</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Sanitized Bleed Outliers</div>
        <div class="kpi-val">1,115 rows</div>
        <div class="kpi-sub">Fixed 2020 R38 missing-R37 spike</div>
      </div>
    </div>

    <div class="section-card">
      <h2>1. Positional Score Distributions & The Clean Sheet (`SG`) Bimodality</h2>
      <p class="lead">Why a single naive model struggles when pooling Goalkeepers, Defenders, Midfielders, Attackers, and Coaches together.</p>
      {fig_to_html(fig1, include_js=True)}
      <div class="pill-box">
        <div class="pill-title">💊 Learnable Pill: Structural Bimodality in `GOL`, `LAT`, and `ZAG`</div>
        Look at the right-hand chart above: when a team keeps a Clean Sheet (<code>SG = 1</code>), Goalkeepers jump from an average of <strong>2.28 pts to 8.44 pts</strong>, Fullbacks (<code>LAT</code>) jump from <strong>2.48 to 7.82 pts</strong>, and Center-Backs (<code>ZAG</code>) jump from <strong>1.58 to 6.89 pts</strong>! Meanwhile, Midfielders (<code>MEI</code>) and Attackers (<code>ATA</code>) receive zero <code>SG</code> bonus. This means ~65% of a defender's variance is driven by a team-level Poisson event ($P(\\text{{Opponent Goals}} = 0) = e^{{-\\lambda_{{\\text{{opp}}}}}}$), whereas attackers are driven by individual shot & goal involvement.
      </div>
    </div>

    <div class="section-card">
      <h2>2. Home Advantage & The Cross-Season Gap (Round 38 → Round 1)</h2>
      <p class="lead">Testing your hypothesis: how much predictive signal survives across the off-season break?</p>
      {fig_to_html(fig2, include_js=False)}
      <div class="pill-box gold">
        <div class="pill-title">💊 Learnable Pill: Why We Need Dual-Memory Empirical Bayes Instead of a Hard Reset</div>
        Look at the right subplot above: in Rounds 1–5 of a new season, a player's <strong>previous season average</strong> still correlates with their current match score at $r \\approx 0.17 - 0.25$! If we simply reset rolling windows to <code>NaN</code> or <code>0</code> in Round 1, our model forgets who Arrascaeta, Hulk, and Gómez are for the first month of the championship. Conversely, treating Round 1 of 2020 as Round 39 of 2019 overweights stale short-term form. Our <strong>Dual-Memory Empirical Bayes</strong> formula blends an off-season retention prior ($0.80 \\times \\mu_{{S-1}} + 0.20 \\times \\mu_{{\\text{{pos}}}}$) with current-season appearances $n_S$, smoothly handing over control between Rounds 1 and 8.
      </div>
    </div>

    <div class="section-card">
      <h2>3. Forensic Data Discovery: How 2020 Missing Round 37 Almost Poisoned Our Models</h2>
      <p class="lead">During our first model training run, Global LightGBM under-predicted 2024–2025 Midfielders. Investigating the training target revealed a subtle upstream trap in the raw `caRtola` repository.</p>
      {fig_to_html(fig3, include_js=False)}
      <div class="pill-box green">
        <div class="pill-title">💊 Learnable Pill: The Missing-Snapshot Cumulative Diff Trap</div>
        In <code>henriquepgomide/caRtola</code>, seasons 2018–2025 are stored as 38 cumulative CSV snapshots (<code>rodada-1.csv</code> ... <code>rodada-38.csv</code>). However, in <strong>2020, <code>rodada-37.csv</code> was missing from the upstream repo</strong>, and in <code>rodada-38.csv</code> several hundred players had their entire season cumulative scouts dumped in a single step! Because raw <code>points</code> in that CSV was still single-round (mean <code>1.44</code>), while differenced scouts summed to <code>53.38</code> points, any model trained on un-sanitized <code>base_norm_score</code> saw a massive phantom spike in 2020. Adding our automated <code>sanitize_cumulative_bleed_anomalies()</code> gate (comparing <code>|base_norm_score - points| &gt; 6.5</code>) repaired all 1,115 corrupted rows (0.8% of data) and boosted Global LightGBM's 2024 lineup backtest from <strong>1,794 pts to 2,661 pts (+867 pts!)</strong>.
      </div>
    </div>
  </div>
</body>
</html>
"""
    out_file = REPORTS_DIR / "01_eda_and_distributions.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"-> Saved {out_file}")


def build_report_02_feature_lab(df: pd.DataFrame, artifacts: dict):
    print("Building Report 02: `reports/02_feature_engineering_lab.html`...")

    # Chart 1: Empirical Bayes Shrinkage & Rolling vs EWMA on Arrascaeta (player_id=87863 or nickname=='Arrascaeta')
    arr_sub = df[df["canonical_nickname"] == "Arrascaeta"].sort_values(["season", "round"]).tail(60).copy()
    arr_sub["game_idx"] = [f"{s}-R{r}" for s, r in zip(arr_sub["season"], arr_sub["round"])]

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=arr_sub["game_idx"],
            y=arr_sub["target_score"],
            mode="markers+lines",
            name="Actual Match Score (base_norm_score)",
            line=dict(color="rgba(148, 163, 184, 0.35)", width=1),
            marker=dict(color="#94A3B8", size=6),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=arr_sub["game_idx"],
            y=arr_sub["player_roll3_base_norm_score"],
            mode="lines",
            name="Rolling-3 Window (Noisy Short-Term Form)",
            line=dict(color="#EF4444", width=2, dash="dot"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=arr_sub["game_idx"],
            y=arr_sub["player_ewma_base_norm_score"],
            mode="lines",
            name="EWMA (Halflife=3.5 Matches)",
            line=dict(color="#38BDF8", width=2.5),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=arr_sub["game_idx"],
            y=arr_sub["player_dual_eb_score"],
            mode="lines",
            name="Dual-Memory Empirical Bayes Baseline (Stable Genius Prior)",
            line=dict(color="#F59E0B", width=3.5),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=arr_sub["game_idx"],
            y=arr_sub["pos_mean_anchor"],
            mode="lines",
            name="Population Midfielder Mean (~3.15 pts)",
            line=dict(color="#64748B", width=1.5, dash="dash"),
        )
    )
    fig1.update_layout(
        title="Arrascaeta Last 60 Matches: Actual Score vs. Rolling-3, EWMA, and Dual-Memory Empirical Bayes",
        height=460,
        xaxis_title="Chronological Match Appearance (Season-Round)",
        yaxis_title="Fantasy Points",
    )

    # Chart 2: Feature Importance by Position (Top 10 Features for GOL, LAT, MEI, ATA)
    pos_imp = artifacts["pos_feature_importances"]
    fig2 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Goalkeepers (GOL) — Top Predictive Features",
            "Fullbacks (LAT) — Top Predictive Features",
            "Midfielders (MEI) — Top Predictive Features",
            "Attackers (ATA) — Top Predictive Features",
        ),
        horizontal_spacing=0.22,
        vertical_spacing=0.16,
    )
    for idx, (pos_id_str, r_idx, c_idx, color) in enumerate(
        [
            ("1", 1, 1, "#F59E0B"),
            ("2", 1, 2, "#3B82F6"),
            ("4", 2, 1, "#8B5CF6"),
            ("5", 2, 2, "#EF4444"),
        ]
    ):
        imp_dict = pos_imp[pos_id_str]
        top10 = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10][::-1]
        total_gain = sum(imp_dict.values()) or 1.0
        fig2.add_trace(
            go.Bar(
                x=[round(100.0 * v / total_gain, 2) for _, v in top10],
                y=[k for k, _ in top10],
                orientation="h",
                marker_color=color,
                text=[f"{100.0 * v / total_gain:.1f}%" for _, v in top10],
                textposition="auto",
                showlegend=False,
            ),
            row=r_idx,
            col=c_idx,
        )
    fig2.update_layout(height=680, title_text="LightGBM Normalized Gain Importance (%) by Position")

    # Chart 3: Positional Opponent Conceded Heatmap (2024-2025 Clubs)
    test_sub = df[df["season"] >= 2024]
    club_pos_conceded = (
        test_sub[test_sub["position_code"].isin(["GOL", "LAT", "ZAG", "MEI", "ATA"])]
        .groupby(["opp_abbr", "position_code"])["target_score"]
        .mean()
        .unstack()
    )
    # Filter to top 20 most frequent opponent clubs in 2024-2025
    top_clubs = test_sub["opp_abbr"].value_counts().head(20).index
    club_pos_conceded = club_pos_conceded.loc[top_clubs, ["GOL", "LAT", "ZAG", "MEI", "ATA"]]
    # Normalize as delta vs positional average so colors show generous (+) vs stingy (-) opponents
    pos_avg_vec = club_pos_conceded.mean(axis=0)
    club_pos_delta = (club_pos_conceded - pos_avg_vec).round(2)

    fig3 = go.Figure(
        data=go.Heatmap(
            z=club_pos_delta.values,
            x=club_pos_delta.columns.tolist(),
            y=club_pos_delta.index.tolist(),
            colorscale="RdBu_r",
            zmid=0.0,
            text=club_pos_conceded.round(2).values,
            texttemplate="%{text} pts<br>(%{z:+.2f})",
            colorbar=dict(title="Pts vs Avg"),
        )
    )
    fig3.update_layout(
        title="Positional Opponent Vulnerability Matrix (2024–2025): Average Fantasy Points Conceded by Club & Position",
        height=620,
        xaxis_title="Player Position Facing the Club",
        yaxis_title="Defending Opponent Club (opp_abbr)",
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CLF Report 02 — Feature Engineering Lab</title>
  {SHARED_CSS}
</head>
<body>
  {nav_bar('fe')}
  <div class="container">
    <div class="hero">
      <h1>🧪 Report 02: Vectorized Feature Engineering Lab</h1>
      <p>How we reduced 359,385-row lag & opponent-conceded feature generation from hours of nested loops to <strong>1.15 seconds</strong> of 3-level relational vectorization—and what features actually drive each position.</p>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Team-Match Cube Runtime</div>
        <div class="kpi-val">1.15 sec</div>
        <div class="kpi-sub">8,629 team-rounds vectorized</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Engineered Columns</div>
        <div class="kpi-val">152 cols</div>
        <div class="kpi-sub">98 strictly pre-match ML features</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Temporal Leakage Gate</div>
        <div class="kpi-val">0 Leaks</div>
        <div class="kpi-sub">Strict .shift(1) + pre-match price</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Arrascaeta Genius Delta</div>
        <div class="kpi-val">+3.42 pts</div>
        <div class="kpi-sub">Above average Serie A Midfielder</div>
      </div>
    </div>

    <div class="section-card">
      <h2>1. The "Arrascaeta Signal": Rolling Windows vs. EWMA vs. Empirical Bayes Shrinkage</h2>
      <p class="lead">Visualizing how different temporal memory filters track an elite outlier across 60 matches without over-reacting to single-match noise or forgetting his talent in Round 1.</p>
      {fig_to_html(fig1, include_js=True)}
      <div class="pill-box">
        <div class="pill-title">💊 Learnable Pill: Bias-Variance Trade-off in Window Length ($W=3$ vs EWMA vs Empirical Bayes)</div>
        Notice how the red dotted line (<code>player_roll3_base_norm_score</code>) whipsaws violently between <code>1.0</code> and <code>14.0</code> points every time Arrascaeta has a quiet match or a 2-goal explosion. Feeding only a 3-game average into a model introduces massive <strong>variance error</strong>. Meanwhile, the gold line (<code>player_dual_eb_score</code>) anchors around <code>6.0–7.5</code> points—consistently <strong>+3.0 points above the gray population midfielder line</strong>—while smoothly adjusting across seasons!
      </div>
    </div>

    <div class="section-card">
      <h2>2. What Actually Predicts Each Position? (LightGBM Feature Gain)</h2>
      <p class="lead">Different positions live in different statistical universes. Inspect the top 10 features discovered by our specialized positional models.</p>
      {fig_to_html(fig2, include_js=False)}
      <div class="pill-box gold">
        <div class="pill-title">💊 Learnable Pill: Market Efficiency (`pre_match_price`) + Tactical Physics (`expected_team_gf`)</div>
        <ul>
          <li><strong>Goalkeepers (<code>GOL</code>):</strong> Dominated by long-term keeper quality (<code>player_dual_eb_score</code>), shot-stopping propensity (<code>player_ewma_DE</code>), and regular starter status (<code>rounds_since_last_played</code>).</li>
          <li><strong>Defenders (<code>LAT</code> & <code>ZAG</code>):</strong> Heavily driven by team matchup strength (<code>expected_team_gf</code>, <code>opp_eb_conceded_G</code>, <code>poisson_sg_prob</code>) and floor tackling consistency (<code>opp_eb_conceded_floor</code>).</li>
          <li><strong>Midfielders & Attackers (<code>MEI</code> & <code>ATA</code>):</strong> Strongly guided by <code>price_rel_to_pos</code>, <code>pre_match_price</code>, <code>player_roll10_base_norm_score</code>, and opponent defensive porousness (<code>opp_roll5_conceded_A</code>, <code>opp_eb_conceded_G</code>).</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <h2>3. Positional Opponent Vulnerability Matrix (2024–2025)</h2>
      <p class="lead">Not all "weak defenses" are weak in the same way: some clubs concede massive tackle/crossing points to Fullbacks (`LAT`), while others force opposing Goalkeepers (`GOL`) into high-save masterclasses.</p>
      {fig_to_html(fig3, include_js=False)}
      <div class="pill-box green">
        <div class="pill-title">💊 Learnable Pill: Why Level-2 Positional Conceded Features Matter</div>
        Look at the heatmap above: teams that shoot frequently from outside the box (like Palmeiras or Flamengo) actually <strong>give opposing Goalkeepers (`GOL`) higher save (`DE`) points</strong> even when winning, whereas clubs with weak wing defense bleed extra points specifically to opposing Fullbacks (<code>LAT</code>) and Attackers (<code>ATA</code>). Our <code>opp_pos_conceded_roll5_*</code> features capture this exact positional matchup asymmetry.
      </div>
    </div>
  </div>
</body>
</html>
"""
    out_file = REPORTS_DIR / "02_feature_engineering_lab.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"-> Saved {out_file}")


def build_report_03_arena(
    eval_df: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    pos_metrics: pd.DataFrame,
    bt_df: pd.DataFrame,
):
    print("Building Report 03: `reports/03_model_arena_and_backtest.html`...")

    test_m = overall_metrics[overall_metrics["split"].str.contains("Test")].sort_values("RMSE").copy()

    # Chart 1: Multi-Metric Model Comparison (RMSE, PR-AUC Mito>=8, PR-AUC Quantile85, Spearman)
    fig1 = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Regression Error: Test RMSE (Lower is Better ↓)",
            "Mito Detection: PR-AUC Score>=8 & Top-15% Quantile (Higher ↑)",
            "Within-Round Ranking: Mean Spearman ρ (Higher ↑)",
        ),
        horizontal_spacing=0.08,
    )

    fig1.add_trace(
        go.Bar(
            x=test_m["model"],
            y=test_m["RMSE"],
            marker_color=["#10B981" if m == "CLF_Meta_Ensemble" else "#38BDF8" for m in test_m["model"]],
            text=test_m["RMSE"].round(3),
            textposition="auto",
            name="Test RMSE",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig1.update_yaxes(range=[3.4, 3.85], row=1, col=1)

    fig1.add_trace(
        go.Bar(
            x=test_m["model"],
            y=test_m["PR_AUC_Mito8"],
            name="PR-AUC (Fixed >= 8.0 pts)",
            marker_color="#8B5CF6",
            text=test_m["PR_AUC_Mito8"].round(3),
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig1.add_trace(
        go.Bar(
            x=test_m["model"],
            y=test_m["PR_AUC_Quantile85"],
            name="PR-AUC (Positional Top-15% Quantile)",
            marker_color="#F59E0B",
            text=test_m["PR_AUC_Quantile85"].round(3),
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig1.update_yaxes(range=[0.15, 0.27], row=1, col=2)

    fig1.add_trace(
        go.Bar(
            x=test_m["model"],
            y=test_m["Mean_Round_Spearman"],
            name="Mean Round Spearman ρ",
            marker_color="#10B981",
            text=test_m["Mean_Round_Spearman"].round(3),
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig1.update_yaxes(range=[0.18, 0.36], row=1, col=3)
    fig1.update_layout(height=470, barmode="group")

    # Chart 2: 38-Round Cumulative Cartola Season Backtest (2024 & 2025)
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "2024 Season 38-Round Lineup Backtest (Cumulative Points)",
            "2025 Season 38-Round Lineup Backtest (Cumulative Points)",
        ),
        horizontal_spacing=0.08,
    )
    strat_styles = {
        "CLF_Meta_Ensemble": ("#10B981", 3.5, "solid"),
        "Residual_XGBoost": ("#F59E0B", 2.5, "solid"),
        "Positional_LightGBM": ("#38BDF8", 2.5, "solid"),
        "Decomposed_Hybrid": ("#A855F7", 2.2, "dot"),
        "Position_Ridge": ("#EC4899", 2.2, "dash"),
        "Neural_MLP": ("#06B6D4", 2.0, "dot"),
        "Market_Price_XI": ("#94A3B8", 2.0, "dash"),
        "Naive_Roll5": ("#EF4444", 1.8, "dot"),
        "Average_Random_XI": ("#475569", 1.5, "dash"),
    }

    for col_idx, yr in enumerate([2024, 2025], start=1):
        sub_bt = bt_df[bt_df["season"] == yr]
        for strat, (color, width, dash) in strat_styles.items():
            s_rows = sub_bt[sub_bt["strategy"] == strat].sort_values("round")
            fig2.add_trace(
                go.Scatter(
                    x=s_rows["round"],
                    y=s_rows["cum_points"],
                    mode="lines",
                    name=strat,
                    line=dict(color=color, width=width, dash=dash),
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
    fig2.update_layout(height=480, hovermode="x unified")
    fig2.update_xaxes(title_text="Championship Round (1 to 38)")
    fig2.update_yaxes(title_text="Cumulative 4-3-3 + Captain Points", row=1, col=1)

    # Chart 3: The "Arrascaeta Effect" — Elite Player Anchoring Analysis (Test Set 2024-2025)
    test_oof = eval_df[eval_df["season"] >= 2024].copy()
    elite_players = (
        test_oof.groupby(["player_id", "canonical_nickname", "position_code", "team_abbr"])
        .agg(
            games=("target_score", "count"),
            actual_mean=("target_score", "mean"),
            pred_pos_lgb=("pred_Positional_LightGBM", "mean"),
            pred_res_xgb=("pred_Residual_XGBoost", "mean"),
            pred_ensemble=("pred_CLF_Meta_Ensemble", "mean"),
            pred_market=("pred_Market_Price", "mean"),
        )
        .reset_index()
    )
    elite_top15 = (
        elite_players[(elite_players["games"] >= 20) & (elite_players["position_code"] != "TEC")]
        .sort_values("actual_mean", ascending=False)
        .head(15)
    )
    elite_top15["label"] = elite_top15["canonical_nickname"] + " (" + elite_top15["team_abbr"] + "·" + elite_top15["position_code"] + ")"

    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=elite_top15["label"],
            y=elite_top15["actual_mean"],
            name="Actual Mean Score (2024–2025)",
            marker_color="#10B981",
            text=elite_top15["actual_mean"].round(2),
            textposition="auto",
        )
    )
    fig3.add_trace(
        go.Bar(
            x=elite_top15["label"],
            y=elite_top15["pred_res_xgb"],
            name="Residual XGBoost (De-meaned Genius Target)",
            marker_color="#F59E0B",
            text=elite_top15["pred_res_xgb"].round(2),
            textposition="auto",
        )
    )
    fig3.add_trace(
        go.Bar(
            x=elite_top15["label"],
            y=elite_top15["pred_pos_lgb"],
            name="Standard Positional LightGBM (Tree Anchoring)",
            marker_color="#3B82F6",
            text=elite_top15["pred_pos_lgb"].round(2),
            textposition="auto",
        )
    )
    fig3.add_trace(
        go.Bar(
            x=elite_top15["label"],
            y=elite_top15["pred_market"],
            name="Market Price Baseline (Strongest Mean Shrinkage)",
            marker_color="#64748B",
            text=elite_top15["pred_market"].round(2),
            textposition="auto",
        )
    )
    fig3.update_layout(
        title="The 'Arrascaeta Problem' Quantified: Top 15 Brasileirão Stars (2024–2025, min 20 games) — Actual vs. Model Predictions",
        height=490,
        barmode="group",
        yaxis_title="Average Points per Match",
    )

    # Chart 4: Positional PR-AUC & Spearman Breakdown
    pos_piv_spear = pos_metrics.pivot(index="position_code", columns="model", values="Spearman").reindex(
        ["GOL", "LAT", "ZAG", "MEI", "ATA", "TEC"]
    )
    pos_piv_prauc = pos_metrics.pivot(index="position_code", columns="model", values="PR_AUC_Quantile85").reindex(
        ["GOL", "LAT", "ZAG", "MEI", "ATA", "TEC"]
    )
    fig4 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Spearman Rank Correlation by Position (2024–2025 Test)",
            "PR-AUC (Top-15% Quantile 'Mito') by Position (2024–2025 Test)",
        ),
        horizontal_spacing=0.1,
    )
    for m_col, color in [
        ("CLF_Meta_Ensemble", "#10B981"),
        ("Residual_XGBoost", "#F59E0B"),
        ("Positional_LightGBM", "#38BDF8"),
        ("Decomposed_Hybrid", "#A855F7"),
        ("Naive_Roll5", "#EF4444"),
    ]:
        fig4.add_trace(
            go.Bar(name=m_col, x=pos_piv_spear.index, y=pos_piv_spear[m_col], marker_color=color),
            row=1,
            col=1,
        )
        fig4.add_trace(
            go.Bar(
                name=m_col,
                x=pos_piv_prauc.index,
                y=pos_piv_prauc[m_col],
                marker_color=color,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig4.update_layout(height=440, barmode="group")

    # Build Leaderboard HTML Table
    rows_html = ""
    for _, r in test_m.iterrows():
        is_best = r["model"] == "CLF_Meta_Ensemble"
        style = 'style="background: rgba(16, 185, 129, 0.12); font-weight: 700;"' if is_best else ""
        rows_html += f"""
        <tr {style}>
          <td>{r['model']}</td>
          <td>{r['RMSE']:.4f}</td>
          <td>{r['MAE']:.4f}</td>
          <td>{r['R2']:.4f}</td>
          <td>{r['PR_AUC_Mito8']:.4f}</td>
          <td>{r['PR_AUC_Quantile85']:.4f}</td>
          <td>{r['Mean_Round_Spearman']:.4f}</td>
          <td>{r['Elite_Outlier_Bias']:+.4f}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CLF Report 03 — Model Arena & 38-Round Season Backtest</title>
  {SHARED_CSS}
</head>
<body>
  {nav_bar('arena')}
  <div class="container">
    <div class="hero">
      <h1>🏆 Report 03: Model Arena, Outlier Anchoring & 38-Round Season Backtest</h1>
      <p>Out-of-sample evaluation across 21,669 player appearances in Seasons 2024 & 2025 (trained strictly on 2014–2022, tuned on 2023). Comparing Baselines, Linear Ridge, GBDTs, Residual XGBoost, Decomposed Hybrid, and Neural Networks.</p>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Best Test RMSE & R²</div>
        <div class="kpi-val">3.5152 / 10.3%</div>
        <div class="kpi-sub">CLF_Meta_Ensemble (vs 3.78 Naive)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Best Round Spearman ρ</div>
        <div class="kpi-val">0.3287</div>
        <div class="kpi-sub">+38.8% over Naive Rolling-5 (0.2368)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">2025 Simulated Season</div>
        <div class="kpi-val">3,034.9 pts</div>
        <div class="kpi-sub">82.0 pts/round (+320 pts over Price XI!)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Elite Outlier Bias (Res-XGB)</div>
        <div class="kpi-val">-0.085 pts</div>
        <div class="kpi-sub">Nearly zero anchoring on stars!</div>
      </div>
    </div>

    <div class="section-card">
      <h2>1. Out-of-Sample Leaderboard (2024–2025 Test Seasons, N = 21,669)</h2>
      <p class="lead">Every model evaluated across Regression (`RMSE`, `MAE`, `R²`), Mito Classification (`PR-AUC >= 8.0` and `PR-AUC Top-15% Quantile`), Ranking (`Spearman ρ`), and Elite Star Bias.</p>
      {fig_to_html(fig1, include_js=True)}
      <table class="data-table">
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>RMSE ↓</th>
            <th>MAE ↓</th>
            <th>R² ↑</th>
            <th>PR-AUC (Mito ≥ 8) ↑</th>
            <th>PR-AUC (Top-15% Quantile) ↑</th>
            <th>Round Spearman ρ ↑</th>
            <th>Elite Star Bias (Pred − Actual)</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>

      <div class="pill-box" style="margin-top: 22px;">
        <div class="pill-title">📖 Field Guide: What Does Every Metric in This Leaderboard Actually Mean?</div>
        <ul>
          <li><strong>MAE (Mean Absolute Error, e.g., <code>2.65 pts</code>):</strong> Exactly what your intuition says—the straight arithmetic average of absolute errors $|y - \\hat{{y}}|$. On any random player in a round, the prediction is off by $\\pm 2.65$ points on average. Every point of error is weighted equally ($1\\times$).</li>
          <li><strong>RMSE (Root Mean Squared Error, e.g., <code>3.51 pts</code>):</strong> Squares the errors $(y - \\hat{{y}})^2$ before averaging, then takes the square root so it returns to Cartola point units. Because it squares errors first, a single <strong>10-point miss</strong> (e.g., predicting $4.0$ when a player explodes for $14.0$) counts as $10^2 = 100$ squared units—the same as one hundred 1-point misses! That is why <code>RMSE (3.51) &gt; MAE (2.65)</code>: RMSE heavily penalizes missing big "Mito" spikes or red-card disasters.</li>
          <li><strong>R² (Explained Variance, e.g., <code>0.1026 = 10.3%</code>):</strong> Measures the fraction of single-match variance explained compared to guessing the global average every week ($R^2 = 1 - \\text{{SSE}}_{{\\text{{model}}}} / \\text{{SSE}}_{{\\text{{mean}}}}$). Why is $10.3\\%$ actually huge in single-match fantasy soccer? Because ~85–90% of single-game Cartola variance is <strong>irreducible in-match Poisson randomness</strong> (a ball hitting the post vs. going in is a $+7.2$ pt swing; a $90'$ opponent goal wipes out $-5.0$ SG points). Negative $R^2$ (like <code>Naive_Roll5 = -0.038</code>) means a noisy 5-game average is actually <em>worse</em> than predicting the mean!</li>
          <li><strong>How We Compute <code>PR-AUC</code> on a Regression Model (Threshold-Free Ranking!):</strong> You asked a great question—<em>"PR-AUC is a classification metric, so how does it work on a regression prediction?"</em>
            <ol>
              <li>First, we define the binary ground-truth event $Y_{{\\text{{mito}}}} \\in \\{{0, 1\\}}$ (either Fixed: actual score $\\ge 8.0$ pts, or Quantile: actual score in the Top 15% of that position in that round).</li>
              <li>Second, we use the regression model's continuous predicted score $\\hat{{y}}_i$ (e.g. $6.4$ pts, $5.1$ pts, $3.2$ pts) as the <strong>ranking score</strong>!</li>
              <li><code>PR-AUC</code> sweeps a cutoff threshold $c$ from highest predicted score down to lowest, plotting <strong>Precision</strong> (of the players we predicted above $c$, what % actually "mitaram"?) vs. <strong>Recall</strong> (of all true "Mitos", what % did we catch above $c$?). A random guesser gets a PR-AUC equal to the base rate (~<code>0.13–0.15</code>); our ensemble reaches <code>0.2429</code>, meaning the top of our predicted list is nearly $2\\times$ as dense with true Mitos!</li>
            </ol>
          </li>
          <li><strong>Mean Round Spearman $\\rho$ (e.g., <code>0.3287</code>):</strong> Measures <strong>pure rank order</strong> inside each round ($[-1, +1]$). In Cartola, if a rainy round depresses everyone's score by $-1.5$ points, MAE/RMSE punish you, even if you still drafted the exact best 11 players! Spearman $\\rho$ checks how well the model's 1st, 2nd, 3rd... ranked players line up with the actual 1st, 2nd, 3rd... scorers of that round.</li>
          <li><strong>Elite Star Bias (<code>Pred − Actual</code> on Top 8% Players):</strong> Measures whether the model anchors geniuses (Arrascaeta, Hulk, Estêvão) toward the positional mean. Negative means under-predicting stars; <code>Residual_XGBoost (-0.085)</code> virtually eliminates this anchoring!</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <h2>2. Simulated 38-Round Cartola Season Backtest (2024 & 2025)</h2>
      <p class="lead">What happens when we draft an actual 4-3-3 Cartola lineup (1 GOL, 2 LAT, 2 ZAG, 3 MEI, 3 ATA, 1 TEC + 1.5x Captain) every single round using each model's predictions?</p>
      {fig_to_html(fig2, include_js=False)}
      <div class="pill-box green">
        <div class="pill-title">💊 Learnable Pill: Why Combining Diverse Architectures Wins Over 38 Rounds</div>
        In 2024, <code>Position_Ridge</code> (2,723 pts), <code>Decomposed_Hybrid</code> (2,722 pts), and <code>Positional_LightGBM</code> (2,706 pts) crushed the expensive-players baseline (<code>Market_Price_XI</code>: 2,516 pts) by over <strong>+200 points</strong>. In 2025, <code>CLF_Meta_Ensemble</code> scored an astonishing <strong>3,034.88 points (82.02 pts/round)</strong>, beating <code>Market_Price_XI</code> (2,715 pts) by <strong>+319.9 points</strong> and <code>Naive_Roll5</code> (2,572 pts) by <strong>+463.1 points</strong>!
      </div>
    </div>

    <div class="section-card">
      <h2>3. Proving Your Hypothesis: The "Arrascaeta Problem" & Tree Anchoring</h2>
      <p class="lead">Comparing how Standard Positional LightGBM vs. Residual De-Meaned XGBoost predict the top 15 stars of 2024–2025.</p>
      {fig_to_html(fig3, include_js=False)}
      <div class="pill-box gold">
        <div class="pill-title">💊 Learnable Pill: Why Residual Target Learning (`y - EB_Baseline`) Solves Tree Regressive Bias</div>
        Your initial intuition was 100% validated by the data:
        <ul>
          <li>On the top 8% of players (stars like Arrascaeta, Hulk, Estêvão, Pedro, Raphael Veiga), standard <code>Positional_LightGBM</code> exhibits an <strong>Elite Outlier Bias of <code>-0.624 pts</code></strong> (and <code>Market_Price</code> has <code>-1.004 pts</code>) because tree leaf regularization shrinks extreme leaves back toward the positional mean (~3.1 pts).</li>
          <li>When we instead train <code>Residual_XGBoost</code> to predict $\\Delta y_{{i,t}} = y_{{i,t}} - \\hat{{\\mu}}_{{\\text{{dual\\_EB}}, i, t-1}}$ and add the player's Dual-Memory Empirical Bayes baseline back, the Elite Outlier Bias drops to <strong><code>-0.085 pts</code></strong> (virtually zero bias!).</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <h2>4. Positional Predictability Breakdown: Why Coach (`TEC`) is the Most Predictable Position!</h2>
      <p class="lead">Comparing Spearman rank correlation and Top-15% Quantile PR-AUC across all 6 positions.</p>
      {fig_to_html(fig4, include_js=False)}
      <div class="pill-box">
        <div class="pill-title">💊 Learnable Pill: Law of Large Numbers on the Pitch (`TEC` vs `GOL`)</div>
        In your prompt, you wondered if a Coach (<code>TEC</code>) model would be a "horoscope". Look at the chart above: <strong>Coach (<code>TEC</code>) actually has the HIGHEST Spearman Rank Correlation ($\rho = 0.43$) and HIGHEST PR-AUC ($0.33$) of any position in Cartola!</strong> Why? Because a single striker's score is a noisy sample of $N=1$ player, whereas the Coach's score is the arithmetic average of $N=11$ players. By the Law of Large Numbers, idiosyncratic individual variance cancels out at the team level ($\sigma_{{\\bar{{X}}}} = \\sigma / \\sqrt{{11}}$), leaving pure team-vs-opponent expected value! Meanwhile, Goalkeepers (<code>GOL</code>) are the hardest to rank ($\rho \\approx 0.14$) because a keeper faces a paradox: weak defenses concede goals (losing $+5.0$ SG), while strong defenses face zero shots (earning zero $+1.3$ <code>DE</code> saves).
      </div>
    </div>
  </div>
</body>
</html>
"""
    out_file = REPORTS_DIR / "03_model_arena_and_backtest.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"-> Saved {out_file}")


def build_interactive_academy():
    print("Building Interactive HTML Academy & Quizzes: `academy/index.html`...")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CLF Interactive Data Science Academy, Deep-Dive Tabs & Quizzes</title>
  {SHARED_CSS}
  <style>
    .tab-bar {{
      display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 26px;
      border-bottom: 2px solid var(--border); padding-bottom: 14px;
    }}
    .tab-btn {{
      background: var(--card); border: 1px solid var(--border); color: var(--muted);
      padding: 12px 20px; border-radius: 10px; cursor: pointer; font-weight: 800;
      font-size: 0.94rem; transition: all 0.15s ease; display: flex; align-items: center; gap: 8px;
    }}
    .tab-btn:hover {{ color: #fff; background: var(--card-alt); border-color: var(--accent); }}
    .tab-btn.active {{
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(168, 85, 247, 0.18));
      border-color: var(--accent); color: #fff;
      box-shadow: 0 4px 18px rgba(56, 189, 248, 0.18);
    }}
    .tab-pane {{ display: none; }}
    .tab-pane.active {{ display: block; }}
    .quiz-card {{
      background: #0e1525; border: 1px solid var(--border); border-radius: 12px;
      padding: 20px 24px; margin-top: 18px;
    }}
    .quiz-q {{ font-weight: 700; color: #fff; font-size: 1rem; margin-bottom: 12px; }}
    .quiz-opt {{
      display: block; width: 100%; text-align: left; background: var(--card);
      border: 1px solid var(--border); color: var(--text); padding: 11px 16px;
      border-radius: 8px; margin-bottom: 8px; cursor: pointer; font-size: 0.9rem;
      transition: all 0.15s ease;
    }}
    .quiz-opt:hover {{ background: var(--card-alt); border-color: var(--accent); }}
    .quiz-opt.correct {{ background: rgba(16, 185, 129, 0.18); border-color: #10B981; color: #fff; font-weight: 700; }}
    .quiz-opt.wrong {{ background: rgba(239, 68, 68, 0.18); border-color: #EF4444; color: #fca5a5; }}
    .quiz-feedback {{
      display: none; margin-top: 12px; padding: 12px 16px; border-radius: 8px;
      font-size: 0.88rem; background: var(--card-alt); border-left: 4px solid var(--accent);
    }}
    .score-banner {{
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(56, 189, 248, 0.15));
      border: 1px solid var(--green); border-radius: 12px; padding: 16px 24px;
      display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;
    }}
    .sim-box {{
      background: #090d16; border: 1px solid var(--border); border-radius: 12px;
      padding: 20px; margin: 16px 0;
    }}
    .slider-row {{
      display: grid; grid-template-columns: 240px 1fr 90px; align-items: center;
      gap: 14px; margin-bottom: 12px; font-size: 0.9rem;
    }}
    input[type=range] {{ width: 100%; accent-color: #38bdf8; }}
    .arch-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 18px; margin-top: 16px;
    }}
    .arch-card {{
      background: #0e1525; border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px;
    }}
    .arch-card h3 {{ margin: 0 0 6px; font-size: 1.08rem; color: #fff; display: flex; justify-content: space-between; align-items: center; }}
    .arch-tag {{ font-size: 0.72rem; padding: 2px 8px; border-radius: 999px; background: var(--card-alt); color: var(--accent); border: 1px solid var(--border); }}
    .search-input {{
      width: 100%; padding: 12px 16px; border-radius: 10px; border: 1px solid var(--border);
      background: #090d16; color: #fff; font-size: 0.95rem; margin-bottom: 14px;
    }}
    .cat-badge {{
      display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.74rem;
      font-weight: 700; background: var(--card-alt); color: var(--accent);
    }}
  </style>
</head>
<body>
  {nav_bar('academy', is_academy=True)}
  <div class="container">
    <div class="hero">
      <h1>🎓 CLF Data Science Academy & 3 Interactive Deep-Dive Tabs</h1>
      <p>Switch between the 4 interactive tabs below: (1) The 6-Module Masterclass & Quizzes, (2) <strong>Deep-Dive 1: Dual-Memory Empirical Bayes</strong>, (3) <strong>Deep-Dive 2: Complete 98-Feature Encyclopedia</strong>, and (4) <strong>Deep-Dive 3: Architecture of All 10 Models & What Is a Meta-Ensemble</strong>.</p>
    </div>

    <!-- INTERACTIVE TAB BAR -->
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('tab-masterclass', this)">📚 Tab 1: Masterclass & 6 Quizzes</button>
      <button class="tab-btn" onclick="switchTab('tab-eb-deepdive', this)">🧬 Tab 2: Dual-Memory Empirical Bayes Deep-Dive</button>
      <button class="tab-btn" onclick="switchTab('tab-feature-catalog', this)">🗂️ Tab 3: Full 98-Feature Encyclopedia</button>
      <button class="tab-btn" onclick="switchTab('tab-model-arch', this)">🏛️ Tab 4: All 10 Model Architectures & Meta-Ensemble</button>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 1: MASTERCLASS & 6 QUIZZES                                        -->
    <!-- ===================================================================== -->
    <div id="tab-masterclass" class="tab-pane active">
      <div class="score-banner">
        <div>
          <strong>🧠 Your Interactive Quiz Progress:</strong>
          <span id="quiz-progress-text" style="margin-left: 8px; color: var(--accent);">0 / 6 questions answered correctly</span>
        </div>
        <button onclick="resetAllQuizzes()" style="background: var(--card-alt); color: #fff; border: 1px solid var(--border); padding: 6px 14px; border-radius: 8px; cursor: pointer;">Reset Quizzes</button>
      </div>

      <!-- MODULE 1 -->
      <div class="section-card" id="mod1">
        <h2>⚡ Module 1: Relational Vectorization (Why 359,385 Rows Took 1.15 Seconds)</h2>
        <p class="lead">How to avoid the O(N²) row-iteration trap when computing team-level and opponent-conceded lag windows.</p>
        <p>When you first attempted to compute <em>"Average assists conceded by the opposing team in the past 10 games"</em> on the player table (N = 359,385 rows), computing it row-by-row meant scanning the table tens of thousands of times. Even worse, 25 players in the same match share the <strong>exact same team and opponent</strong>!</p>

        <div class="pill-box">
          <div class="pill-title">💊 The 3-Level Relational Vectorization Pattern</div>
          <ol>
            <li><strong>Level 1 (Entity Projection):</strong> Aggregate player scouts up to the <code>(season, round, team_id)</code> grain first. That reduces 359,385 player rows to just <strong>8,629 team-match rows</strong> (a 41&times; reduction!).</li>
            <li><strong>Level 2 (Self-Join Mirroring):</strong> Join the 8,629-row table with itself on <code>(season, round, team_id == opp_id)</code>. What Team A <em>scored</em> in Match M is by definition what Team B <em>conceded</em> in Match M! We do the exact same thing at the <code>(season, round, opp_id, position_id)</code> grain to get <strong>Positional Opponent Vulnerability</strong>.</li>
            <li><strong>Level 3 (Shifted Groupby Rolling + Broadcast Join):</strong> Apply <code>.groupby(["team_id", "season"]).shift(1).rolling(W).mean()</code> on the tiny 8,629-row table (takes ~40 milliseconds), then left-join the result back onto the 131,660 active player appearances.</li>
          </ol>
        </div>

        <pre><code># Vectorized Shifted Rolling Pattern (Zero Future Leakage Guaranteed by .shift(1))
g_team = cube.groupby(["team_id", "season"], sort=False)
shifted_conceded_A = g_team["conceded_A_in_match"].shift(1)
cube["opp_roll10_conceded_A"] = (
    shifted_conceded_A.groupby([cube["team_id"], cube["season"]])
    .rolling(10, min_periods=1).mean()
    .reset_index(level=[0, 1], drop=True)
)</code></pre>

        <div class="quiz-card" data-qid="1">
          <div class="quiz-q">Quiz 1: Why MUST `.shift(1)` be applied inside `.groupby(...)` BEFORE `.rolling(10).mean()` when building pre-match features for Round t?</div>
          <button class="quiz-opt" onclick="answerQuiz(1, this, false)">A) Because `.rolling(10)` in Pandas does not work without `.shift(1)` first.</button>
          <button class="quiz-opt" onclick="answerQuiz(1, this, true)">B) Because without `.shift(1)`, the rolling window at Round t includes Round t's own outcome (target leakage), making validation metrics look artificially high while failing in real-world prediction.</button>
          <button class="quiz-opt" onclick="answerQuiz(1, this, false)">C) Because `.shift(1)` speeds up Cython execution by 10x.</button>
          <div class="quiz-feedback" id="fb-1">
            <strong>Spot on!</strong> In Pandas, <code>df.rolling(10).mean()</code> at row t averages rows [t-9 ... t]. Including row t leaks the match outcome into the features. Applying <code>.groupby(...).shift(1)</code> first shifts the series so row t only sees [t-10 ... t-1].
          </div>
        </div>
      </div>

      <!-- MODULE 2 -->
      <div class="section-card" id="mod2">
        <h2>🌉 Module 2: The Cross-Season Gap & Dual-Memory Empirical Bayes Shrinkage</h2>
        <p class="lead">Solving the Round 38 → Round 1 dilemma mathematically with James-Stein / Empirical Bayes Shrinkage.</p>
        <p>You raised a classic domain dilemma: <em>"Round 38 of 2019 and Round 1 of 2020 look 1 round apart in an integer index, but months passed with transfers and roster overhauls."</em></p>

        <div class="formula-block">
          &mu;&#770;_offseason(i, S) = [ n_(i,S-1) &middot; (0.80 &middot; y&#772;_(i,S-1) + 0.20 &middot; &mu;_pos) + k_0 &middot; &mu;&#770;_career(i) ] / (n_(i,S-1) + k_0)
          <br><br>
          &mu;&#770;_dual_EB(i, S, r) = [ n_(i,S,r) &middot; y&#772;_(i,S,1..r-1) + k_S &middot; &mu;&#770;_offseason(i, S) ] / (n_(i,S,r) + k_S)
        </div>

        <!-- INTERACTIVE EMPIRICAL BAYES CALCULATOR -->
        <div class="sim-box">
          <div style="font-weight: 800; color: #fff; margin-bottom: 12px;">🎛️ Interactive Empirical Bayes Simulator — Try It Yourself!</div>
          <div class="slider-row">
            <label>Positional Population Average (&mu;<sub>pos</sub>):</label>
            <input type="range" id="eb_pos" min="2.0" max="5.0" step="0.1" value="3.1" oninput="updateEB()">
            <span id="eb_pos_val">3.10 pts</span>
          </div>
          <div class="slider-row">
            <label>Player Off-Season Prior (&mu;&#770;<sub>offseason</sub>):</label>
            <input type="range" id="eb_prior" min="1.0" max="10.0" step="0.1" value="6.8" oninput="updateEB()">
            <span id="eb_prior_val">6.80 pts</span>
          </div>
          <div class="slider-row">
            <label>Matches Played in Current Season (n<sub>S</sub>):</label>
            <input type="range" id="eb_ns" min="0" max="37" step="1" value="2" oninput="updateEB()">
            <span id="eb_ns_val">2 games</span>
          </div>
          <div class="slider-row">
            <label>Current Season Sample Mean (y&#772;<sub>S</sub>):</label>
            <input type="range" id="eb_ys" min="-2.0" max="15.0" step="0.2" value="1.5" oninput="updateEB()">
            <span id="eb_ys_val">1.50 pts</span>
          </div>
          <div class="slider-row">
            <label>Prior Strength Parameter (k<sub>S</sub>):</label>
            <input type="range" id="eb_k" min="1.0" max="15.0" step="0.5" value="5.0" oninput="updateEB()">
            <span id="eb_k_val">k = 5.0</span>
          </div>
          <div style="margin-top: 14px; padding: 12px 16px; background: var(--card); border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
            <span><strong>Resulting Dual-Memory EB Baseline (&mu;&#770;<sub>dual_EB</sub>):</strong></span>
            <span id="eb_result" style="font-size: 1.4rem; font-weight: 800; color: var(--gold);">5.29 pts</span>
          </div>
        </div>

        <div class="quiz-card" data-qid="2">
          <div class="quiz-q">Quiz 2: Suppose Arrascaeta enters Round 1 of 2025 (n<sub>S</sub> = 0 games played in 2025 so far) with an off-season shrunk prior of 6.80 pts. What weight does the Dual-Memory Empirical Bayes formula assign to his off-season prior in Round 1, and when does current-season form reach exactly 50% weight (assuming k<sub>S</sub> = 5)?</div>
          <button class="quiz-opt" onclick="answerQuiz(2, this, true)">A) 100% weight on the off-season prior in Round 1 (n_S = 0), and 50%/50% weight after he has played n_S = 5 matches in the new season.</button>
          <button class="quiz-opt" onclick="answerQuiz(2, this, false)">B) 0% weight in Round 1, and 50% weight in Round 19.</button>
          <button class="quiz-opt" onclick="answerQuiz(2, this, false)">C) 50% weight in Round 1 regardless of n_S.</button>
          <div class="quiz-feedback" id="fb-2">
            <strong>Exactly right!</strong> When n<sub>S</sub> = 0, 0 / (0 + 5) = 0% on the current season and 5 / (0 + 5) = 100% on the off-season prior. When n<sub>S</sub> = k<sub>S</sub> = 5, both weights equal 5 / (5 + 5) = 50%.
          </div>
        </div>
      </div>

      <!-- MODULE 3 -->
      <div class="section-card" id="mod3">
        <h2>🎯 Module 3: Solving the "Arrascaeta Problem" (Why Tree Models Anchor Outliers)</h2>
        <p class="lead">Why decision trees pull geniuses toward the mean—and how Residual Target Learning fixes it.</p>
        <p>A Gradient Boosted Decision Tree (LightGBM / XGBoost) predicts a leaf value using regularized leaf averages:</p>
        <div class="formula-block">
          w_leaf* = - ( &sum;_(i &in; leaf) g_i ) / ( &sum;_(i &in; leaf) h_i + &lambda; )
        </div>
        <p>Because every leaf requires at least <code>min_child_samples</code> (e.g., 60 observations) and adds L2 penalty &lambda;, an elite player who averages 7.5 points gets grouped in leaves with merely "good" players averaging 4.8 points. As we saw in Report 03, standard <code>Positional_LightGBM</code> under-predicted elite stars by <strong>-0.624 points per game</strong>!</p>

        <div class="pill-box gold">
          <div class="pill-title">💊 The Residual Learning Decomposition</div>
          Instead of training the tree on y<sub>i,t</sub>, we decompose the target into:
          <div class="formula-block">y_(i,t) = &mu;&#770;_dual_EB(i, t-1) [Individual Genius Anchor] + &Delta;y_(i,t) [Matchup &amp; Form Residual]</div>
          The tree only predicts &Delta;y<sub>i,t</sub> (whether this fixture against Opponent B at home is +1.2 pts better or -0.8 pts worse than the player's own baseline). When we add &Delta;y&#770;<sub>i,t</sub> back to Arrascaeta's baseline, our Elite Outlier Bias drops from <code>-0.624</code> to <strong><code>-0.085</code></strong>!
        </div>

        <div class="quiz-card" data-qid="3">
          <div class="quiz-q">Quiz 3: Why did our `Residual_XGBoost` model reduce under-prediction bias on elite stars by 86% compared to standard `Positional_LightGBM`?</div>
          <button class="quiz-opt" onclick="answerQuiz(3, this, false)">A) Because XGBoost uses deeper trees than LightGBM.</button>
          <button class="quiz-opt" onclick="answerQuiz(3, this, true)">B) Because tree leaf regularization shrinks predictions toward zero; when predicting the residual &Delta;y = y - &mu;&#770;_player, shrinking toward zero shrinks toward the player's own elite baseline rather than the population mean!</button>
          <button class="quiz-opt" onclick="answerQuiz(3, this, false)">C) Because Residual XGBoost ignores opponent features completely.</button>
          <div class="quiz-feedback" id="fb-3">
            <strong>Brilliant insight!</strong> Shrinking a raw prediction toward zero pulls a star toward the global mean. Shrinking a <em>residual</em> toward zero pulls the prediction toward &mu;&#770;<sub>player</sub> + 0 = &mu;&#770;<sub>player</sub>, preserving the player's full talent baseline!
          </div>
        </div>
      </div>

      <!-- MODULE 4 -->
      <div class="section-card" id="mod4">
        <h2>🧩 Module 4: Positional Physics, Poisson Clean Sheets & The Coach (`TEC`) Law of Large Numbers</h2>
        <p class="lead">Why Coaches are NOT a horoscope, and how Poisson goal modeling powers Defender (`GOL/LAT/ZAG`) predictions.</p>
        <p>For Goalkeepers and Defenders, the +5.0 point Clean Sheet (<code>SG</code>) bonus is binary: concede 0 goals &rarr; +5.0 pts; concede &ge; 1 goal &rarr; 0.0 pts. If Opponent B's expected goals in the fixture is &lambda;<sub>opp</sub>, then under a Poisson scoring process:</p>
        <div class="formula-block">
          P(Opponent Goals = k) = (&lambda;_opp^k &middot; e^(-&lambda;_opp)) / k!   &rArr;   P(SG = 1) = P(k = 0) = e^(-&lambda;_opp)
        </div>

        <div class="quiz-card" data-qid="4">
          <div class="quiz-q">Quiz 4: In our 2024–2025 out-of-sample evaluation, which position achieved the HIGHEST Spearman rank correlation (&rho; = 0.43) and WHY?</div>
          <button class="quiz-opt" onclick="answerQuiz(4, this, false)">A) Goalkeeper (`GOL`), because saves are easy to predict.</button>
          <button class="quiz-opt" onclick="answerQuiz(4, this, true)">B) Coach (`TEC`), because a Coach's score is the average of all 11 fielded players, so individual player noise cancels out by the Law of Large Numbers (&sigma; / &radic;11).</button>
          <button class="quiz-opt" onclick="answerQuiz(4, this, false)">C) Striker (`ATA`), because goals happen in every match.</button>
          <div class="quiz-feedback" id="fb-4">
            <strong>Spot on!</strong> While an individual player's single-match score has huge Poisson variance, averaging 11 players divides the standard error by &radic;11 &approx; 3.3&times;, making team-level expected points highly predictable!
          </div>
        </div>
      </div>

      <!-- MODULE 5 -->
      <div class="section-card" id="mod5">
        <h2>🕵️ Module 5: Forensic Data Auditing & Pre-Match Market Price Recovery</h2>
        <p class="lead">Two subtle data traps that separate production-grade ML from leaky or corrupted notebooks.</p>
        <ul>
          <li><strong>Trap 1 — Post-Match Price Leakage:</strong> In Cartola's raw CSVs, the column <code>preco_num</code> (`price`) recorded in Round r's snapshot is the player's price <strong>after</strong> Round r finishes, and <code>variacao_num</code> (`price_variation`) is how much their price changed as a direct mathematical function of Round r's points! If you feed raw <code>price</code> into a model predicting Round r, you leak the target. However, <code>pre_match_price = price - price_variation</code> recovers the exact pre-match price at Round r-1 with 0% leakage!</li>
          <li><strong>Trap 2 — Missing Snapshot Cumulative Bleed:</strong> When 2020 <code>rodada-37.csv</code> was missing upstream, differencing <code>rodada-38.csv</code> caused 493 players in Round 38 to receive multi-round cumulative scout totals (averaging 53.4 pts!). Auditing <code>|base_norm_score - points| &gt; 6.5</code> caught and repaired all 1,115 anomalous rows across the 12-year dataset.</li>
        </ul>

        <div class="quiz-card" data-qid="5">
          <div class="quiz-q">Quiz 5: Suppose a player enters Round 10 costing C$ 12.00, scores 15.0 points in Round 10, and appreciates by +C$ 2.50 so the Round 10 CSV snapshot records `price = 14.50` and `price_variation = +2.50`. What happens if you include `price` or `price_variation` directly as features to predict Round 10?</div>
          <button class="quiz-opt" onclick="answerQuiz(5, this, true)">A) Severe target leakage! The model learns that positive `price_variation` means a high score in Round 10. We must instead compute `pre_match_price = price - price_variation = 12.00` and shift `price_variation` by 1 round (`prev_price_variation`).</button>
          <button class="quiz-opt" onclick="answerQuiz(5, this, false)">B) Nothing bad happens; Cartola sets prices before the match.</button>
          <button class="quiz-opt" onclick="answerQuiz(5, this, false)">C) Price has zero correlation with player quality.</button>
          <div class="quiz-feedback" id="fb-5">
            <strong>100% Correct!</strong> Subtracting <code>price_variation</code> from <code>price</code> exactly reconstructs the C$ 12.00 pre-round price that managers saw before the market closed!
          </div>
        </div>
      </div>

      <!-- MODULE 6 -->
      <div class="section-card" id="mod6">
        <h2>📏 Module 6: Multi-Lens Evaluation (`RMSE`, `MAE`, `R²`, `PR-AUC` on Regression, `Spearman ρ`, & The 38-Round Model League)</h2>
        <p class="lead">A complete dictionary of every evaluation metric in our Leaderboard—and how a regression model is evaluated with `PR-AUC` and a simulated Cartola Championship.</p>

        <div class="pill-box">
          <div class="pill-title">💊 1. MAE vs. RMSE: Why Are Both Around ~2.6 to ~3.5 Points, and Why Is RMSE Always Higher?</div>
          <div class="formula-block">
            MAE = (1 / N) &sum; |y_i - y&#770;_i| &approx; 2.65 pts &nbsp;&nbsp;|&nbsp;&nbsp; RMSE = &radic;[ (1 / N) &sum; (y_i - y&#770;_i)^2 ] &approx; 3.51 pts
          </div>
          <ul>
            <li><strong>MAE (Mean Absolute Error = <code>2.65 pts</code>):</strong> It is the literal average error in Cartola points (&plusmn;2.65 pts). Every point of error carries equal linear weight.</li>
            <li><strong>RMSE (Root Mean Squared Error = <code>3.51 pts</code>):</strong> Squares each error <em>before</em> averaging, then takes the square root so the unit returns to Cartola points. Squaring punishes large outliers quadratically: missing a player by <strong>10 points</strong> contributes 100 squared units—equal to <strong>one hundred 1-point errors</strong>!</li>
          </ul>
        </div>

        <div class="pill-box gold">
          <div class="pill-title">💊 2. How Can We Compute `PR-AUC` on a Regression Model?!</div>
          <ol>
            <li><strong>Step 1 — Define the Binary "Mito" Event (Y<sub>mito</sub> &in; {{0, 1}}):</strong> Either Fixed (<code>y_i &ge; 8.0 pts</code>) or Dynamic Positional Quantile (<code>y_i</code> in the Top 15% of that position in that round).</li>
            <li><strong>Step 2 — Use the Continuous Regression Prediction y&#770;<sub>i</sub> as a Ranking Score:</strong> Because <code>PR-AUC</code> sweeps a threshold c from highest prediction to lowest, it only cares about the relative ordering of predictions!</li>
          </ol>
        </div>

        <div class="quiz-card" data-qid="6">
          <div class="quiz-q">Quiz 6: Why did you rightly suggest tracking a Quantile-based Mito target (`target_mito_q85`) alongside a fixed threshold (`score >= 8.0`) for `PR-AUC`, and how does a regression model compute `PR-AUC`?</div>
          <button class="quiz-opt" onclick="answerQuiz(6, this, false)">A) Because `PR-AUC` cannot be computed on fixed thresholds.</button>
          <button class="quiz-opt" onclick="answerQuiz(6, this, true)">B) By treating the continuous predicted score y&#770;_i as a threshold-free ranking score! And using a per-round positional 85th percentile dynamically adapts to round difficulty and positional scoring scale (unlike a fixed 8.0 cutoff).</button>
          <button class="quiz-opt" onclick="answerQuiz(6, this, false)">C) Because quantiles eliminate all variance in soccer.</button>
          <div class="quiz-feedback" id="fb-6">
            <strong>Nailed it!</strong> Because `PR-AUC` sweeps a threshold from highest prediction to lowest, any continuous regression prediction y&#770;<sub>i</sub> serves directly as a ranking score—and positional round quantiles normalize across both positional baselines and low/high-scoring rounds!
          </div>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 2: DEEP-DIVE 1 — DUAL-MEMORY EMPIRICAL BAYES                      -->
    <!-- ===================================================================== -->
    <div id="tab-eb-deepdive" class="tab-pane">
      <div class="section-card">
        <h2>🧬 Deep-Dive 1: What Exactly Is Dual-Memory Empirical Bayes (And Why We Invented It for CLF)</h2>
        <p class="lead">How to combine a player's multi-year career talent, their previous season's level, and their current-season form without falling into the "Small Sample Trap" or the "Off-Season Amnesia Trap".</p>

        <div class="pill-box">
          <div class="pill-title">💊 1. First Principles: What Is Empirical Bayes (James-Stein Shrinkage)?</div>
          <p>Imagine two midfielders after <strong>Round 1</strong> of the championship:</p>
          <ul>
            <li><strong>Player A (Random Reserve):</strong> Has played 1 game in his life, scored a lucky deflected goal in Round 1 &rarr; <strong>Sample Average = 11.0 pts</strong>.</li>
            <li><strong>Player B (Arrascaeta):</strong> Has played 180 career Serie A games averaging 6.8 pts, but had a quiet Round 1 &rarr; <strong>Sample Average = 1.2 pts</strong>.</li>
          </ul>
          <p>If you use a naive sample average (or <code>Rolling-3</code>), your model believes Player A (11.0) is 9&times; better than Arrascaeta (1.2)! <strong>Empirical Bayes (EB)</strong> solves this by computing a <strong>credibility-weighted average</strong> between the noisy sample mean y&#772; (based on n games) and a stable Prior Anchor &mu;<sub>0</sub> (with virtual sample size k, called the "shrinkage strength"):</p>
          <div class="formula-block">
            &mu;&#770;_EB = [ n / (n + k) ] &middot; y&#772;_sample  +  [ k / (n + k) ] &middot; &mu;_0
          </div>
          <p>Look at the weights: when n = 0 (before Round 1), the credibility weight on the sample is <code>0 / (0 + k) = 0%</code> and the weight on the prior &mu;<sub>0</sub> is <code>100%</code>. When n = k, both get <code>50%</code> weight. When n &gg; k (e.g. Round 30), the sample gets nearly <code>100%</code> weight!</p>
        </div>

        <div class="pill-box gold">
          <div class="pill-title">💊 2. Why Standard Empirical Bayes Isn't Enough for Soccer: The "Dual-Memory" Hierarchy</div>
          <p>Standard Empirical Bayes only has <strong>one</strong> memory level: it shrinks a player toward the global positional mean (&mu;<sub>pos</sub> &approx; 3.10 pts for MEI). But in soccer, time operates on <strong>two distinct biological and tactical clocks</strong>:</p>
          <ol>
            <li><strong>Long-Term Career Memory (Slow Clock — Talent Baseline):</strong> Over 2014–2025, a player builds a career track record across n<sub>career</sub> appearances. We first shrink their career expanding average toward their positional mean (&mu;<sub>pos</sub>) with k<sub>career</sub> = 8 games:
              <div class="formula-block">
                Level 1 (Career Prior): &nbsp; &mu;&#770;_career(i, t) = [ n_career &middot; y&#772;_career(i, 1..t-1) + 8 &middot; &mu;_pos ] / (n_career + 8)
              </div>
              <em>What this does:</em> A rookie with n<sub>career</sub> = 0 starts at &mu;<sub>pos</sub> (3.10 pts). A veteran with 100 games starts at their true career average.
            </li>
            <li><strong>Off-Season Transition Bridge (The December &rarr; April Reset):</strong> Between Round 38 of Season S-1 and Round 1 of Season S, teams change coaches, players age, and squads are rebuilt. However, 80% of a player's skill from last season carries over! So at the start of Season S, we form the <strong>Off-Season Prior</strong> by shrinking last season's mean (decayed by 20% toward &mu;<sub>pos</sub>) against their Career Prior (with k<sub>off</sub> = 6 games):
              <div class="formula-block">
                Level 2 (Off-Season Bridge): &nbsp; &mu;&#770;_offseason(i, S) = [ n_(S-1) &middot; (0.80 &middot; y&#772;_(S-1) + 0.20 &middot; &mu;_pos) + 6 &middot; &mu;&#770;_career ] / (n_(S-1) + 6)
              </div>
              <em>What this does:</em> If a player had a breakout 30-game season last year (n<sub>S-1</sub> = 30), their Off-Season Prior heavily reflects last year's breakout (with a healthy 20% regression-to-the-mean discount). If they were injured all last year (n<sub>S-1</sub> = 0), it falls back to their multi-year Career Prior!
            </li>
            <li><strong>Within-Season Active Form (Fast Clock — Current Championship):</strong> Finally, as Round r = 1, 2, ..., 38 unfolds in Season S, we blend their current-season average y&#772;<sub>S</sub> (over n<sub>S</sub> games played this season) with their Off-Season Bridge using k<sub>S</sub> = 5 games:
              <div class="formula-block">
                Level 3 (Dual-Memory EB Score): &nbsp; &mu;&#770;_dual_EB(i, S, r) = [ n_S &middot; y&#772;_S(1..r-1) + 5 &middot; &mu;&#770;_offseason(i, S) ] / (n_S + 5)
              </div>
            </li>
          </ol>
        </div>

        <div class="pill-box green">
          <div class="pill-title">💊 3. Concrete Numerical Walkthrough: Arrascaeta Across Seasons</div>
          <table class="data-table">
            <thead>
              <tr>
                <th>Moment in Time</th>
                <th>Career Prior (&mu;&#770;<sub>career</sub>)</th>
                <th>Off-Season Prior (&mu;&#770;<sub>offseason</sub>)</th>
                <th>Current Season Games (n<sub>S</sub>) &amp; Mean (y&#772;<sub>S</sub>)</th>
                <th>Weight on Current Season [n<sub>S</sub>/(n<sub>S</sub>+5)]</th>
                <th>Final <code>player_dual_eb_score</code></th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Round 1 of Season</strong> (Season kickoff)</td>
                <td>6.60 pts</td>
                <td>6.35 pts</td>
                <td>n<sub>S</sub> = 0 games (No data yet)</td>
                <td>0 / (0 + 5) = <strong>0.0%</strong></td>
                <td><strong>6.35 pts</strong> (100% Off-Season Prior!)</td>
              </tr>
              <tr>
                <td><strong>Round 3 of Season</strong> (Slow start: 2 quiet games)</td>
                <td>6.55 pts</td>
                <td>6.35 pts</td>
                <td>n<sub>S</sub> = 2 games, y&#772;<sub>S</sub> = 2.10 pts</td>
                <td>2 / (2 + 5) = <strong>28.6%</strong></td>
                <td><strong>5.14 pts</strong> (Protects against panic!)</td>
              </tr>
              <tr>
                <td><strong>Round 6 of Season</strong> (Back in form)</td>
                <td>6.58 pts</td>
                <td>6.35 pts</td>
                <td>n<sub>S</sub> = 5 games, y&#772;<sub>S</sub> = 6.90 pts</td>
                <td>5 / (5 + 5) = <strong>50.0%</strong></td>
                <td><strong>6.63 pts</strong> (50/50 balanced blend)</td>
              </tr>
              <tr>
                <td><strong>Round 25 of Season</strong> (Mid-late championship)</td>
                <td>6.70 pts</td>
                <td>6.35 pts</td>
                <td>n<sub>S</sub> = 20 games, y&#772;<sub>S</sub> = 7.40 pts</td>
                <td>20 / (20 + 5) = <strong>80.0%</strong></td>
                <td><strong>7.19 pts</strong> (Current season dominates!)</td>
              </tr>
            </tbody>
          </table>
          <p style="margin-top: 12px;">And from this, we also derive <code>player_genius_delta = player_dual_eb_score - pos_mean_anchor</code> (for Arrascaeta: <code>7.19 - 3.10 = +4.09 pts</code>), which tells every model exactly how many points per game this individual player sits above an average player in their position!</p>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 3: DEEP-DIVE 2 — FULL 98-FEATURE ENCYCLOPEDIA                     -->
    <!-- ===================================================================== -->
    <div id="tab-feature-catalog" class="tab-pane">
      <div class="section-card">
        <h2>🗂️ Deep-Dive 2: Complete Encyclopedia of All 98 Engineered Pre-Match Features</h2>
        <p class="lead">Every single feature fed into our models in `clf_feature_store.parquet`, grouped by domain family, with its exact mathematical formula and tactical meaning. Use the search bar below to filter by keyword (e.g. `price`, `opp_`, `eb_`, `DS`, `floor`, `poisson`).</p>

        <input type="text" id="feat-search" class="search-input" placeholder="🔍 Filter the 98 features by name, scout, or tactical concept (e.g., 'conceded', 'floor', 'price', 'poisson', 'genius')..." oninput="filterFeatures()">

        <table class="data-table" id="feat-table">
          <thead>
            <tr>
              <th style="width: 50px;">#</th>
              <th style="width: 290px;">Feature Name</th>
              <th style="width: 170px;">Domain Family</th>
              <th style="width: 250px;">Mathematical Formula (at Round t)</th>
              <th>Exact Tactical &amp; Statistical Meaning</th>
            </tr>
          </thead>
          <tbody>
            <!-- FAMILY 1: MATCH & AVAILABILITY CONTEXT (7) -->
            <tr><td>1</td><td><code>is_home</code></td><td><span class="cat-badge">1. Match Context</span></td><td>1 if Home, 0 if Away</td><td>Home pitch indicator. Home players average +4.06 pts vs +2.88 pts away (+41% boost).</td></tr>
            <tr><td>2</td><td><code>position_id</code></td><td><span class="cat-badge">1. Match Context</span></td><td>Integer &in; {{1..6}}</td><td>1=GOL, 2=LAT, 3=ZAG, 4=MEI, 5=ATA, 6=TEC. Defines scoring rules (e.g. SG eligibility).</td></tr>
            <tr><td>3</td><td><code>pos_mean_anchor</code></td><td><span class="cat-badge">1. Match Context</span></td><td>E[y | position_id]</td><td>Global historical average score for the player's position (e.g., ~3.10 for MEI, ~4.40 for GOL).</td></tr>
            <tr><td>4</td><td><code>career_games_prior</code></td><td><span class="cat-badge">1. Match Context</span></td><td>n<sub>career</sub>(i, 1..t-1)</td><td>Total active Serie A appearances by this player prior to Round t (measures veteran experience).</td></tr>
            <tr><td>5</td><td><code>season_games_prior</code></td><td><span class="cat-badge">1. Match Context</span></td><td>n<sub>S</sub>(i, 1..r-1)</td><td>Number of active appearances by this player in the current season prior to Round r (0 in Round 1).</td></tr>
            <tr><td>6</td><td><code>rounds_since_last_played</code></td><td><span class="cat-badge">1. Match Context</span></td><td>min(20, r - r<sub>prev_played</sub>)</td><td>Rounds elapsed since the player last took the pitch in the current season. Equals 1 for regular starters; &gt;3 signals injury return, suspension, or bench rotation risk.</td></tr>
            <tr><td>7</td><td><code>played_last_round</code></td><td><span class="cat-badge">1. Match Context</span></td><td>I(rounds_since_last_played == 1)</td><td>Binary flag: 1.0 if the player played in the immediately preceding round (starter continuity).</td></tr>

            <!-- FAMILY 2: PRE-MATCH MARKET WISDOM (4) -->
            <tr><td>8</td><td><code>pre_match_price</code></td><td><span class="cat-badge">2. Market Wisdom</span></td><td>price<sub>t</sub> - price_variation<sub>t</sub></td><td>Exact Cartola market price (C$) <strong>before</strong> Round t begins. Captures crowd &amp; pricing-algorithm assessment of player value with zero post-match leakage.</td></tr>
            <tr><td>9</td><td><code>prev_price_variation</code></td><td><span class="cat-badge">2. Market Wisdom</span></td><td>price_variation<sub>t-1</sub></td><td>How much the player's price appreciated or depreciated in their previous appearance (market momentum).</td></tr>
            <tr><td>10</td><td><code>price_rel_to_team</code></td><td><span class="cat-badge">2. Market Wisdom</span></td><td>pre_match_price / mean_team_price<sub>t</sub></td><td>Player's price divided by the average price of their teammates in that match (&gt;1.3 identifies the team's star focal point).</td></tr>
            <tr><td>11</td><td><code>price_rel_to_pos</code></td><td><span class="cat-badge">2. Market Wisdom</span></td><td>pre_match_price / mean_pos_price<sub>t</sub></td><td>Player's price divided by the round average for their position (&gt;1.5 identifies premium positional tier).</td></tr>

            <!-- FAMILY 3: PLAYER EMPIRICAL BAYES & DUAL-MEMORY PRIORS (8) -->
            <tr><td>12</td><td><code>player_career_eb_score</code></td><td><span class="cat-badge">3. Dual-Memory EB</span></td><td>(n<sub>c</sub>&middot;y&#772;<sub>c</sub> + 8&middot;&mu;<sub>pos</sub>)/(n<sub>c</sub>+8)</td><td>Empirical Bayes career total score baseline up to t-1, shrunk toward positional mean with k=8.</td></tr>
            <tr><td>13</td><td><code>player_career_eb_floor</code></td><td><span class="cat-badge">3. Dual-Memory EB</span></td><td>(n<sub>c</sub>&middot;f&#772;<sub>c</sub> + 8&middot;&mu;<sub>floor</sub>)/(n<sub>c</sub>+8)</td><td>Empirical Bayes career <strong>Floor Score</strong> (points excluding Goals, Assists, and Clean Sheets: pure DS, FD, FS, DE consistency).</td></tr>
            <tr><td>14</td><td><code>player_career_eb_ga</code></td><td><span class="cat-badge">3. Dual-Memory EB</span></td><td>(n<sub>c</sub>&middot;ga&#772;<sub>c</sub> + 8&middot;&mu;<sub>ga</sub>)/(n<sub>c</sub>+8)</td><td>Empirical Bayes career Goal+Assist points (8&middot;G + 5&middot;A) per match up to t-1.</td></tr>
            <tr><td>15</td><td><code>player_offseason_prior</code></td><td><span class="cat-badge">3. Dual-Memory EB</span></td><td>[n<sub>S-1</sub>(0.8y&#772;<sub>S-1</sub>+0.2&mu;<sub>pos</sub>)+6&mu;&#770;<sub>c</sub>]/(n<sub>S-1</sub>+6)</td><td>Off-season bridge prior entering Season S: blends 80%-retained previous season average with career EB prior.</td></tr>
            <tr><td>16</td><td><code>player_dual_eb_score</code></td><td><span class="cat-badge">3. Dual-Memory EB</span></td><td>(n<sub>S</sub>&middot;y&#772;<sub>S</sub> + 5&middot;&mu;&#770;<sub>off</sub>)/(n<sub>S</sub>+5)</td><td><strong>Core Talent Baseline:</strong> Dual-Memory Empirical Bayes total score estimate combining career, previous season, and current season form.</td></tr>
            <tr><td>17</td><td><code>player_dual_eb_floor</code></td><td><span class="cat-badge">3. Dual-Memory EB</span></td><td>(n<sub>S</sub>&middot;f&#772;<sub>S</sub> + 5&middot;&mu;&#770;<sub>off_floor</sub>)/(n<sub>S</sub>+5)</td><td>Dual-Memory Empirical Bayes baseline for pure non-goal/non-SG action floor points.</td></tr>
            <tr><td>18</td><td><code>player_genius_delta</code></td><td><span class="cat-badge">3. Dual-Memory EB</span></td><td>player_dual_eb_score - pos_mean_anchor</td><td><strong>The "Arrascaeta Feature":</strong> How many points per game this player sits above (+) or below (-) the global positional average.</td></tr>
            <tr><td>19</td><td><code>player_floor_delta</code></td><td><span class="cat-badge">3. Dual-Memory EB</span></td><td>player_dual_eb_floor - &mu;<sub>pos_floor</sub></td><td>How many base action floor points (tackles, shots, fouls drawn, saves) this player generates above their positional average.</td></tr>

            <!-- FAMILY 4: PLAYER ROLLING WINDOWS & EWMA FORM (34) -->
            <tr><td>20</td><td><code>player_roll3_base_norm_score</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(y<sub>t-3..t-1</sub>)</td><td>Short-term 3-appearance rolling average of total points (blended with EB prior when n<sub>S</sub> &lt; 3).</td></tr>
            <tr><td>21</td><td><code>player_roll5_base_norm_score</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(y<sub>t-5..t-1</sub>)</td><td>Medium-term 5-appearance rolling average of total points.</td></tr>
            <tr><td>22</td><td><code>player_roll10_base_norm_score</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(y<sub>t-10..t-1</sub>)</td><td>10-appearance rolling average of total points.</td></tr>
            <tr><td>23</td><td><code>player_ewma_base_norm_score</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(y<sub>1..t-1</sub>, halflife=3.5)</td><td>Exponentially Weighted Moving Average of total points (recent games weighted exponentially higher, halflife = 3.5 games).</td></tr>
            <tr><td>24</td><td><code>player_roll3_floor_Component</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(floor<sub>t-3..t-1</sub>)</td><td>3-game rolling average of Floor Score (base_norm_score minus G, A, SG).</td></tr>
            <tr><td>25</td><td><code>player_roll5_floor_Component</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(floor<sub>t-5..t-1</sub>)</td><td>5-game rolling average of Floor Score (captures reliable "regularidade" players).</td></tr>
            <tr><td>26</td><td><code>player_roll10_floor_Component</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(floor<sub>t-10..t-1</sub>)</td><td>10-game rolling average of Floor Score.</td></tr>
            <tr><td>27</td><td><code>player_ewma_floor_Component</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(floor<sub>1..t-1</sub>, hl=3.5)</td><td>Exponentially weighted moving average of Floor Score.</td></tr>
            <tr><td>28</td><td><code>player_roll3_ga_Component</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(8G+5A)<sub>t-3..t-1</sub></td><td>3-game rolling average of points coming specifically from Goals (+8) and Assists (+5).</td></tr>
            <tr><td>29</td><td><code>player_roll5_ga_Component</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(8G+5A)<sub>t-5..t-1</sub></td><td>5-game rolling average of Goal + Assist points.</td></tr>
            <tr><td>30</td><td><code>player_roll10_ga_Component</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(8G+5A)<sub>t-10..t-1</sub></td><td>10-game rolling average of Goal + Assist points.</td></tr>
            <tr><td>31</td><td><code>player_ewma_ga_Component</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(8G+5A, hl=3.5)</td><td>Exponentially weighted moving average of Goal + Assist points.</td></tr>
            <tr><td>32</td><td><code>player_roll5_G</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(G<sub>t-5..t-1</sub>)</td><td>Player's average Goals (G) per game over their last 5 appearances.</td></tr>
            <tr><td>33</td><td><code>player_ewma_G</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(G, hl=3.5)</td><td>Player's EWMA of Goals (G) per game.</td></tr>
            <tr><td>34</td><td><code>player_roll5_A</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(A<sub>t-5..t-1</sub>)</td><td>Player's average Assists (A) per game over their last 5 appearances.</td></tr>
            <tr><td>35</td><td><code>player_ewma_A</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(A, hl=3.5)</td><td>Player's EWMA of Assists (A) per game.</td></tr>
            <tr><td>36</td><td><code>player_roll3_DS</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(DS<sub>t-3..t-1</sub>)</td><td>3-game rolling average of Tackles / Ball Recoveries (Desarmes, DS = +1.0 pt, harmonized with legacy RB).</td></tr>
            <tr><td>37</td><td><code>player_roll5_DS</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(DS<sub>t-5..t-1</sub>)</td><td>5-game rolling average of Tackles (DS). Crucial for defensive midfielders (volantes) and fullbacks.</td></tr>
            <tr><td>38</td><td><code>player_roll10_DS</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(DS<sub>t-10..t-1</sub>)</td><td>10-game rolling average of Tackles (DS).</td></tr>
            <tr><td>39</td><td><code>player_ewma_DS</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(DS, hl=3.5)</td><td>EWMA of Tackles (DS) per game.</td></tr>
            <tr><td>40</td><td><code>player_roll3_FD</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(FD<sub>t-3..t-1</sub>)</td><td>3-game rolling average of Shots Saved on Target (Finalizações Defendidas, FD = +1.2 pts).</td></tr>
            <tr><td>41</td><td><code>player_roll5_FD</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(FD<sub>t-5..t-1</sub>)</td><td>5-game rolling average of Shots on Target (FD)—strongest proxy for player xG (Expected Goals).</td></tr>
            <tr><td>42</td><td><code>player_roll10_FD</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(FD<sub>t-10..t-1</sub>)</td><td>10-game rolling average of Shots on Target (FD).</td></tr>
            <tr><td>43</td><td><code>player_ewma_FD</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(FD, hl=3.5)</td><td>EWMA of Shots on Target (FD).</td></tr>
            <tr><td>44</td><td><code>player_roll5_FF</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(FF<sub>t-5..t-1</sub>)</td><td>5-game rolling average of Shots Off Target (Finalizações para Fora, FF = +0.8 pts).</td></tr>
            <tr><td>45</td><td><code>player_ewma_FF</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(FF, hl=3.5)</td><td>EWMA of Shots Off Target (FF). Together with FD, measures total shot volume.</td></tr>
            <tr><td>46</td><td><code>player_roll5_FS</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(FS<sub>t-5..t-1</sub>)</td><td>5-game rolling average of Fouls Suffered (Faltas Sofridas, FS = +0.5 pts). High for dribblers.</td></tr>
            <tr><td>47</td><td><code>player_ewma_FS</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(FS, hl=3.5)</td><td>EWMA of Fouls Suffered (FS).</td></tr>
            <tr><td>48</td><td><code>player_roll5_DE</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(DE<sub>t-5..t-1</sub>)</td><td>5-game rolling average of Goalkeeper Saves (Defesas, DE = +1.3 pts, harmonized with legacy DD).</td></tr>
            <tr><td>49</td><td><code>player_ewma_DE</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(DE, hl=3.5)</td><td>EWMA of Goalkeeper Saves (DE). Top feature for GOL models.</td></tr>
            <tr><td>50</td><td><code>player_roll5_FC</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(FC<sub>t-5..t-1</sub>)</td><td>5-game rolling average of Fouls Committed (Faltas Cometidas, FC = -0.3 pts). Measures disciplinary drag.</td></tr>
            <tr><td>51</td><td><code>player_ewma_FC</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(FC, hl=3.5)</td><td>EWMA of Fouls Committed (FC).</td></tr>
            <tr><td>52</td><td><code>player_roll5_SG</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>Shrunk Mean(SG<sub>t-5..t-1</sub>)</td><td>5-game rolling Clean Sheet rate experienced by this player.</td></tr>
            <tr><td>53</td><td><code>player_ewma_SG</code></td><td><span class="cat-badge">4. Player Form Lags</span></td><td>EWMA(SG, hl=3.5)</td><td>EWMA of Clean Sheets experienced by this player.</td></tr>

            <!-- FAMILY 5: PLAYER VOLATILITY & QUANTILE ENVELOPES (3) -->
            <tr><td>54</td><td><code>player_roll10_std</code></td><td><span class="cat-badge">5. Risk &amp; Ceiling</span></td><td>StdDev(y<sub>t-10..t-1</sub>)</td><td>10-game rolling standard deviation of points. Distinguishes "boom-or-bust" strikers from steady floor midfielders.</td></tr>
            <tr><td>55</td><td><code>player_roll10_ceiling_q85</code></td><td><span class="cat-badge">5. Risk &amp; Ceiling</span></td><td>Quantile<sub>0.85</sub>(y<sub>t-10..t-1</sub>)</td><td>85th percentile score over the player's last 10 games (measures explosive "Mito" ceiling).</td></tr>
            <tr><td>56</td><td><code>player_roll10_floor_q25</code></td><td><span class="cat-badge">5. Risk &amp; Ceiling</span></td><td>Quantile<sub>0.25</sub>(y<sub>t-10..t-1</sub>)</td><td>25th percentile score over the player's last 10 games (measures downside safety floor).</td></tr>

            <!-- FAMILY 6: OWN TEAM FORM & STRENGTH (12) -->
            <tr><td>57</td><td><code>eb_gf</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team Dual-Memory EB Goals For</td><td>Team's shrunk season baseline of goals scored per match up to t-1.</td></tr>
            <tr><td>58</td><td><code>eb_ga</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team Dual-Memory EB Goals Against</td><td>Team's shrunk season baseline of goals conceded per match up to t-1.</td></tr>
            <tr><td>59</td><td><code>eb_clean_sheet</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team Dual-Memory EB Clean Sheet Rate</td><td>Team's shrunk probability of keeping a Clean Sheet (0 goals conceded) up to t-1.</td></tr>
            <tr><td>60</td><td><code>eb_team_pts_mean</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team EB Mean Fantasy Pts / Player</td><td>Average Cartola points per fielded player on this team (also equals expected Coach TEC baseline!).</td></tr>
            <tr><td>61</td><td><code>eb_team_floor_mean</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team EB Mean Floor Pts / Player</td><td>Team's baseline non-goal/non-SG fantasy points per player (tactical volume).</td></tr>
            <tr><td>62</td><td><code>eb_team_avg_price</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team EB Mean Player Price (C$)</td><td>Average market price of the club's starting lineup (proxy for overall squad budget/tier).</td></tr>
            <tr><td>63</td><td><code>roll5_gf</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team 5-Match Rolling Goals For</td><td>Goals scored per game by the player's club over the last 5 rounds.</td></tr>
            <tr><td>64</td><td><code>roll5_ga</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team 5-Match Rolling Goals Against</td><td>Goals conceded per game by the player's club over the last 5 rounds.</td></tr>
            <tr><td>65</td><td><code>roll5_clean_sheet</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team 5-Match Rolling Clean Sheet %</td><td>Fraction of last 5 matches in which the player's club kept a Clean Sheet.</td></tr>
            <tr><td>66</td><td><code>roll5_team_pts_mean</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team 5-Match Rolling Fantasy Pts</td><td>Average Cartola points per player on the club over the last 5 matches.</td></tr>
            <tr><td>67</td><td><code>roll5_team_floor_mean</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team 5-Match Rolling Floor Pts</td><td>Average Floor points per player on the club over the last 5 matches.</td></tr>
            <tr><td>68</td><td><code>roll10_team_pts_mean</code></td><td><span class="cat-badge">6. Own Team Strength</span></td><td>Team 10-Match Rolling Fantasy Pts</td><td>Average Cartola points per player on the club over the last 10 matches.</td></tr>

            <!-- FAMILY 7: OPPONENT TEAM CONCEDED & VULNERABILITY (21) -->
            <tr><td>69</td><td><code>opp_eb_gf</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Goals Scored / Match</td><td>Opponent's attacking baseline (threatens our GOL/LAT/ZAG Clean Sheet, but generates GK saves).</td></tr>
            <tr><td>70</td><td><code>opp_eb_ga</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Goals Conceded / Match</td><td>Opponent's defensive leakiness baseline (boosts our MEI/ATA/LAT goal &amp; assist expectation).</td></tr>
            <tr><td>71</td><td><code>opp_eb_clean_sheet</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Clean Sheet Rate</td><td>How often the opponent shuts out teams (high value means tough defense for our attackers).</td></tr>
            <tr><td>72</td><td><code>opp_eb_team_pts_mean</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Own Fantasy Pts</td><td>Overall strength of the opposing squad.</td></tr>
            <tr><td>73</td><td><code>opp_eb_conceded_pts</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Conceded Fantasy Pts</td><td>Season shrunk average of Cartola points the opponent <strong>allows to opposing players</strong>.</td></tr>
            <tr><td>74</td><td><code>opp_eb_conceded_floor</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Conceded Floor Pts</td><td>Season shrunk average of Floor points (DS, FD, FS, DE) the opponent gives up per player.</td></tr>
            <tr><td>75</td><td><code>opp_eb_conceded_G</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Conceded Goals</td><td>Season shrunk average of Cartola G scouts conceded by the opponent per match.</td></tr>
            <tr><td>76</td><td><code>opp_eb_conceded_A</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Conceded Assists</td><td>Season shrunk average of Assists (A) conceded by the opponent per match.</td></tr>
            <tr><td>77</td><td><code>opp_eb_conceded_DS</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Conceded Tackles (DS)</td><td>How many tackles (DS) the opponent allows opposing teams to rack up per match (dribble-heavy teams lose the ball more!).</td></tr>
            <tr><td>78</td><td><code>opp_eb_conceded_FD</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent EB Conceded Shots on Target</td><td>How many shots on target (FD) the opponent allows per match.</td></tr>
            <tr><td>79</td><td><code>opp_roll5_gf</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 5-Match Goals Scored</td><td>Opponent's short-term attacking form over their last 5 games.</td></tr>
            <tr><td>80</td><td><code>opp_roll5_ga</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 5-Match Goals Conceded</td><td>Opponent's short-term defensive leakiness over their last 5 games.</td></tr>
            <tr><td>81</td><td><code>opp_roll5_conceded_pts</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 5-Match Conceded Pts</td><td>Fantasy points per player conceded by the opponent in their last 5 games.</td></tr>
            <tr><td>82</td><td><code>opp_roll10_conceded_pts</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 10-Match Conceded Pts</td><td>Fantasy points per player conceded by the opponent in their last 10 games.</td></tr>
            <tr><td>83</td><td><code>opp_roll5_conceded_floor</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 5-Match Conceded Floor</td><td>Floor points per player conceded by the opponent in their last 5 games.</td></tr>
            <tr><td>84</td><td><code>opp_roll5_conceded_G</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 5-Match Conceded G</td><td>Goals per match conceded by the opponent in their last 5 games.</td></tr>
            <tr><td>85</td><td><code>opp_roll5_conceded_A</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 5-Match Conceded Assists</td><td>Assists per match conceded by the opponent in their last 5 games.</td></tr>
            <tr><td>86</td><td><code>opp_roll10_conceded_A</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 10-Match Conceded Assists</td><td><strong>Your Exact Prompt Feature:</strong> Average assists conceded by the opposing team in their past 10 games!</td></tr>
            <tr><td>87</td><td><code>opp_roll5_conceded_DS</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 5-Match Conceded DS</td><td>Total tackles (DS) conceded by the opponent in their last 5 games.</td></tr>
            <tr><td>88</td><td><code>opp_roll10_conceded_DS</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 10-Match Conceded DS</td><td>Total tackles (DS) conceded by the opponent in their last 10 games.</td></tr>
            <tr><td>89</td><td><code>opp_roll5_conceded_FD</code></td><td><span class="cat-badge">7. Opponent Conceded</span></td><td>Opponent 5-Match Conceded FD</td><td>Total shots on target (FD) conceded by the opponent in their last 5 games.</td></tr>

            <!-- FAMILY 8: POSITIONAL OPPONENT VULNERABILITY (7) -->
            <tr><td>90</td><td><code>opp_pos_conceded_roll5_pos_pts_mean</code></td><td><span class="cat-badge">8. Positional Matchup</span></td><td>Opp 5-Game Conceded Pts to Position P</td><td>How many points the opponent concedes <strong>specifically to the player's exact position</strong> (e.g. opposing LATs) over the last 5 games.</td></tr>
            <tr><td>91</td><td><code>opp_pos_conceded_roll10_pos_pts_mean</code></td><td><span class="cat-badge">8. Positional Matchup</span></td><td>Opp 10-Game Conceded Pts to Position P</td><td>10-game rolling average of points conceded by the opponent specifically to the player's position.</td></tr>
            <tr><td>92</td><td><code>opp_pos_conceded_roll5_pos_floor_mean</code></td><td><span class="cat-badge">8. Positional Matchup</span></td><td>Opp 5-Game Conceded Floor to Pos P</td><td>5-game rolling average of Floor points conceded by the opponent specifically to Position P.</td></tr>
            <tr><td>93</td><td><code>opp_pos_conceded_roll10_pos_floor_mean</code></td><td><span class="cat-badge">8. Positional Matchup</span></td><td>Opp 10-Game Conceded Floor to Pos P</td><td>10-game rolling average of Floor points conceded by the opponent specifically to Position P.</td></tr>
            <tr><td>94</td><td><code>opp_pos_conceded_roll5_pos_DS_mean</code></td><td><span class="cat-badge">8. Positional Matchup</span></td><td>Opp 5-Game Conceded DS to Pos P</td><td>How many tackles (DS) per player the opponent yields specifically to Position P over the last 5 games.</td></tr>
            <tr><td>95</td><td><code>opp_pos_conceded_roll5_pos_FD_mean</code></td><td><span class="cat-badge">8. Positional Matchup</span></td><td>Opp 5-Game Conceded FD to Pos P</td><td>How many shots on target (FD) per player the opponent yields specifically to Position P over the last 5 games.</td></tr>
            <tr><td>96</td><td><code>opp_pos_conceded_roll5_pos_DE_mean</code></td><td><span class="cat-badge">8. Positional Matchup</span></td><td>Opp 5-Game Conceded Saves (DE) to GK</td><td>How many saves (DE) the opponent's attack forces opposing Goalkeepers to make over the last 5 games!</td></tr>

            <!-- FAMILY 9: POISSON PHYSICS & NON-LINEAR INTERACTIONS (8 -> Total 98 in FEATURE_COLS + 6 context) -->
            <tr><td>97</td><td><code>team_vs_opp_xg_diff</code></td><td><span class="cat-badge">9. Physics &amp; Interactions</span></td><td>Expected Goal Supremacy (&lambda;<sub>team</sub> - &lambda;<sub>opp</sub>)</td><td>Net expected goal superiority of the player's club over the opponent in this fixture.</td></tr>
            <tr><td>98</td><td><code>expected_team_gf</code></td><td><span class="cat-badge">9. Physics &amp; Interactions</span></td><td>0.55&middot;Roll5_Attack + 0.45&middot;EB_Attack &plusmn; 0.22</td><td>Expected goals (&lambda;<sub>team</sub>) scored by the player's team in this fixture, adjusted for home/away pitch.</td></tr>
            <tr><td>99</td><td><code>expected_opp_gf</code></td><td><span class="cat-badge">9. Physics &amp; Interactions</span></td><td>0.55&middot;Roll5_OppAtt + 0.45&middot;EB_OppAtt &mp; 0.22</td><td>Expected goals (&lambda;<sub>opp</sub>) scored by the opponent in this fixture.</td></tr>
            <tr><td>100</td><td><code>poisson_sg_prob</code></td><td><span class="cat-badge">9. Physics &amp; Interactions</span></td><td>exp(-expected_opp_gf)</td><td><strong>Poisson Clean Sheet Probability:</strong> Exact mathematical probability P(Opponent Goals = 0) = e<sup>-&lambda;_opp</sup>.</td></tr>
            <tr><td>101</td><td><code>expected_sg_points</code></td><td><span class="cat-badge">9. Physics &amp; Interactions</span></td><td>5.0 &middot; poisson_sg_prob &middot; I(pos &in; {{1,2,3}})</td><td>Expected fantasy points from the Clean Sheet bonus (+5.0 &times; P(SG=1)) for GOL, LAT, ZAG; 0 for MEI/ATA/TEC.</td></tr>
            <tr><td>102</td><td><code>ix_player_ga_vs_opp_conceded</code></td><td><span class="cat-badge">9. Physics &amp; Interactions</span></td><td>player_roll5_ga &times; (opp_roll5_conceded_G / 1.25)</td><td>Multiplicative interaction: Striker/Midfielder goal-involvement form &times; Opponent recent goals conceded.</td></tr>
            <tr><td>103</td><td><code>ix_player_ds_vs_opp_pos_ds</code></td><td><span class="cat-badge">9. Physics &amp; Interactions</span></td><td>player_roll5_DS &times; opp_pos_conceded_DS</td><td>Multiplicative interaction: Player's tackle rate &times; Opponent's positional tackle vulnerability.</td></tr>
            <tr><td>104</td><td><code>ix_gk_de_vs_opp_attack</code></td><td><span class="cat-badge">9. Physics &amp; Interactions</span></td><td>I(pos==1) &middot; player_roll5_DE &times; expected_opp_gf</td><td>Goalkeeper Save Synergy: Shot-stopping keeper rate &times; Opponent attacking volume.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 4: DEEP-DIVE 3 — ALL 10 MODEL ARCHITECTURES & META-ENSEMBLE       -->
    <!-- ===================================================================== -->
    <div id="tab-model-arch" class="tab-pane">
      <div class="section-card">
        <h2>🏛️ Deep-Dive 3: Architecture of All 10 Models & What Exactly Is a `META_ENSEMBLE`?</h2>
        <p class="lead">How each of our 10 models thinks, its mathematical equation, its strengths & blind spots, and how Stacking / Meta-Ensembling combines them to win the 38-round championship.</p>

        <div class="pill-box gold">
          <div class="pill-title">👑 First: What Is the `CLF_Meta_Ensemble` (And Why Did It Win Both RMSE and the 2025 Lineup Championship)?</div>
          <p>In Machine Learning, every model family has an <strong>inductive bias</strong> (a structural blind spot):</p>
          <ul>
            <li><strong>Standard Trees (`Positional_LightGBM`)</strong> are great at sharp conditional rules (<em>"If home AND opponent concedes &gt;1.8 goals..."</em>), but their leaf regularization pulls elite outliers like Arrascaeta down toward the positional mean.</li>
            <li><strong>Residual Trees (`Residual_XGBoost`)</strong> preserve Arrascaeta's elite baseline (`-0.085` bias!), but can slightly over-trust a star who is currently in a slump.</li>
            <li><strong>Decomposed Physical Models (`Decomposed_Hybrid`)</strong> explicitly separate Clean Sheet probability $P(\\text{{SG}}=1)$, Floor tackles, and Goal spikes, capturing bimodal defender physics.</li>
            <li><strong>Neural Networks (`Neural_MLP`)</strong> learn smooth, continuous linear + non-linear hyperplanes across all 98 standardized features simultaneously, excelling at global ranking (`Spearman &rho; = 0.3227`).</li>
          </ul>
          <p>A <strong>Meta-Ensemble (Level-2 Stacked Generalizer)</strong> takes the out-of-fold predictions of these 4 diverse specialist architectures and combines them via validation-tuned convex weights ($\sum w_m = 1$):</p>
          <div class="formula-block">
            y&#770;_Meta_Ensemble = 0.30 &middot; y&#770;_Positional_LGBM + 0.30 &middot; y&#770;_Residual_XGB + 0.25 &middot; y&#770;_Decomposed_Hybrid + 0.15 &middot; y&#770;_Neural_MLP
          </div>
          <p><strong>Why does the ensemble beat every single individual model in RMSE (`3.5152`), Spearman (`0.3287`), and 2025 Season Points (`3,034.9 pts`)?</strong> Because by the <em>Diversity Prediction Theorem</em>, for any convex combination of models with weights $w_m$:</p>
          <div class="formula-block">
            (Ensemble Squared Error) = &sum; w_m &middot; (Individual Model Error_m)  &minus;  &sum; w_m &middot; (y&#770;_m &minus; y&#770;_Ensemble)^2  [The Diversity Bonus!]
          </div>
          <p>Notice the minus sign on the second term: whenever two uncorrelated models make errors in <em>different</em> directions (e.g., LightGBM under-predicts Arrascaeta by `-0.6` while Residual XGBoost over-predicts by `+0.2`), their errors cancel out mathematically!</p>
        </div>

        <h3 style="color: #fff; margin-top: 28px;">🔍 Architectural Blueprint of All 10 Models in Our Zoo</h3>
        <div class="arch-grid">
          <div class="arch-card">
            <h3>1. Naive_Roll5 <span class="arch-tag">Heuristic Baseline</span></h3>
            <p><strong>Equation:</strong> <code>y&#770;_(i,t) = player_roll5_base_norm_score</code></p>
            <p><strong>How it works:</strong> Predicts that a player will score whatever they averaged in their last 5 appearances (shrunk toward prior in Rounds 1–4).</p>
            <p><strong>Blind spot:</strong> Completely ignores whether the next match is Home vs. Away or against Fortaleza vs. Cuiabá, and chases noisy 5-game goal streaks (Test RMSE: <code>3.7808</code>, R²: <code>-0.038</code>).</p>
          </div>

          <div class="arch-card">
            <h3>2. Market_Price_XI <span class="arch-tag">Crowd Wisdom Baseline</span></h3>
            <p><strong>Equation:</strong> <code>y&#770;_(i,t) = &beta;_0 + &beta;_1&middot;pre_match_price + &beta;_2&middot;price_rel_team + &beta;_3&middot;price_rel_pos + &beta;_4&middot;is_home</code></p>
            <p><strong>How it works:</strong> Uses a per-position Ridge regression on the player's pre-match Cartola price and home indicator ("Time dos Mais Caros").</p>
            <p><strong>Blind spot:</strong> Cartola price reflects past season-to-date average, not specific opponent tactical vulnerabilities (Test RMSE: <code>3.5753</code>).</p>
          </div>

          <div class="arch-card">
            <h3>3. Dual_Memory_EB <span class="arch-tag">Bayesian Baseline</span></h3>
            <p><strong>Equation:</strong> <code>y&#770;_(i,t) = player_dual_eb_score</code></p>
            <p><strong>How it works:</strong> Our 3-level hierarchical Empirical Bayes estimator blending Career Prior (k=8), Off-Season Retention Bridge (80% retention, k=6), and Current-Season Expanding Mean (k=5).</p>
            <p><strong>Strength:</strong> Beats Naive_Roll5 by +119 points in 2025 without even looking at opponent or home/away context!</p>
          </div>

          <div class="arch-card">
            <h3>4. Position_Ridge <span class="arch-tag">Linear L2 Family</span></h3>
            <p><strong>Equation:</strong> <code>y&#770;_(i,t) = w_p^T &middot; z_(i,t) + b_p</code> &nbsp; (L2 penalty <code>&alpha; = 250</code> per position <code>p &in; {{1..6}}</code>)</p>
            <p><strong>How it works:</strong> Fits 6 separate standardized Ridge regressions on all 98 features. Linear models extrapolate smoothly when an elite player has a 3-sigma feature value.</p>
            <p><strong>Strength:</strong> Best Test MAE (<code>2.6415</code>) and #1 in the 2024 Lineup Backtest (<code>2,723.2 pts</code>)!</p>
          </div>

          <div class="arch-card">
            <h3>5. Global_LightGBM <span class="arch-tag">Pooled GBDT</span></h3>
            <p><strong>Architecture:</strong> Single LightGBM ensemble (<code>num_leaves=31</code>, <code>lr=0.03</code>, <code>min_child_samples=80</code>, L1=2, L2=5) trained across all 98,651 rows pooled together.</p>
            <p><strong>Strength &amp; Weakness:</strong> Learns shared macro patterns across the whole dataset, but must spend tree splits separating GOL, LAT, ZAG, MEI, ATA, and TEC rules.</p>
          </div>

          <div class="arch-card">
            <h3>6. Positional_LightGBM <span class="arch-tag">6 Specialist GBDTs</span></h3>
            <p><strong>Architecture:</strong> 6 completely independent LightGBM models—one each for <code>GOL</code>, <code>LAT</code>, <code>ZAG</code>, <code>MEI</code>, <code>ATA</code>, and <code>TEC</code> (<code>num_leaves=23</code> for outfield, <code>15</code> for GOL/TEC).</p>
            <p><strong>Strength:</strong> Allows `GOL` trees to split on saves (`DE`) and `LAT/ZAG` trees to split on `poisson_sg_prob` without polluting `MEI/ATA` splits.</p>
          </div>

          <div class="arch-card">
            <h3>7. Residual_XGBoost <span class="arch-tag">De-Meaned Genius GBDT</span></h3>
            <p><strong>Equation:</strong> <code>y&#770;_(i,t) = player_dual_eb_score + XGB_pos(X_(i,t))</code> where XGB predicts <code>&Delta;y = y - player_dual_eb_score</code>.</p>
            <p><strong>Why it's special:</strong> Solves the "Arrascaeta Anchoring Problem"! Because the tree predicts only the fixture deviation &Delta;y around the player's own talent baseline, L2 leaf regularization shrinks toward the player's baseline rather than the population mean (Elite Bias: <code>-0.085 pts</code>!).</p>
          </div>

          <div class="arch-card">
            <h3>8. Decomposed_Hybrid <span class="arch-tag">3-Component Physics GBDT</span></h3>
            <p><strong>Equation:</strong> <code>y&#770;_(i,t) = 5.0 &middot; P&#770;_LGBM(SG=1) &middot; I(Def) + y&#770;_LGBM(Floor) + y&#770;_LGBM(8G + 5A)</code></p>
            <p><strong>Why it's special:</strong> Instead of predicting total score as one opaque number, it trains: (a) a binary classifier for Clean Sheet probability, (b) a regressor for continuous tackle/shot Floor, and (c) a regressor for Goal/Assist spikes, then sums them by Cartola's exact physical laws!</p>
          </div>

          <div class="arch-card">
            <h3>9. Neural_MLP <span class="arch-tag">Deep Tabular Network</span></h3>
            <p><strong>Architecture:</strong> 3-Hidden-Layer Perceptron (<code>128 &rarr; 64 &rarr; 32</code> ReLU neurons, Batch=512, L2 <code>&alpha;=2.5</code>, Adam optimizer) with positional one-hot embeddings + 98 z-scored features, predicting residual over EB baseline.</p>
            <p><strong>Strength:</strong> Smoothly interpolates across high-dimensional feature space, achieving #2 overall Test RMSE (<code>3.5186</code>) and #3 in 2025 Lineup Points (<code>3,000.4 pts</code>).</p>
          </div>

          <div class="arch-card" style="border-color: var(--green); background: rgba(16, 185, 129, 0.06);">
            <h3>10. CLF_Meta_Ensemble <span class="arch-tag" style="color: #10B981; border-color: #10B981;">Champion Stack</span></h3>
            <p><strong>Equation:</strong> <code>0.30&middot;Pos_LGBM + 0.30&middot;Res_XGB + 0.25&middot;Decomp_Hybrid + 0.15&middot;Neural_MLP</code></p>
            <p><strong>Why it wins:</strong> Combines specialist positional trees, residual star-preserving boosting, decomposed clean-sheet/floor/goal physics, and smooth neural representations into a single superior predictor.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function switchTab(tabId, btn) {{
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btn.classList.add('active');
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}
    function filterFeatures() {{
      const q = document.getElementById('feat-search').value.toLowerCase();
      const rows = document.querySelectorAll('#feat-table tbody tr');
      rows.forEach(r => {{
        const txt = r.textContent.toLowerCase();
        r.style.display = txt.includes(q) ? '' : 'none';
      }});
    }}
    const quizState = {{}};
    function answerQuiz(qid, btn, isCorrect) {{
      const card = document.querySelector(`.quiz-card[data-qid="${{qid}}"]`);
      const buttons = card.querySelectorAll('.quiz-opt');
      buttons.forEach(b => b.classList.remove('correct', 'wrong'));
      if (isCorrect) {{
        btn.classList.add('correct');
        quizState[qid] = true;
      }} else {{
        btn.classList.add('wrong');
        quizState[qid] = false;
      }}
      const fb = document.getElementById(`fb-${{qid}}`);
      fb.style.display = 'block';
      updateScoreBanner();
    }}
    function updateScoreBanner() {{
      const correctCount = Object.values(quizState).filter(Boolean).length;
      document.getElementById('quiz-progress-text').textContent =
        `${{correctCount}} / 6 questions answered correctly` + (correctCount === 6 ? ' 🏆 Mastered!' : '');
    }}
    function resetAllQuizzes() {{
      Object.keys(quizState).forEach(k => delete quizState[k]);
      document.querySelectorAll('.quiz-opt').forEach(b => b.classList.remove('correct', 'wrong'));
      document.querySelectorAll('.quiz-feedback').forEach(f => f.style.display = 'none');
      updateScoreBanner();
    }}
    function updateEB() {{
      const pos = parseFloat(document.getElementById('eb_pos').value);
      const prior = parseFloat(document.getElementById('eb_prior').value);
      const ns = parseFloat(document.getElementById('eb_ns').value);
      const ys = parseFloat(document.getElementById('eb_ys').value);
      const k = parseFloat(document.getElementById('eb_k').value);

      document.getElementById('eb_pos_val').textContent = pos.toFixed(2) + ' pts';
      document.getElementById('eb_prior_val').textContent = prior.toFixed(2) + ' pts';
      document.getElementById('eb_ns_val').textContent = ns + ' games';
      document.getElementById('eb_ys_val').textContent = ys.toFixed(2) + ' pts';
      document.getElementById('eb_k_val').textContent = 'k = ' + k.toFixed(1);

      const eb = (ns * ys + k * prior) / (ns + k);
      document.getElementById('eb_result').textContent = eb.toFixed(2) + ' pts (Season Weight: ' + Math.round(100 * ns / (ns + k)) + '%)';
    }}
  </script>
</body>
</html>
"""
    out_file = ACADEMY_DIR / "index.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"-> Saved {out_file}")


def main():
    print("=" * 72)
    print("CLF Phase 5: Generating Standalone Plotly Reports & Interactive Academy")
    print("=" * 72)

    df = pd.read_parquet(DATA_DIR / "clf_feature_store.parquet")
    eval_df = pd.read_parquet(DATA_DIR / "predictions_oof.parquet")
    overall_metrics = pd.read_parquet(DATA_DIR / "metrics_overall.parquet")
    pos_metrics = pd.read_parquet(DATA_DIR / "metrics_by_position.parquet")
    bt_df = pd.read_parquet(DATA_DIR / "lineup_backtest.parquet")
    with open(DATA_DIR / "model_artifacts.json", "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    build_report_01_eda(df)
    build_report_02_feature_lab(df, artifacts)
    build_report_03_arena(eval_df, overall_metrics, pos_metrics, bt_df)
    build_interactive_academy()

    print("=" * 72)
    print("All 4 Standalone HTML Reports & Academy Generated Successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()
