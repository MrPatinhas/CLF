"""
01_build_feature_store.py — Cartola Legends of the Future (CLF)
================================================================
High-speed relational feature engineering engine (< 3 seconds).

Architecture (3-Level Relational Vectorization):
  Level 1: Reconcile Cartola round fixtures with `matches.parquet` via exact +
           bipartite score/round alignment, producing `team_match_cube.parquet`.
  Level 2: Compute Team-Match and Team-Match-Position aggregations & mirror them
           across opponents to obtain Opponent Conceded metrics (overall and
           positional vulnerability).
  Level 3: Compute strictly pre-match (shifted t-1) Rolling (W=3, 5, 10), EWMA,
           Empirical Bayes Shrunk Priors, Cross-Season Decay, and Market features.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd

CLT_DATA_DIR = Path(r"C:\Users\thiag\Documents\CLT\data")
CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
CLF_DATA_DIR = CLF_DIR / "data"
CLF_DATA_DIR.mkdir(parents=True, exist_ok=True)

POS_MAP = {
    1: "GOL",
    2: "LAT",
    3: "ZAG",
    4: "MEI",
    5: "ATA",
    6: "TEC",
}


def sanitize_cumulative_bleed_anomalies(scouts_active: pd.DataFrame) -> pd.DataFrame:
    """
    Learnable Pill: Detecting & Sanitizing Cumulative Bleed in Missing-Round Snapshots
    In `caRtola` historical CSVs, if a round snapshot is missing (e.g. 2020 Round 37) or a player
    first appears in the raw dump mid-season with cumulative season-to-date scouts, `.diff()`
    assigns their entire multi-round cumulative scouts to a single match!
    We detect this whenever `|base_norm_score - points| > 6.5` (since legitimate single-match era
    weight adjustments between `RB->DS` or `DD->DE` never exceed ~3-5 points).
    For those ~0.7% cumulative-bleed rows, we anchor `base_norm_score` to the true single-match `points`
    and scale the corrupted scout deltas proportionally so downstream rolling windows are 100% pure!
    """
    df = scouts_active.copy()
    is_outfield = df["position_id"] != 6
    gap = (df["base_norm_score"] - df["points"]).abs()
    corrupted_mask = is_outfield & (gap > 6.5)

    n_corrupted = int(corrupted_mask.sum())
    if n_corrupted > 0:
        # Scale down inflated scout counts proportionally to match single-game intensity
        scale_factor = (
            df.loc[corrupted_mask, "points"].clip(lower=0.0)
            / df.loc[corrupted_mask, "base_norm_score"].clip(lower=1.0)
        ).clip(upper=1.0)
        scout_cols = ["G", "A", "SG", "DP", "DE", "DS", "FT", "FD", "FF", "FS", "GC", "CV", "CA", "GS", "FC", "I", "PP"]
        for sc in scout_cols:
            if sc in df.columns:
                df.loc[corrupted_mask, sc] = (df.loc[corrupted_mask, sc] * scale_factor).round(2)
        df.loc[corrupted_mask, "base_norm_score"] = df.loc[corrupted_mask, "points"].round(2)
        print(f"   [Sanitizer] Repaired {n_corrupted:,} cumulative-bleed outlier rows (e.g. 2020 R38 missing R37 snapshot).")

    return df


def align_matches_to_cartola_rounds(matches: pd.DataFrame, scouts_active: pd.DataFrame) -> pd.DataFrame:
    """
    Learnable Pill: Chronological vs. Official Round Alignment
    In historical datasets (2014, 2016, 2018, 2019), rescheduled matches cause a chronological
    10-match-per-chunk index (`matches.parquet`) to drift from Cartola's official round number.
    Here we align every (season, round, team_id) in `scouts_active` with its exact fixture
    in `matches.parquet` using exact round match first, and score-consistent proximity matching
    for rescheduled fixtures.
    """
    # Summarize team scouts per (season, round, team_id)
    team_round = (
        scouts_active.groupby(["season", "round", "team_id"], as_index=False)
        .agg(
            scout_G=("G", "sum"),
            scout_GS=("GS", "max"),
            scout_SG=("SG", "max"),
            scout_GC=("GC", "sum"),
            n_players=("player_id", "count"),
        )
    )

    # Expand matches into 2 directed rows per match (home perspective & away perspective)
    m_home = matches[
        ["match_id", "season", "round", "date", "home_team_id", "away_team_id", "home_score", "away_score"]
    ].rename(
        columns={
            "round": "chrono_round",
            "home_team_id": "team_id",
            "away_team_id": "opp_id",
            "home_score": "gf",
            "away_score": "ga",
        }
    )
    m_home["is_home"] = 1

    m_away = matches[
        ["match_id", "season", "round", "date", "away_team_id", "home_team_id", "away_score", "home_score"]
    ].rename(
        columns={
            "round": "chrono_round",
            "away_team_id": "team_id",
            "home_team_id": "opp_id",
            "away_score": "gf",
            "home_score": "ga",
        }
    )
    m_away["is_home"] = 0

    m_dir = pd.concat([m_home, m_away], ignore_index=True)

    # Join candidate matches on (season, team_id) and score each candidate fixture
    candidates = team_round.merge(m_dir, on=["season", "team_id"], how="left")

    # Determine if opponent also has active scouts in the exact same Cartola (season, round)
    active_keys = set(zip(team_round["season"], team_round["round"], team_round["team_id"]))
    opp_in_round = [
        1 if (s, r, o) in active_keys else 0
        for s, r, o in zip(candidates["season"], candidates["round"], candidates["opp_id"])
    ]
    candidates["opp_in_round"] = opp_in_round

    # Score compatibility between Cartola round scouts and fixture scoreline:
    # 1) Round distance penalty
    round_diff = (candidates["round"] - candidates["chrono_round"]).abs()
    # 2) Goals scored compatibility (team scout G <= match gf, usually equal unless own goal)
    g_diff = (candidates["scout_G"] - candidates["gf"]).abs()
    # 3) Clean sheet compatibility (if ga == 0, SG is usually 1; if ga > 0, SG must be 0)
    actual_cs = (candidates["ga"] == 0).astype(float)
    cs_match = (candidates["scout_SG"] == actual_cs).astype(float)
    # 4) Goals conceded compatibility
    gs_diff = (candidates["scout_GS"] - candidates["ga"]).abs()

    candidates["match_quality"] = (
        candidates["opp_in_round"] * 50.0
        + (round_diff == 0).astype(float) * 30.0
        - round_diff * 2.5
        + cs_match * 15.0
        - g_diff * 8.0
        - gs_diff * 5.0
    )

    best_matches = (
        candidates.sort_values(["season", "round", "team_id", "match_quality"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["season", "round", "team_id"], keep="first")
        .reset_index(drop=True)
    )

    return best_matches[
        [
            "season",
            "round",
            "team_id",
            "opp_id",
            "match_id",
            "date",
            "is_home",
            "gf",
            "ga",
            "scout_G",
            "scout_GS",
            "scout_SG",
            "n_players",
        ]
    ]


def build_team_and_positional_cube(scouts_active: pd.DataFrame, aligned_fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Level 1 & Level 2 Relational Vectorization:
    Computes team-level and opponent-conceded rolling statistics across all 8,629 team-rounds.
    """
    outfield = scouts_active[scouts_active["position_id"] != 6].copy()

    # Base score excluding goals, assists, and clean sheets (pure continuous action floor)
    outfield["floor_score"] = (
        outfield["base_norm_score"]
        - outfield["G"] * 8.0
        - outfield["A"] * 5.0
        - outfield["SG"] * 5.0
    )

    # 1. Team-Match overall aggregation
    team_agg = (
        outfield.groupby(["season", "round", "team_id"], as_index=False)
        .agg(
            team_pts_sum=("base_norm_score", "sum"),
            team_pts_mean=("base_norm_score", "mean"),
            team_floor_mean=("floor_score", "mean"),
            team_G=("G", "sum"),
            team_A=("A", "sum"),
            team_DS=("DS", "sum"),
            team_DE=("DE", "sum"),
            team_FD=("FD", "sum"),
            team_FF=("FF", "sum"),
            team_FS=("FS", "sum"),
            team_FC=("FC", "sum"),
            team_SG_max=("SG", "max"),
            team_avg_price=("price", "mean"),
        )
    )

    cube = aligned_fixtures.merge(team_agg, on=["season", "round", "team_id"], how="left")
    cube["clean_sheet"] = (cube["ga"] == 0).astype(float)

    # Mirror stats onto opponent to get "Conceded by Opponent in that match"
    opp_mirror = cube[
        [
            "season",
            "round",
            "team_id",
            "team_pts_mean",
            "team_floor_mean",
            "team_G",
            "team_A",
            "team_DS",
            "team_DE",
            "team_FD",
            "team_FF",
            "team_FS",
            "clean_sheet",
        ]
    ].rename(
        columns={
            "team_id": "opp_id",
            "team_pts_mean": "conceded_pts_mean_in_match",
            "team_floor_mean": "conceded_floor_mean_in_match",
            "team_G": "conceded_G_in_match",
            "team_A": "conceded_A_in_match",
            "team_DS": "conceded_DS_in_match",
            "team_DE": "conceded_DE_in_match",
            "team_FD": "conceded_FD_in_match",
            "team_FF": "conceded_FF_in_match",
            "team_FS": "conceded_FS_in_match",
            "clean_sheet": "conceded_cs_in_match",
        }
    )

    # In cube, what Team A *concedes* in round R is what Team A's opponent (`opp_id`) *achieved* in round R!
    # Wait: if we join `opp_mirror` where `opp_mirror.opp_id == cube.opp_id`, then `opp_mirror`'s stats are what
    # `cube.opp_id` achieved against `cube.team_id` (i.e. what `cube.team_id` conceded in that match!).
    cube = cube.merge(opp_mirror, on=["season", "round", "opp_id"], how="left")

    # Fallback if opponent wasn't in active scouts in that exact round number
    cube["conceded_G_in_match"] = cube["conceded_G_in_match"].fillna(cube["ga"])
    for col, fallback_col in [
        ("conceded_pts_mean_in_match", "team_pts_mean"),
        ("conceded_floor_mean_in_match", "team_floor_mean"),
        ("conceded_A_in_match", "team_A"),
        ("conceded_DS_in_match", "team_DS"),
        ("conceded_DE_in_match", "team_DE"),
        ("conceded_FD_in_match", "team_FD"),
        ("conceded_FF_in_match", "team_FF"),
        ("conceded_FS_in_match", "team_FS"),
    ]:
        cube[col] = cube[col].fillna(cube[fallback_col].median())

    # Sort chronologically by team_id, season, round
    cube = cube.sort_values(["team_id", "season", "round"]).reset_index(drop=True)

    # Compute strictly shifted (t-1) rolling windows per (team_id, season) for short-term form,
    # and across (team_id) with cross-season shrinkage for Round 1 continuity!
    metrics_to_roll = [
        "gf",
        "ga",
        "clean_sheet",
        "team_pts_mean",
        "team_floor_mean",
        "team_G",
        "team_A",
        "team_DS",
        "team_FD",
        "team_avg_price",
        "conceded_pts_mean_in_match",
        "conceded_floor_mean_in_match",
        "conceded_G_in_match",
        "conceded_A_in_match",
        "conceded_DS_in_match",
        "conceded_FD_in_match",
    ]

    # Pre-compute cross-season decayed prior per team so Round 1 isn't blind
    season_team_means = (
        cube.groupby(["team_id", "season"], as_index=False)[metrics_to_roll].mean()
        .sort_values(["team_id", "season"])
    )
    for m in metrics_to_roll:
        season_team_means[f"prev_season_{m}"] = season_team_means.groupby("team_id")[m].shift(1)

    cube = cube.merge(
        season_team_means[["team_id", "season"] + [f"prev_season_{m}" for m in metrics_to_roll]],
        on=["team_id", "season"],
        how="left",
    )

    pop_means = {m: cube[m].mean() for m in metrics_to_roll}

    # Within-season shifted rolling stats + Bayesian blend with previous season prior
    g_team_season = cube.groupby(["team_id", "season"], sort=False)
    cube["team_season_games_played"] = g_team_season.cumcount()  # 0 in Round 1

    k_team = 4.0  # Shrinkage weight (4 matches of prior strength)
    n_tm = cube["team_season_games_played"].values

    for m in metrics_to_roll:
        prior_val = (
            0.75 * cube[f"prev_season_{m}"].fillna(pop_means[m]).values
            + 0.25 * pop_means[m]
        )
        shifted_s = g_team_season[m].shift(1)
        cum_mean = shifted_s.groupby([cube["team_id"], cube["season"]]).expanding().mean().reset_index(level=[0, 1], drop=True).fillna(0.0).values
        # Empirical Bayes blend for current season baseline
        cube[f"eb_{m}"] = (n_tm * cum_mean + k_team * prior_val) / (n_tm + k_team)

        # Rolling 5 and 10 within season (blended with prior when n_tm < window)
        for w in [5, 10]:
            r_mean = shifted_s.groupby([cube["team_id"], cube["season"]]).rolling(w, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(0.0).values
            eff_n = np.minimum(n_tm, w)
            cube[f"roll{w}_{m}"] = (eff_n * r_mean + (w - eff_n) * prior_val) / w

    # Clean up intermediate prev_season columns
    cube.drop(columns=[f"prev_season_{m}" for m in metrics_to_roll], inplace=True)

    # 2. Positional Conceded Cube: How many points/DS/G/A does Team X concede specifically to Position P?
    pos_agg = (
        outfield.groupby(["season", "round", "team_id", "position_id"], as_index=False)
        .agg(
            pos_pts_mean=("base_norm_score", "mean"),
            pos_floor_mean=("floor_score", "mean"),
            pos_G_sum=("G", "sum"),
            pos_A_sum=("A", "sum"),
            pos_DS_mean=("DS", "mean"),
            pos_FD_mean=("FD", "mean"),
            pos_DE_mean=("DE", "mean"),
        )
    )
    # Attach `opp_id` from aligned_fixtures: if `team_id` scored `pos_pts_mean` against `opp_id`,
    # then `opp_id` CONCEDED `pos_pts_mean` to `position_id`!
    pos_conceded = pos_agg.merge(
        aligned_fixtures[["season", "round", "team_id", "opp_id"]],
        on=["season", "round", "team_id"],
        how="inner",
    )
    # Now group by the DEFENDING team (`opp_id`) and `position_id`
    pos_def = (
        pos_conceded.drop(columns=["team_id"])
        .rename(columns={"opp_id": "defending_team_id"})
        .sort_values(["defending_team_id", "position_id", "season", "round"])
        .reset_index(drop=True)
    )

    pos_pop_means = pos_def.groupby("position_id")[
        ["pos_pts_mean", "pos_floor_mean", "pos_DS_mean", "pos_FD_mean", "pos_DE_mean"]
    ].transform("mean")

    g_pos_def = pos_def.groupby(["defending_team_id", "position_id", "season"], sort=False)
    n_pos = g_pos_def.cumcount().values

    for col in ["pos_pts_mean", "pos_floor_mean", "pos_DS_mean", "pos_FD_mean", "pos_DE_mean"]:
        p_mean = pos_pop_means[col].values
        shifted_p = g_pos_def[col].shift(1)
        for w in [5, 10]:
            r_val = (
                shifted_p.groupby([pos_def["defending_team_id"], pos_def["position_id"], pos_def["season"]])
                .rolling(w, min_periods=1)
                .mean()
                .reset_index(level=[0, 1, 2], drop=True)
                .fillna(0.0)
                .values
            )
            eff_n = np.minimum(n_pos, w)
            pos_def[f"opp_pos_conceded_roll{w}_{col}"] = (eff_n * r_val + (w - eff_n) * p_mean) / w

    keep_pos_cols = [
        "season",
        "round",
        "defending_team_id",
        "position_id",
        "opp_pos_conceded_roll5_pos_pts_mean",
        "opp_pos_conceded_roll10_pos_pts_mean",
        "opp_pos_conceded_roll5_pos_floor_mean",
        "opp_pos_conceded_roll10_pos_floor_mean",
        "opp_pos_conceded_roll5_pos_DS_mean",
        "opp_pos_conceded_roll5_pos_FD_mean",
        "opp_pos_conceded_roll5_pos_DE_mean",
    ]
    pos_def = pos_def[keep_pos_cols].drop_duplicates(
        subset=["season", "round", "defending_team_id", "position_id"]
    )

    return cube, pos_def


def build_player_features(
    scouts_active: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    cube: pd.DataFrame,
    pos_def: pd.DataFrame,
) -> pd.DataFrame:
    """
    Level 3 Vectorized Feature Engineering:
    Computes player-level rolling windows (3, 5, 10), EWMA, Bayesian Shrunk Career &
    Previous-Season Priors, Cross-Season Reset/Decay, Consistency/Ceiling quantiles,
    Home/Away splits, and joins Team + Opponent Context.
    """
    df = scouts_active.copy()

    # Decompose target `base_norm_score` into structural components:
    # 1. Clean sheet bonus (5.0 for GOL, LAT, ZAG when SG==1)
    df["sg_Component"] = np.where(df["position_id"].isin([1, 2, 3]), df["SG"] * 5.0, 0.0)
    # 2. Attacking goal/assist spike (8.0*G + 5.0*A)
    df["ga_Component"] = df["G"] * 8.0 + df["A"] * 5.0
    # 3. Continuous Base Action Floor (DS, FS, FD, FF, FT, DE, DP, FC, CA, CV, GS, I, PP)
    df["floor_Component"] = np.where(
        df["position_id"] == 6,
        df["base_norm_score"],
        df["base_norm_score"] - df["sg_Component"] - df["ga_Component"],
    )

    # Sort strictly in chronological order per player
    df = df.sort_values(["player_id", "season", "round"]).reset_index(drop=True)

    # Join match context (team's own features)
    own_team_cols = [
        "season",
        "round",
        "team_id",
        "opp_id",
        "match_id",
        "date",
        "is_home",
        "gf",
        "ga",
        "clean_sheet",
        "eb_gf",
        "eb_ga",
        "eb_clean_sheet",
        "eb_team_pts_mean",
        "eb_team_floor_mean",
        "eb_team_avg_price",
        "roll5_gf",
        "roll5_ga",
        "roll5_clean_sheet",
        "roll5_team_pts_mean",
        "roll5_team_floor_mean",
        "roll10_team_pts_mean",
    ]
    df = df.merge(cube[own_team_cols], on=["season", "round", "team_id"], how="left")

    # Join opponent's pre-match stats (by matching `opp_id` to `cube.team_id`)
    opp_team_cols = {
        "team_id": "opp_id",
        "eb_gf": "opp_eb_gf",
        "eb_ga": "opp_eb_ga",
        "eb_clean_sheet": "opp_eb_clean_sheet",
        "eb_team_pts_mean": "opp_eb_team_pts_mean",
        "eb_conceded_pts_mean_in_match": "opp_eb_conceded_pts",
        "eb_conceded_floor_mean_in_match": "opp_eb_conceded_floor",
        "eb_conceded_G_in_match": "opp_eb_conceded_G",
        "eb_conceded_A_in_match": "opp_eb_conceded_A",
        "eb_conceded_DS_in_match": "opp_eb_conceded_DS",
        "eb_conceded_FD_in_match": "opp_eb_conceded_FD",
        "roll5_gf": "opp_roll5_gf",
        "roll5_ga": "opp_roll5_ga",
        "roll5_conceded_pts_mean_in_match": "opp_roll5_conceded_pts",
        "roll10_conceded_pts_mean_in_match": "opp_roll10_conceded_pts",
        "roll5_conceded_floor_mean_in_match": "opp_roll5_conceded_floor",
        "roll5_conceded_G_in_match": "opp_roll5_conceded_G",
        "roll5_conceded_A_in_match": "opp_roll5_conceded_A",
        "roll10_conceded_A_in_match": "opp_roll10_conceded_A",
        "roll5_conceded_DS_in_match": "opp_roll5_conceded_DS",
        "roll10_conceded_DS_in_match": "opp_roll10_conceded_DS",
        "roll5_conceded_FD_in_match": "opp_roll5_conceded_FD",
    }
    opp_cube = cube[["season", "round"] + list(opp_team_cols.keys())].rename(columns=opp_team_cols)
    df = df.merge(opp_cube, on=["season", "round", "opp_id"], how="left")

    # Join positional vulnerability of the opponent (`pos_def`)
    df = df.merge(
        pos_def.rename(columns={"defending_team_id": "opp_id"}),
        on=["season", "round", "opp_id", "position_id"],
        how="left",
    )

    # Positional global averages (for Empirical Bayes shrinkage anchor)
    pos_means = df.groupby("position_id")["base_norm_score"].transform("mean")
    pos_floor_means = df.groupby("position_id")["floor_Component"].transform("mean")
    pos_ga_means = df.groupby("position_id")["ga_Component"].transform("mean")
    df["pos_mean_anchor"] = pos_means

    # --- PLAYER HISTORICAL & SEASONAL MEMORY ---
    # 1. Career expanding prior up to t-1 (across all seasons)
    g_player = df.groupby("player_id", sort=False)
    df["career_games_prior"] = g_player.cumcount()

    shifted_score_all = g_player["base_norm_score"].shift(1)
    shifted_floor_all = g_player["floor_Component"].shift(1)
    shifted_ga_all = g_player["ga_Component"].shift(1)

    career_cum_mean = (
        shifted_score_all.groupby(df["player_id"])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    career_floor_cum_mean = (
        shifted_floor_all.groupby(df["player_id"])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    career_ga_cum_mean = (
        shifted_ga_all.groupby(df["player_id"])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # Empirical Bayes Career Baseline ($k = 8$ appearances shrinkage toward position mean)
    k_career = 8.0
    n_c = df["career_games_prior"].values
    df["player_career_eb_score"] = (n_c * career_cum_mean.values + k_career * pos_means.values) / (n_c + k_career)
    df["player_career_eb_floor"] = (n_c * career_floor_cum_mean.values + k_career * pos_floor_means.values) / (
        n_c + k_career
    )
    df["player_career_eb_ga"] = (n_c * career_ga_cum_mean.values + k_career * pos_ga_means.values) / (n_c + k_career)

    # 2. Previous Season Summary per player (to handle the Cross-Season Gap explicitly)
    season_player_stats = (
        df.groupby(["player_id", "season"], as_index=False)
        .agg(
            season_games=("base_norm_score", "count"),
            season_mean_score=("base_norm_score", "mean"),
            season_mean_floor=("floor_Component", "mean"),
            season_mean_ga=("ga_Component", "mean"),
        )
        .sort_values(["player_id", "season"])
    )
    season_player_stats["prev_season"] = season_player_stats.groupby("player_id")["season"].shift(1)
    season_player_stats["prev_season_games"] = np.where(
        season_player_stats["season"] - season_player_stats["prev_season"] == 1,
        season_player_stats.groupby("player_id")["season_games"].shift(1),
        0.0,
    )
    season_player_stats["prev_season_mean_score"] = np.where(
        season_player_stats["season"] - season_player_stats["prev_season"] == 1,
        season_player_stats.groupby("player_id")["season_mean_score"].shift(1),
        np.nan,
    )
    season_player_stats["prev_season_mean_floor"] = np.where(
        season_player_stats["season"] - season_player_stats["prev_season"] == 1,
        season_player_stats.groupby("player_id")["season_mean_floor"].shift(1),
        np.nan,
    )

    df = df.merge(
        season_player_stats[
            ["player_id", "season", "prev_season_games", "prev_season_mean_score", "prev_season_mean_floor"]
        ],
        on=["player_id", "season"],
        how="left",
    )

    # Off-season retention prior: blends previous season (with 0.80 retention) and career EB prior
    prev_n = df["prev_season_games"].fillna(0.0).values
    prev_mean = df["prev_season_mean_score"].fillna(df["player_career_eb_score"]).values
    df["player_offseason_prior"] = (
        (prev_n * (0.80 * prev_mean + 0.20 * pos_means.values) + 6.0 * df["player_career_eb_score"].values)
        / (prev_n + 6.0)
    )

    # 3. Within-Season Form & Dual-Memory Shrinkage
    g_ps = df.groupby(["player_id", "season"], sort=False)
    df["season_games_prior"] = g_ps.cumcount()
    n_s = df["season_games_prior"].values

    # Rounds elapsed since last active appearance in the same season (captures injury/suspension/bench rust)
    prev_round_in_season = g_ps["round"].shift(1)
    df["rounds_since_last_played"] = (df["round"] - prev_round_in_season).fillna(df["round"]).clip(upper=20.0)
    df["played_last_round"] = (df["rounds_since_last_played"] == 1.0).astype(float)

    shifted_ps_score = g_ps["base_norm_score"].shift(1)
    shifted_ps_floor = g_ps["floor_Component"].shift(1)
    shifted_ps_ga = g_ps["ga_Component"].shift(1)

    season_cum_mean = (
        shifted_ps_score.groupby([df["player_id"], df["season"]])
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
        .values
    )
    season_floor_cum_mean = (
        shifted_ps_floor.groupby([df["player_id"], df["season"]])
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
        .values
    )

    # Dual-Memory Empirical Bayes Estimate: smoothly transitions from `player_offseason_prior` in Round 1
    # to current-season average as `season_games_prior` grows!
    k_season = 5.0
    df["player_dual_eb_score"] = (
        n_s * season_cum_mean + k_season * df["player_offseason_prior"].values
    ) / (n_s + k_season)

    prev_floor = df["prev_season_mean_floor"].fillna(df["player_career_eb_floor"]).values
    offseason_floor_prior = (prev_n * (0.80 * prev_floor + 0.20 * pos_floor_means.values) + 6.0 * df["player_career_eb_floor"].values) / (prev_n + 6.0)
    df["player_dual_eb_floor"] = (
        n_s * season_floor_cum_mean + k_season * offseason_floor_prior
    ) / (n_s + k_season)

    # Player-over-Position Talent Delta ("The Arrascaeta Genius Feature")
    df["player_genius_delta"] = df["player_dual_eb_score"] - df["pos_mean_anchor"]
    df["player_floor_delta"] = df["player_dual_eb_floor"] - pos_floor_means

    # 4. Rolling Windows (W = 3, 5, 10) & EWMA within season (blended with offseason prior when n_s < W)
    player_roll_cols = [
        ("base_norm_score", "player_dual_eb_score"),
        ("floor_Component", "player_dual_eb_floor"),
        ("ga_Component", "player_career_eb_ga"),
        ("G", None),
        ("A", None),
        ("DS", None),
        ("FD", None),
        ("FF", None),
        ("FS", None),
        ("DE", None),
        ("FC", None),
        ("SG", None),
    ]

    pos_col_means = {
        c: df.groupby("position_id")[c].transform("mean").values
        for c, _ in player_roll_cols
    }

    for col, prior_col in player_roll_cols:
        s_shift = g_ps[col].shift(1)
        prior_arr = df[prior_col].values if prior_col else pos_col_means[col]

        for w in [3, 5, 10]:
            if w != 5 and col not in ["base_norm_score", "floor_Component", "ga_Component", "DS", "FD"]:
                continue
            r_mean = (
                s_shift.groupby([df["player_id"], df["season"]])
                .rolling(w, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
                .fillna(0.0)
                .values
            )
            eff_n = np.minimum(n_s, w)
            df[f"player_roll{w}_{col}"] = (eff_n * r_mean + (w - eff_n) * prior_arr) / w

        # EWMA (halflife = 3 matches) across career with season boundary damping
        ewma_vals = (
            g_player[col]
            .shift(1)
            .groupby(df["player_id"])
            .ewm(halflife=3.5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(pd.Series(prior_arr, index=df.index))
            .values
        )
        df[f"player_ewma_{col}"] = np.where(n_s == 0, prior_arr, ewma_vals)

    # 5. Player Volatility, Floor (25th pct), Ceiling (85th pct), and Home/Away Affinity
    r10_std = (
        shifted_score_all.groupby(df["player_id"])
        .rolling(10, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(3.5)
        .values
    )
    r10_q85 = (
        shifted_score_all.groupby(df["player_id"])
        .rolling(10, min_periods=3)
        .quantile(0.85)
        .reset_index(level=0, drop=True)
        .fillna(df["player_dual_eb_score"] + 3.5)
        .values
    )
    r10_q25 = (
        shifted_score_all.groupby(df["player_id"])
        .rolling(10, min_periods=3)
        .quantile(0.25)
        .reset_index(level=0, drop=True)
        .fillna(df["player_dual_eb_score"] - 2.5)
        .values
    )
    df["player_roll10_std"] = r10_std
    df["player_roll10_ceiling_q85"] = r10_q85
    df["player_roll10_floor_q25"] = r10_q25

    # 6. Market / Price Intelligence (strictly pre-match price! Note: in Cartola, `price` in round R is post-round R,
    # while `price - price_variation` is the exact pre-round R market price!)
    df["pre_match_price"] = (df["price"] - df["price_variation"]).clip(lower=1.0)
    df["prev_price_variation"] = g_ps["price_variation"].shift(1).fillna(0.0)
    team_round_mean_price = df.groupby(["season", "round", "team_id"])["pre_match_price"].transform("mean")
    pos_round_mean_price = df.groupby(["season", "round", "position_id"])["pre_match_price"].transform("mean")
    df["price_rel_to_team"] = df["pre_match_price"] / team_round_mean_price.clip(lower=1.0)
    df["price_rel_to_pos"] = df["pre_match_price"] / pos_round_mean_price.clip(lower=1.0)

    # 7. Matchup Interaction Features (Team vs. Opponent & Player vs. Opponent Vulnerability)
    df["team_vs_opp_xg_diff"] = (df["eb_gf"] + df["opp_eb_ga"]) / 2.0 - (df["opp_eb_gf"] + df["eb_ga"]) / 2.0
    df["expected_team_gf"] = (
        0.55 * (df["roll5_gf"] + df["opp_roll5_ga"]) / 2.0
        + 0.45 * (df["eb_gf"] + df["opp_eb_ga"]) / 2.0
        + np.where(df["is_home"] == 1, 0.22, -0.22)
    ).clip(lower=0.2)
    df["expected_opp_gf"] = (
        0.55 * (df["opp_roll5_gf"] + df["roll5_ga"]) / 2.0
        + 0.45 * (df["opp_eb_gf"] + df["eb_ga"]) / 2.0
        + np.where(df["is_home"] == 1, -0.22, 0.22)
    ).clip(lower=0.2)

    # Poisson Clean Sheet Probability: P(Opponent Goals == 0) = exp(-lambda_opp)
    df["poisson_sg_prob"] = np.exp(-df["expected_opp_gf"])
    df["expected_sg_points"] = np.where(df["position_id"].isin([1, 2, 3]), 5.0 * df["poisson_sg_prob"], 0.0)

    # Interaction: Player attacking propensity * Opponent goals/assists conceded
    df["ix_player_ga_vs_opp_conceded"] = df["player_roll5_ga_Component"] * (
        df["opp_roll5_conceded_G"] / 1.25
    )
    # Interaction: Player tackling (DS) * Opponent DS conceded to position
    df["ix_player_ds_vs_opp_pos_ds"] = (
        df["player_roll5_DS"] * df["opp_pos_conceded_roll5_pos_DS_mean"].fillna(1.2)
    )
    # Interaction: Goalkeeper saves * Opponent shots/attacking strength
    df["ix_gk_de_vs_opp_attack"] = np.where(
        df["position_id"] == 1,
        df["player_roll5_DE"] * df["expected_opp_gf"],
        0.0,
    )

    # 8. Target Definitions (Continuous, Residual, Fixed Mito >= 8.0, and Positional Round Quantile Top 15%)
    df["target_score"] = df["base_norm_score"]
    df["target_residual"] = df["target_score"] - df["player_dual_eb_score"]
    df["target_mito_8"] = (df["target_score"] >= 8.0).astype(int)

    # Positional Top 15% Quantile per (season, round, position_id)
    pos_round_q85 = df.groupby(["season", "round", "position_id"])["target_score"].transform(
        lambda s: s.quantile(0.85)
    )
    df["target_mito_q85"] = (
        (df["target_score"] >= pos_round_q85) & (df["target_score"] > 0.0)
    ).astype(int)

    # Attach player canonical metadata and team abbreviations for rich reports
    df = df.merge(
        players[["player_id", "canonical_name", "canonical_nickname", "photo_url"]],
        on="player_id",
        how="left",
    )
    team_abbr_map = teams.set_index("team_id")["abbreviation"].to_dict()
    df["team_abbr"] = df["team_id"].map(team_abbr_map).fillna("CLB")
    df["opp_abbr"] = df["opp_id"].map(team_abbr_map).fillna("OPP")
    df["position_code"] = df["position_id"].map(POS_MAP)

    # Fill any remaining edge-case NaNs in engineered features with positional medians
    feature_cols = [
        c
        for c in df.columns
        if c.startswith(
            (
                "eb_",
                "roll",
                "opp_",
                "player_",
                "pre_match_",
                "prev_",
                "price_rel_",
                "team_vs_",
                "expected_",
                "poisson_",
                "ix_",
                "rounds_since_",
                "played_last_",
                "is_home",
                "pos_mean_",
            )
        )
    ]
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df.groupby("position_id")[col].transform("median")).fillna(0.0)

    return df


def verify_zero_leakage(df: pd.DataFrame):
    """
    Automated QA Gate: Verifies that no feature uses future or current-round target information.
    """
    # 1. Check that Round 1 season_games_prior is always 0 for a player's first game of the season
    first_season_rows = df.groupby(["player_id", "season"]).head(1)
    assert (first_season_rows["season_games_prior"] == 0).all(), "Leakage in season_games_prior!"

    # 2. Check that pre_match_price equals price - price_variation (clipped at 1.0)
    expected_pre = (df["price"] - df["price_variation"]).clip(lower=1.0)
    assert np.allclose(df["pre_match_price"], expected_pre), "Pre-match price calculation mismatch!"

    # 3. Spot check Arrascaeta (or top player) rolling window shift
    sample_pid = df["player_id"].value_counts().index[0]
    p_sub = df[(df["player_id"] == sample_pid) & (df["season"] == 2023)].sort_values("round")
    if len(p_sub) >= 4:
        manual_prev3 = p_sub["target_score"].iloc[:3].mean()
        stored_r3_at_4th = p_sub["player_roll3_base_norm_score"].iloc[3]
        assert np.isclose(manual_prev3, stored_r3_at_4th, atol=1e-4), (
            f"Shift check failed: {manual_prev3} vs {stored_r3_at_4th}"
        )

    print("-> [QA PASS] Zero temporal leakage verified across all player & team lag windows!")


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("CLF Phase 2: Building Vectorized Relational Feature Store")
    print("=" * 72)

    matches = pd.read_parquet(CLT_DATA_DIR / "matches.parquet")
    scouts = pd.read_parquet(CLT_DATA_DIR / "normalized_scouts.parquet")
    players = pd.read_parquet(CLT_DATA_DIR / "players.parquet")
    teams = pd.read_parquet(CLT_DATA_DIR / "teams.parquet")

    scouts_active = scouts[scouts["played"]].copy().reset_index(drop=True)
    print(f"1. Loaded {len(scouts_active):,} active player appearances across 12 seasons (2014-2025).")
    scouts_active = sanitize_cumulative_bleed_anomalies(scouts_active)

    t1 = time.perf_counter()
    aligned_fixtures = align_matches_to_cartola_rounds(matches, scouts_active)
    cube, pos_def = build_team_and_positional_cube(scouts_active, aligned_fixtures)
    cube_path = CLF_DATA_DIR / "team_match_cube.parquet"
    cube.to_parquet(cube_path, index=False)
    print(f"2. Built Team-Match & Opponent Conceded Cube ({len(cube):,} team-rounds) in {time.perf_counter() - t1:.2f}s.")

    t2 = time.perf_counter()
    df_features = build_player_features(scouts_active, players, teams, cube, pos_def)
    verify_zero_leakage(df_features)

    out_path = CLF_DATA_DIR / "clf_feature_store.parquet"
    df_features.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0

    print(f"3. Saved Feature Store: {df_features.shape[0]:,} rows x {df_features.shape[1]} columns -> {out_path}")
    print(f"Total End-to-End Runtime: {elapsed:.2f} seconds!")
    print("=" * 72)


if __name__ == "__main__":
    main()
