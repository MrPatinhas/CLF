"""
run_inference.py — CLF Automated 'Click-and-Go' Round Inference Engine
========================================================================
CLI tool to forecast Cartola player performance for any round in $<5$ seconds.

Usage Examples:
  python run_inference.py                     # Auto-detects latest data and predicts next round
  python run_inference.py --season 2026 --round 35
  python run_inference.py --season 2026 --round 36    # Throws clean error if R35 scouts not published
  python run_inference.py --season 2024 --round 20    # Emits historical training-ground audit warning
  python run_inference.py --retrain                   # Forces fresh model training before inference

Outputs:
  * Formatted console summary (Meta-Ensemble squad + 4 Specialists + Consensus Stars)
  * data/inference_{season}_r{round}.parquet
  * reports/inference_{season}_r{round}.html
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.stdout.reconfigure(encoding="utf-8")

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
SRC_DIR = CLF_DIR / "src"
DATA_DIR = CLF_DIR / "data"
REPORTS_DIR = CLF_DIR / "reports"
MODELS_DIR = DATA_DIR / "models"
CLT_DATA_DIR = Path(r"C:\Users\thiag\Documents\CLT\data")

sys.path.insert(0, str(SRC_DIR))
import model_registry
import round_fetcher
import portfolio_optimizer

STATUS_MAP = {2: "Dúvida", 3: "Suspenso", 5: "Contundido", 6: "Nulo", 7: "Provável"}
POS_CODE = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}
POS_NAME = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}

# 21-player squad per model: 4-3-3 starting XI + Coach (12) + 9 reserves (1 GOL, 2 LAT, 2 ZAG, 2 MEI, 2 ATA)
SQUAD_QUOTA = {1: 2, 2: 4, 3: 4, 4: 5, 5: 5, 6: 1}
STARTING_POS_QUOTA = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1}

SPECIALISTS = {
    "Positional_LightGBM": "pred_Positional_LightGBM",
    "Residual_XGBoost": "pred_Residual_XGBoost",
    "Decomposed_Hybrid": "pred_Decomposed_Hybrid",
    "Neural_MLP": "pred_Neural_MLP",
    "Generative_Poisson": "pred_Generative_Poisson",
}
ENSEMBLE_COL = "pred_CLF_Meta_Ensemble"


def load_feature_store_module():
    spec = importlib.util.spec_from_file_location("fs", SRC_DIR / "01_build_feature_store.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    return fs


def _tag_squad_slots(df_sub: pd.DataFrame, pos_id: int) -> pd.DataFrame:
    """Assigns STARTER vs RES 1 / RES 2 based on official Cartola 4-3-3 starting quotas."""
    df_copy = df_sub.copy().reset_index(drop=True)
    n_starters = STARTING_POS_QUOTA[pos_id]
    slots = []
    for i in range(len(df_copy)):
        if pos_id == 6:
            slots.append("COACH")
        elif i < n_starters:
            slots.append("STARTER")
        else:
            sub_num = i - n_starters + 1
            slots.append(f"RES {sub_num}")
    df_copy["lineup_slot"] = slots
    return df_copy


def build_squads(pool: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Selects the top 21-player squad (12 starters + 9 reserves) for each model and computes consensus votes."""
    if "upside_spread" not in pool.columns:
        pool["upside_spread"] = np.round(pool["gen_ceiling_p90"] - pool["pred_Generative_Poisson"], 2)
    if "vol_multiplier" not in pool.columns:
        pool["vol_multiplier"] = np.round(pool["gen_ceiling_p90"] / np.maximum(pool["pred_Generative_Poisson"], 0.5), 2)

    squads = {}

    def _select_ranked_squad(col_name: str, pool_df: pd.DataFrame = pool) -> pd.DataFrame:
        picks = []
        for pos_id, quota in SQUAD_QUOTA.items():
            sub = pool_df[pool_df["position_id"] == pos_id].nlargest(quota, col_name)
            picks.append(_tag_squad_slots(sub, pos_id))
        return pd.concat(picks, ignore_index=True)

    # 1. Standard Specialists (21 Players each)
    for name, col in SPECIALISTS.items():
        squads[name] = _select_ranked_squad(col)

    # 2. Meta-Ensemble (21 Players)
    squads["CLF_Meta_Ensemble"] = _select_ranked_squad(ENSEMBLE_COL)

    # 3. Rank A: Generative Expected Mean (21 Players)
    squads["Generative_Expected_Mean"] = squads["Generative_Poisson"]

    # 4. Rank B: Generative Upside Spread (Ceiling - Mean, 21 Players)
    squads["Generative_Upside_Spread"] = _select_ranked_squad("upside_spread")

    # 5. Rank C: Generative Boom-or-Bust (Ceiling / Mean with Mean >= 2.5, 21 Players)
    boom_picks = []
    boom_pool = pool[pool["pred_Generative_Poisson"] >= 2.5]
    for pos_id, quota in SQUAD_QUOTA.items():
        sub_pos = boom_pool[boom_pool["position_id"] == pos_id]
        if len(sub_pos) < quota:
            sub_pos = pool[pool["position_id"] == pos_id]
        boom_picks.append(_tag_squad_slots(sub_pos.nlargest(quota, "vol_multiplier"), pos_id))
    squads["Generative_Boom_Bust"] = pd.concat(boom_picks, ignore_index=True)

    core_specialists = ["Positional_LightGBM", "Residual_XGBoost", "Decomposed_Hybrid", "Neural_MLP", "Generative_Poisson"]
    votes = pd.concat([s[["player_id"]].assign(specialist=m) for m, s in squads.items() if m in core_specialists])
    consensus = votes.groupby("player_id")["specialist"].agg(
        n_votes="count",
        picked_by=lambda x: ", ".join(sorted(x)),
    ).reset_index()

    return squads, consensus


def format_terminal_squad(name: str, sq: pd.DataFrame, consensus: pd.DataFrame) -> str:
    merged = sq.merge(consensus, on="player_id", how="left")
    merged["n_votes"] = merged["n_votes"].fillna(0).astype(int)

    lines = []
    lines.append(f"\n{'='*98}")
    lines.append(f"  ★ SQUAD SELECTION: {name} (21 Players: 12 Starters + 9 Reserves)")
    lines.append(f"{'='*98}")

    in_bench = False
    if "gen_ceiling_p90" in merged.columns and "gen_floor_p10" in merged.columns:
        lines.append(f" {'SLOT':<8} {'POS':<4} {'PLAYER':<18} {'TEAM':<5} {'OPP':<5} {'PRICE':<7} {'FLOOR':<6} {'MEAN':<6} {'CEIL':<6} {'SPREAD':<7} {'MULT':<6} {'MITO%':<6} {'VOTES'}")
        lines.append(f" {'-'*8} {'-'*4} {'-'*18} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*10}")
        for _, r in merged.iterrows():
            slot = str(r.get("lineup_slot", "STARTER"))[:8]
            is_reserve = slot.startswith("RES") or slot == "RESERVE"
            if is_reserve and not in_bench:
                in_bench = True
                lines.append(f" {'-'*8} {'-'*4} {'-'*18} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*10}")
                lines.append("  >>> 🛡️ LUXURY BENCH RESERVES (9 SUBS: 1 GOL, 2 LAT, 2 ZAG, 2 MEI, 2 ATA) <<<")
                lines.append(f" {'-'*8} {'-'*4} {'-'*18} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*10}")

            p_name = str(r["name"])[:17]
            t_abbr = str(r.get("team_abbr", r.get("team_id", "?")))[:4]
            o_abbr = str(r.get("opp_abbr", r.get("opp_id", "?")))[:4]
            price_str = f"C${float(r.get('pre_match_price', 0.0)):.1f}"
            fl = f"{float(r.get('gen_floor_p10', 0.0)):.1f}"
            mn = f"{float(r.get('pred_Generative_Poisson', 0.0)):.2f}"
            cl = f"{float(r.get('gen_ceiling_p90', 0.0)):.1f}"
            sp = f"+{float(r.get('upside_spread', cl_val := float(cl) - float(mn))):.1f}"
            mult = f"{float(r.get('vol_multiplier', 2.0)):.1f}x"
            pm = f"{float(r.get('gen_prob_mito', 0.0))*100:.0f}%"
            votes = int(r.get("n_votes", 0))
            stars = "★" * votes + "☆" * (5 - votes)
            lines.append(f" {slot:<8} {r['position_code']:<4} {p_name:<18} {t_abbr:<5} {o_abbr:<5} {price_str:<7} {fl:<6} {mn:<6} {cl:<6} {sp:<7} {mult:<6} {pm:<6} {stars} ({votes}/5)")
    else:
        lines.append(f" {'SLOT':<8} {'POS':<4} {'PLAYER':<20} {'TEAM':<6} {'OPP':<6} {'LOC':<5} {'PRICE':<7} {'PRED PTS':<9} {'CONSENSUS'}")
        lines.append(f" {'-'*8} {'-'*4} {'-'*20} {'-'*6} {'-'*6} {'-'*5} {'-'*7} {'-'*9} {'-'*15}")
        for _, r in merged.iterrows():
            slot = str(r.get("lineup_slot", "STARTER"))[:8]
            is_reserve = slot.startswith("RES") or slot == "RESERVE"
            if is_reserve and not in_bench:
                in_bench = True
                lines.append(f" {'-'*8} {'-'*4} {'-'*20} {'-'*6} {'-'*6} {'-'*5} {'-'*7} {'-'*9} {'-'*15}")
                lines.append("  >>> 💎 LUXURY ADVISORY BENCH (9 BEST-IN-SLOT UPGRADES: 1 GOL, 2 LAT, 2 ZAG, 2 MEI, 2 ATA) <<<")
                lines.append("      (Ranked strictly by points expectation — unconstrained upgrade options for bankrolls > C$135)")
                lines.append(f" {'-'*8} {'-'*4} {'-'*20} {'-'*6} {'-'*6} {'-'*5} {'-'*7} {'-'*9} {'-'*15}")

            loc = "Home" if r.get("is_home", 0) == 1 else "Away"
            p_name = str(r["name"])[:19]
            t_abbr = str(r.get("team_abbr", r.get("team_id", "?")))[:5]
            o_abbr = str(r.get("opp_abbr", r.get("opp_id", "?")))[:5]
            price_str = f"C${float(r.get('pre_match_price', 0.0)):.1f}"
            pred_str = f"{float(r.get(ENSEMBLE_COL, 0.0)):.2f}"
            votes = int(r.get("n_votes", 0))
            stars = "★" * votes + "☆" * (5 - votes)
            lines.append(f" {slot:<8} {r['position_code']:<4} {p_name:<20} {t_abbr:<6} {o_abbr:<6} {loc:<5} {price_str:<7} {pred_str:<9} {stars} ({votes}/5)")

    return "\n".join(lines)


def format_strategic_rosters_terminal(suite: dict, consensus: pd.DataFrame) -> str:
    """Formats the 4 distinct Strategic Roster Compositions and Defensive Lock audit for terminal display."""
    lines = []
    def_locks = suite.get("defensive_locks_table", pd.DataFrame())
    if len(def_locks) > 0:
        lines.append(f"\n{'='*98}")
        lines.append("  🛡️  FECHAMENTO DE DEFESA (5-MAN CORRELATED DEFENSIVE LOCK CANDIDATES)")
        lines.append(f"{'='*98}")
        lines.append(f" {'CLUB':<6} {'OPP':<6} {'LOC':<5} {'P(SG)':<8} {'THRESHOLD':<12} {'5-MAN COST':<11} {'MEAN PTS':<10} {'IF SG HOLDS (+25)'}")
        lines.append(f" {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*11} {'-'*10} {'-'*20}")
        for _, dl in def_locks.head(5).iterrows():
            loc = "Home" if dl["is_home"] == 1 else "Away"
            psg_s = f"{dl['p_sg']*100:.1f}%"
            thr_s = "✅ QUALIFIED" if dl["meets_sg_threshold"] else "⚠️ BELOW 38%"
            cost_s = f"C${dl['unit_price']:.1f}"
            mn_s = f"{dl['unit_expected_mean']:.1f} pts"
            ceil_s = f"{dl['unit_ceiling_if_sg']:.1f} pts"
            lines.append(f" {dl['team_abbr']:<6} {dl['opp_abbr']:<6} {loc:<5} {psg_s:<8} {thr_s:<12} {cost_s:<11} {mn_s:<10} {ceil_s}")

    for key, pkg in suite.get("rosters", {}).items():
        sq = pkg["squad"].merge(consensus, on="player_id", how="left")
        sq["n_votes"] = sq["n_votes"].fillna(0).astype(int)
        meta = pkg["meta"]

        lines.append(f"\n{'='*98}")
        lines.append(f"  ♟️  STRATEGIC PORTFOLIO: {meta['strategy_title']}")
        lines.append(f"     {meta['strategy_desc']}")
        lines.append(f"     Starters Cost: C${meta['total_price']:.2f} + Bench: C${meta.get('bench_price', 0.0):.2f} / Budget: C${meta['budget']:.1f} | "
                     f"Correlated Floor (P10): {meta['team_floor_p10']:.1f} | "
                     f"Mean: {meta['team_mean']:.1f} | "
                     f"Ceiling (P90): {meta['team_ceiling_p90']:.1f} | "
                     f"Roster: 12 Starters + 9 Reserves (21 Players) | "
                     f"Solids/Wildcards: {meta['n_solids']}S/{meta['n_wildcards']}W | "
                     f"Conflicts: {meta['n_conflicts']}")
        if meta["n_conflicts"] > 0:
            conf_str = "; ".join(f"{c['attacker']} vs {c['defender']} (-{c['penalty_pts']} pts)" for c in meta["conflicts"])
            lines.append(f"     ⚠️ Active Matchup Conflicts: {conf_str}")
        lines.append(f"{'='*98}")
        lines.append(f" {'SLOT':<8} {'POS':<4} {'ROLE':<9} {'PLAYER':<18} {'TEAM':<5} {'OPP':<5} {'PRICE':<7} {'ENS':<6} {'FLOOR':<6} {'CEIL':<6} {'SPREAD':<7} {'VOTES'}")
        lines.append(f" {'-'*8} {'-'*4} {'-'*9} {'-'*18} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")

        in_bench_roster = False
        for _, r in sq.iterrows():
            slot = str(r.get("lineup_slot", "STARTER"))[:8]
            is_reserve = slot.startswith("RES") or slot == "RESERVE"
            if is_reserve and not in_bench_roster:
                in_bench_roster = True
                lines.append(f" {'-'*8} {'-'*4} {'-'*9} {'-'*18} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")
                lines.append("  >>> 💎 LUXURY ADVISORY BENCH (9 BEST-IN-SLOT UPGRADES: 1 GOL, 2 LAT, 2 ZAG, 2 MEI, 2 ATA) <<<")
                lines.append("      (Ranked strictly by points expectation — unconstrained upgrade options for bankrolls > C$135)")
                lines.append(f" {'-'*8} {'-'*4} {'-'*9} {'-'*18} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")

            role = str(r.get("role_class", "SOLID"))[:8]
            p_name = str(r["name"])[:17]
            t_abbr = str(r.get("team_abbr", "?"))[:4]
            o_abbr = str(r.get("opp_abbr", "?"))[:4]
            price_str = f"C${float(r.get('pre_match_price', 0.0)):.1f}"
            ens = f"{float(r.get(ENSEMBLE_COL, 0.0)):.2f}"
            fl = f"{float(r.get('gen_floor_p10', 0.0)):.1f}"
            cl = f"{float(r.get('gen_ceiling_p90', 0.0)):.1f}"
            sp = f"+{float(r.get('upside_spread', 0.0)):.1f}"
            votes = int(r.get("n_votes", 0))
            stars = "★" * votes + "☆" * (5 - votes)
            lines.append(f" {slot:<8} {r['position_code']:<4} {role:<9} {p_name:<18} {t_abbr:<5} {o_abbr:<5} {price_str:<7} {ens:<6} {fl:<6} {cl:<6} {sp:<7} {stars}")

    return "\n".join(lines)


import report_generator


def generate_interactive_report(
    season: int,
    target_round: int,
    audit_info: dict,
    pool: pd.DataFrame,
    squads: dict,
    consensus: pd.DataFrame,
    meta_info: dict,
    strategic_suite: dict = None,
    budget: float = 135.0,
) -> Path:
    return report_generator.generate_interactive_report(
        season=season,
        target_round=target_round,
        audit_info=audit_info,
        pool=pool,
        squads=squads,
        consensus=consensus,
        meta_info=meta_info,
        strategic_suite=strategic_suite,
        budget=budget,
    )


def main():
    parser = argparse.ArgumentParser(description="CLF Automated Click-and-Go Cartola Round Inference")
    parser.add_argument("--season", type=int, default=None, help="Championship season (e.g. 2026)")
    parser.add_argument("--round", type=int, default=None, help="Target round to infer (e.g. 35)")
    parser.add_argument("--retrain", action="store_true", help="Force retraining and reserialization of model zoo")
    parser.add_argument("--top", type=int, default=15, help="Number of suggestions per specialist (default: 15)")
    parser.add_argument("--min-games", type=int, default=4, help="Minimum appearances in season to be in candidate pool")
    parser.add_argument("--budget", type=float, default=135.0, help="Cartoletas budget C$ for 12-starter portfolio optimization (default: 135.0)")
    parser.add_argument("--sg-threshold", type=float, default=0.38, help="Minimum P(SG) to qualify for 5-man Defensive Lock (default: 0.38)")
    parser.add_argument("--no-html", action="store_true", help="Skip generating HTML report")

    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=" * 80)
    print("  ⚽ CARTOLA LEGENDS OF THE FUTURE (CLF) — AUTOMATED INFERENCE ENGINE")
    print("=" * 80)

    # 1. Resolve Target Season & Round
    if args.season is None or args.round is None:
        det_season, det_round = round_fetcher.auto_detect_season_and_round()
        season = args.season if args.season is not None else det_season
        target_round = args.round if args.round is not None else det_round
        print(f"[Auto-Detection] Defaulted to Season {season}, Target Round {target_round}.")
    else:
        season = args.season
        target_round = args.round

    # 2. Check Training Grounds & Audit Status
    audit_info = round_fetcher.check_training_grounds(season, target_round)
    if not audit_info["is_future"]:
        print(f"\n⚠️  [AUDIT WARNING] {audit_info['message']}\n")
    else:
        print(f"\n✨ [LIVE MODE] {audit_info['message']}\n")

    # 3. Load or Retrain Model Registry
    models, weights, metadata = model_registry.ensure_models(force_retrain=args.retrain)

    # 4. Ingest, Validate & Build Pre-Match Features
    fs_mod = load_feature_store_module()
    print(f"\n[Pipeline] Building pre-match features for Season {season} Round {target_round}...")
    try:
        target_features, roster = round_fetcher.prepare_inference_feature_dataset(
            season, target_round, fs_mod
        )
    except round_fetcher.MissingScoutsError as e:
        print(f"\n❌ [ERROR: MISSING SCOUT DATA]\n{e}")
        sys.exit(1)
    except round_fetcher.MissingFixturesError as e:
        print(f"\n❌ [ERROR: MISSING FIXTURES]\n{e}")
        sys.exit(1)

    # 5. Run Specialist Predictions & Meta-Ensemble
    t_inf = time.perf_counter()
    pred_df = model_registry.predict_all_specialists(models, weights, target_features)
    print(f"[Inference] Generated predictions for {len(pred_df):,} candidates across all models in {time.perf_counter() - t_inf:.2f}s.")

    # 6. Candidate Pool Filtering
    pred_df = pred_df.merge(roster[["player_id", "status_id"]], on="player_id", how="left")
    pred_df["status"] = pred_df["status_id"].map(STATUS_MAP).fillna("?")
    pred_df["position_code"] = pred_df["position_id"].map(POS_CODE)
    pred_df["name"] = pred_df["canonical_nickname"].fillna(pred_df["canonical_name"]).fillna(pred_df["player_id"].astype(str))

    # Rotation regulars filter: drop injured (status 5), long inactive (>3 rounds), and fringe (<min_games)
    min_games = args.min_games if target_round > args.min_games + 2 else 1
    pool = pred_df[
        (pred_df["status_id"] != 5)
        & (pred_df["season_games_prior"] >= min_games)
        & (pred_df["rounds_since_last_played"] <= 3)
    ].copy()

    print(f"[Pool Filtering] Filtered {len(pred_df)} rostered players down to {len(pool)} active rotation candidates "
          f"(dropped Contundido, <{min_games} appearances, or inactive >3 rounds).")

    # 7. Squad Selection, Consensus Votes & Strategic Portfolio Suite (4 Rosters)
    squads, consensus = build_squads(pool)
    strategic_suite = portfolio_optimizer.build_strategic_roster_suite(
        pool, budget=args.budget, sg_threshold=args.sg_threshold
    )

    # 8. Terminal Output: 4 Strategic Lineup Packages + Individual Ranks
    print(format_strategic_rosters_terminal(strategic_suite, consensus))
    print(format_terminal_squad("CLF_Meta_Ensemble", squads["CLF_Meta_Ensemble"], consensus))

    # Generative Poisson Squads (Rank A: Mean, Rank B: Upside Spread, Rank C: Boom-or-Bust Multiplier)
    if "Generative_Expected_Mean" in squads:
        print(format_terminal_squad("Generative_Expected_Mean (Rank A: Safe Foundation)", squads["Generative_Expected_Mean"], consensus))
    if "Generative_Upside_Spread" in squads:
        print(format_terminal_squad("Generative_Upside_Spread (Rank B: Maximum Positive Jump +Δ)", squads["Generative_Upside_Spread"], consensus))
    if "Generative_Boom_Bust" in squads:
        print(format_terminal_squad("Generative_Boom_Bust (Rank C: Volatility Multiplier Lottery Differentials)", squads["Generative_Boom_Bust"], consensus))

    # Consensus Stars summary
    top_stars = pool.merge(consensus, on="player_id", how="left")
    top_stars = top_stars[top_stars["n_votes"] >= 3].sort_values("n_votes", ascending=False)
    print(f"\n{'='*78}")
    print(f"  🌟 CONSENSUS STARS (Selected by 3 or more Specialists)")
    print(f"{'='*78}")
    for _, r in top_stars.iterrows():
        stars = "★" * int(r["n_votes"]) + "☆" * (5 - int(r["n_votes"]))
        print(f"  {r['position_code']:<4} {r['name']:<20} {r['team_abbr']:<5} vs {r['opp_abbr']:<5} | C${float(r['pre_match_price']):.1f} | Pred: {float(r[ENSEMBLE_COL]):.2f} pts | {stars} ({int(r['n_votes'])}/5: {r['picked_by']})")

    # Specialist highlights
    print(f"\n{'='*78}")
    print("  📋 SPECIALIST SQUAD HIGHLIGHTS (Top 3 per Specialist)")
    print(f"{'='*78}")
    for s_name in SPECIALISTS:
        s_col = SPECIALISTS[s_name]
        top3 = squads[s_name].nlargest(3, s_col)
        top_str = ", ".join(f"{r['name']} ({r[s_col]:.2f})" for _, r in top3.iterrows())
        print(f"  • {s_name:<20}: {top_str}")

    # 9. Save Parquet & HTML Dashboard
    parquet_path = DATA_DIR / f"inference_{season}_r{target_round}.parquet"
    pred_df.to_parquet(parquet_path, index=False)
    print(f"\n[Artifacts] Full predictions saved -> {parquet_path}")

    if not args.no_html:
        report_path = generate_interactive_report(
            season, target_round, audit_info, pool, squads, consensus, metadata, strategic_suite=strategic_suite, budget=args.budget
        )
        print(f"[Artifacts] Standalone interactive HTML report -> {report_path}")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n✅ Total Execution Time: {elapsed_total:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
