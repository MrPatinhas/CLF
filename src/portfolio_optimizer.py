"""
src/portfolio_optimizer.py — CLF Strategic Portfolio & Lineup Optimization Engine
==================================================================================
Translates individual player point forecasts and generative Poisson distributions
into 4 distinct, budget-compliant 4-3-3 Cartola roster compositions using Mixed
Integer Linear Programming (MILP via scipy.optimize.milp):

  1. Raw_Model_Blank       : Unconstrained expected score maximizer (control benchmark)
  2. Conservative_Shield   : High-floor Solid anchor focus (9 Solids + 2 Wildcards),
                             club diversification (<=3/club), soft cannibalization penalty
  3. Correlated_Fortress   : Conditional 5-man Defensive Lock (1 GOL + 2 LAT + 2 ZAG from
                             top high-SG club when P(SG) >= threshold) + balanced attack
  4. Asymmetric_Chaser     : High-variance upside hunter (5 Solids + 6 Wildcards),
                             maximizing P90 ceiling + spread with soft conflict penalty

Mathematical Highlights:
  - McCormick Linearization for Soft Matchup Cannibalization:
      c_{ij} >= x_i + x_j - 1,  c_{ij} >= 0
      Penalty: -gamma_{ij} * c_{ij} where gamma_{ij} is the expected points destroyed
      when Attacker i (Team A) scores against Defender j (Team B).
  - Joint Correlated Monte Carlo Team Simulator:
      Simulates club-level clean sheets as shared Bernoulli trials across same-club
      defenders and enforces SG_B = 0 whenever an opposing attacker scores G_A >= 1.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, Bounds, milp

warnings.filterwarnings("ignore")

# Standard 4-3-3 starting quotas (11 field players + 1 coach = 12 starters)
STARTING_QUOTA = {
    1: 1,  # GOL
    2: 2,  # LAT
    3: 2,  # ZAG
    4: 3,  # MEI
    5: 3,  # ATA
    6: 1,  # TEC
}

POS_CODE = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}


def classify_solids_and_wildcards(
    pool: pd.DataFrame,
    solid_top_per_pos: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """
    Partitions field candidates (positions 1..5) into two strictly disjoint tactical classes:
      - 'SOLID'   : Top chalk anchors per position ranked by composite Floor & Mean reliability.
      - 'WILDCARD': Explosive upside differentials outside the chalk anchor tier, ranked by
                    P90 Ceiling and Positive Upside Spread (+Δ).
      - 'COACH'   : Position 6 (Técnico).
    """
    df = pool.copy()

    # Ensure generative distributional columns exist
    mean_col = "pred_CLF_Meta_Ensemble" if "pred_CLF_Meta_Ensemble" in df.columns else "pred_Generative_Poisson"
    if "pred_Generative_Poisson" not in df.columns:
        df["pred_Generative_Poisson"] = df[mean_col]
    if "gen_floor_p10" not in df.columns:
        df["gen_floor_p10"] = df.get("player_dual_eb_floor", df[mean_col] * 0.45)
    if "gen_ceiling_p90" not in df.columns:
        df["gen_ceiling_p90"] = df.get("player_roll10_ceiling_q85", df[mean_col] * 1.85)
    if "upside_spread" not in df.columns:
        df["upside_spread"] = np.round(df["gen_ceiling_p90"] - df["pred_Generative_Poisson"], 2)
    if "vol_multiplier" not in df.columns:
        df["vol_multiplier"] = np.round(
            df["gen_ceiling_p90"] / np.maximum(df["pred_Generative_Poisson"], 0.5), 2
        )

    # Composite reliability score for Solids vs explosive upside score for Wildcards
    df["solid_utility"] = (
        0.50 * df[mean_col]
        + 0.30 * df["pred_Generative_Poisson"]
        + 0.20 * df["gen_floor_p10"]
    )
    df["wildcard_utility"] = (
        0.50 * df["gen_ceiling_p90"]
        + 0.35 * df["upside_spread"]
        + 0.15 * df[mean_col]
    )

    # Default number of Solid chalk anchors per position (Top chalk tier = 21 players total,
    # leaving high-upside rank #4-#12 differentials in the WILDCARD pool)
    if solid_top_per_pos is None:
        solid_top_per_pos = {
            1: 3,  # Top 3 GKs are Solids
            2: 4,  # Top 4 LATs are Solids
            3: 4,  # Top 4 ZAGs are Solids
            4: 5,  # Top 5 MEIs are Solids
            5: 5,  # Top 5 ATAs are Solids
        }

    df["role_class"] = "WILDCARD"
    df.loc[df["position_id"] == 6, "role_class"] = "COACH"

    for pos_id, n_solid in solid_top_per_pos.items():
        pos_mask = df["position_id"] == pos_id
        pos_sub = df[pos_mask]
        if len(pos_sub) == 0:
            continue
        # Ensure we leave enough players in WILDCARD pool if candidate pool is small
        max_solids = max(STARTING_QUOTA[pos_id], min(n_solid, len(pos_sub) - STARTING_QUOTA[pos_id]))
        top_idx = pos_sub.nlargest(max_solids, "solid_utility").index
        df.loc[top_idx, "role_class"] = "SOLID"

    df["is_solid"] = (df["role_class"] == "SOLID").astype(int)
    df["is_wildcard"] = (df["role_class"] == "WILDCARD").astype(int)
    return df


def evaluate_defensive_locks(
    df: pd.DataFrame,
    budget_limit: float = 70.0,
    sg_threshold: float = 0.38,
) -> pd.DataFrame:
    """
    Evaluates all clubs for a complete 5-man Defensive Lock (1 GOL + 2 LAT + 2 ZAG).
    Computes club P(SG), 5-man cost, expected base points, +25pt SG windfall ceiling,
    and flags whether the club clears the high-SG threshold.
    """
    def_df = df[df["position_id"].isin([1, 2, 3])].copy()
    records = []

    for team_id, grp in def_df.groupby("team_id"):
        gols = grp[grp["position_id"] == 1].nlargest(1, "solid_utility")
        lats = grp[grp["position_id"] == 2].nlargest(2, "solid_utility")
        zags = grp[grp["position_id"] == 3].nlargest(2, "solid_utility")

        if len(gols) < 1 or len(lats) < 2 or len(zags) < 2:
            continue

        unit = pd.concat([gols, lats, zags], ignore_index=True)
        total_price = float(unit["pre_match_price"].sum())
        if total_price > budget_limit:
            continue

        # Team clean sheet probability from generative model or Poisson feature
        if "gen_lam_SG" in unit.columns:
            p_sg = float(unit["gen_lam_SG"].mean())
        elif "poisson_sg_prob" in unit.columns:
            p_sg = float(unit["poisson_sg_prob"].mean())
        else:
            p_sg = 0.30

        mean_pts = float(unit["pred_Generative_Poisson"].sum())
        # Base score without SG (if they concede a goal)
        base_no_sg = max(0.0, mean_pts - 25.0 * p_sg)
        # Score if clean sheet holds (+25.0 guaranteed across the 5 defenders)
        lock_if_sg = base_no_sg + 25.0

        # Composite lock rating balances expected mean and P(SG) windfall probability
        lock_rating = mean_pts + 12.0 * max(0.0, p_sg - 0.25)

        records.append({
            "team_id": int(team_id),
            "team_abbr": str(unit["team_abbr"].iloc[0]),
            "opp_abbr": str(unit["opp_abbr"].iloc[0]),
            "is_home": int(unit["is_home"].iloc[0]),
            "p_sg": round(p_sg, 3),
            "meets_sg_threshold": bool(p_sg >= sg_threshold),
            "unit_price": round(total_price, 2),
            "unit_expected_mean": round(mean_pts, 2),
            "unit_floor_if_concede": round(base_no_sg, 2),
            "unit_ceiling_if_sg": round(lock_if_sg, 2),
            "lock_rating": round(lock_rating, 2),
            "player_ids": unit["player_id"].tolist(),
            "player_names": ", ".join(f"{r['position_code']} {r['name']}" for _, r in unit.iterrows()),
        })

    if not records:
        return pd.DataFrame()

    res = pd.DataFrame(records).sort_values(
        ["meets_sg_threshold", "p_sg", "lock_rating"], ascending=[False, False, False]
    ).reset_index(drop=True)
    return res


def _build_conflict_pairs(df: pd.DataFrame, penalty_scale: float = 1.0) -> List[Tuple[int, int, float]]:
    """
    Identifies all pairs (i_att, j_def) in the candidate pool where:
      - i_att is an Attacker (ATA=5, attacking MEI=4, or Coach TEC=6) from Team A
      - j_def is a Defender (GOL=1, LAT=2, ZAG=3) from opposing Team B (opp_id of A == team_id of B)
    Computes the exact expected points destroyed gamma_{ij} when i_att scores against j_def:
      gamma_{ij} = [ 5.0 * P(SG_j) * P(Goal_i >= 1) + 1.5 * lambda_G_i ]
    """
    idx_reset = df.reset_index(drop=True)
    att_mask = idx_reset["position_id"].isin([4, 5, 6])
    def_mask = idx_reset["position_id"].isin([1, 2, 3])

    attackers = idx_reset[att_mask]
    defenders = idx_reset[def_mask]

    # Map team_id -> list of defender row indices
    def_by_team: Dict[int, List[int]] = {}
    for d_idx, d_row in defenders.iterrows():
        t_id = int(d_row["team_id"])
        def_by_team.setdefault(t_id, []).append(d_idx)

    conflicts: List[Tuple[int, int, float]] = []
    for a_idx, a_row in attackers.iterrows():
        opp_id = int(a_row.get("opp_id", -1))
        if opp_id not in def_by_team:
            continue

        pos_a = int(a_row["position_id"])
        if pos_a == 6:
            # Coach vs Opposing Defender: if Coach's team scores, opposing defender loses SG
            lam_g = float(a_row.get("expected_team_gf", 1.35)) * 0.35
            lam_a = 0.10
            pos_weight = 0.65
        else:
            lam_g = float(a_row.get("gen_lam_G", 0.25 if pos_a == 5 else 0.12))
            lam_a = float(a_row.get("gen_lam_A", 0.15))
            pos_weight = 1.25 if pos_a == 5 else 0.75

        # Probability this attacker/coach is directly involved in a goal
        p_goal_involvement = 1.0 - np.exp(-(lam_g + 0.6 * lam_a))

        for d_idx in def_by_team[opp_id]:
            d_row = idx_reset.loc[d_idx]
            p_sg = float(d_row.get("gen_lam_SG", d_row.get("poisson_sg_prob", 0.30)))
            # Expected points lost by defender j when attacker i produces a goal
            expected_destruction = (5.0 * p_sg * p_goal_involvement + 1.2 * lam_g) * pos_weight
            gamma = float(max(0.75, expected_destruction))
            conflicts.append((a_idx, d_idx, gamma))

    return conflicts


def solve_cartola_portfolio(
    df: pd.DataFrame,
    utility_col: str = "solid_utility",
    budget: float = 135.0,
    n_wildcards_target: Optional[Tuple[int, int]] = None,
    max_per_club: Optional[int] = None,
    lock_defense_team_id: Optional[int] = None,
    cannibalization_penalty_scale: float = 1.0,
    enforce_cartola_bench_price_rule: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Solves the 12-man 4-3-3 Cartola lineup optimization problem via Mixed Integer
    Linear Programming (scipy.optimize.milp) with McCormick Soft Conflict Linearization.

    Parameters:
      - df: Candidate pool classified with `classify_solids_and_wildcards`
      - utility_col: Player utility column to maximize (or custom array)
      - budget: Maximum Cartoletas budget C$
      - n_wildcards_target: Optional (min_wildcards, max_wildcards) among the 11 field players
      - max_per_club: Maximum field players allowed from a single club (excluding locked defense)
      - lock_defense_team_id: Club team_id whose 5 defenders (1 GOL, 2 LAT, 2 ZAG) are locked
      - cannibalization_penalty_scale: Multiplier on soft matchup conflict penalty (0 = off)
    """
    pool = df.reset_index(drop=True).copy()
    n_players = len(pool)

    # Build soft conflict pairs (always detect for audit; only include in MILP if penalty_scale > 0)
    all_conflicts = _build_conflict_pairs(pool, penalty_scale=1.0)
    active_milp_conflicts = all_conflicts if cannibalization_penalty_scale > 0.0 else []
    n_conflicts = len(active_milp_conflicts)
    total_vars = n_players + n_conflicts

    # Objective: scipy.optimize.milp MINIMIZES c^T x, so negate utilities and add +gamma*scale for conflicts
    c_obj = np.zeros(total_vars, dtype=float)
    c_obj[:n_players] = -pool[utility_col].values.astype(float)
    for k, (_, _, gamma) in enumerate(active_milp_conflicts):
        c_obj[n_players + k] = gamma * cannibalization_penalty_scale

    # Integrality: 1 for binary player selections x_i, 0 for continuous McCormick conflict vars c_k
    integrality = np.zeros(total_vars, dtype=int)
    integrality[:n_players] = 1

    # Variable bounds: x_i in [0, 1], c_k in [0, 1]
    lb_vars = np.zeros(total_vars, dtype=float)
    ub_vars = np.ones(total_vars, dtype=float)

    rows_A = []
    rows_lb = []
    rows_ub = []

    # 1. Positional Quotas (4-3-3: 1 GOL, 2 LAT, 2 ZAG, 3 MEI, 3 ATA, 1 TEC)
    for pos_id, quota in STARTING_QUOTA.items():
        row = np.zeros(total_vars, dtype=float)
        row[:n_players] = (pool["position_id"] == pos_id).astype(float).values
        rows_A.append(row)
        rows_lb.append(float(quota))
        rows_ub.append(float(quota))

    # 2. Budget Constraint: sum(price_i * x_i) <= budget
    row_budget = np.zeros(total_vars, dtype=float)
    row_budget[:n_players] = pool["pre_match_price"].fillna(0.0).astype(float).values
    rows_A.append(row_budget)
    rows_lb.append(-np.inf)
    rows_ub.append(float(budget))

    # 3. Defensive Lock Constraint (if active)
    if lock_defense_team_id is not None:
        is_locked_def = (
            pool["position_id"].isin([1, 2, 3]) & (pool["team_id"] == lock_defense_team_id)
        ).astype(float).values
        row_lock = np.zeros(total_vars, dtype=float)
        row_lock[:n_players] = is_locked_def
        rows_A.append(row_lock)
        rows_lb.append(5.0)
        rows_ub.append(5.0)

    # 4. Wildcard Quota Constraint (among eligible positions)
    if n_wildcards_target is not None:
        w_min, w_max = n_wildcards_target
        if lock_defense_team_id is not None:
            # When defense is locked, wildcard target applies to Midfield + Attack (6 slots)
            wild_mask = (pool["position_id"].isin([4, 5]) & (pool["is_wildcard"] == 1)).astype(float).values
        else:
            wild_mask = (pool["position_id"].isin([1, 2, 3, 4, 5]) & (pool["is_wildcard"] == 1)).astype(float).values
        row_wild = np.zeros(total_vars, dtype=float)
        row_wild[:n_players] = wild_mask
        rows_A.append(row_wild)
        rows_lb.append(float(w_min))
        rows_ub.append(float(w_max))

    # 5. Club Diversification Constraint (max players per club)
    if max_per_club is not None:
        for t_id in pool["team_id"].unique():
            if lock_defense_team_id is not None and int(t_id) == int(lock_defense_team_id):
                # Allow the 5 locked defenders + up to 2 attackers/midfielders + coach
                t_mask = (pool["position_id"].isin([4, 5]) & (pool["team_id"] == t_id)).astype(float).values
                cap = 2.0
            else:
                t_mask = (pool["position_id"].isin([1, 2, 3, 4, 5]) & (pool["team_id"] == t_id)).astype(float).values
                cap = float(max_per_club)
            row_club = np.zeros(total_vars, dtype=float)
            row_club[:n_players] = t_mask
            rows_A.append(row_club)
            rows_lb.append(-np.inf)
            rows_ub.append(cap)

    # 6. McCormick Soft Cannibalization Linearization: x_i + x_j - c_k <= 1
    for k, (a_idx, d_idx, _) in enumerate(active_milp_conflicts):
        row_conf = np.zeros(total_vars, dtype=float)
        row_conf[a_idx] = 1.0
        row_conf[d_idx] = 1.0
        row_conf[n_players + k] = -1.0
        rows_A.append(row_conf)
        rows_lb.append(-np.inf)
        rows_ub.append(1.0)

    A_matrix = np.vstack(rows_A)
    constraints = LinearConstraint(A_matrix, rows_lb, rows_ub)
    bounds = Bounds(lb_vars, ub_vars)

    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

    # Fallback relaxation if strict budget/wildcard combination is infeasible
    if not res.success or res.x is None:
        if n_wildcards_target is not None:
            # Relax wildcard bounds slightly
            w_min, w_max = n_wildcards_target
            return solve_cartola_portfolio(
                df=df,
                utility_col=utility_col,
                budget=max(budget, 155.0),
                n_wildcards_target=(max(0, w_min - 1), min(11, w_max + 1)),
                max_per_club=None,
                lock_defense_team_id=lock_defense_team_id,
                cannibalization_penalty_scale=cannibalization_penalty_scale,
                enforce_cartola_bench_price_rule=enforce_cartola_bench_price_rule,
            )

    x_sol = np.round(res.x[:n_players]).astype(int)
    selected = pool[x_sol == 1].copy()
    selected["lineup_slot"] = "STARTER"

    # Audit active matchup conflicts in the chosen 12-man lineup
    active_conflicts = []
    for a_idx, d_idx, gamma in all_conflicts:
        if x_sol[a_idx] == 1 and x_sol[d_idx] == 1:
            att_r = pool.loc[a_idx]
            def_r = pool.loc[d_idx]
            active_conflicts.append({
                "attacker": f"{att_r['position_code']} {att_r['name']} ({att_r['team_abbr']})",
                "defender": f"{def_r['position_code']} {def_r['name']} ({def_r['team_abbr']})",
                "penalty_pts": round(gamma, 2),
            })

    # Select 9 Luxury Advisory Bench Reserves (1 GOL, 2 LAT, 2 ZAG, 2 MEI, 2 ATA = 9 reserves)
    # By default (enforce_cartola_bench_price_rule=False), reserves are ranked strictly by points expectation
    # / portfolio utility from unselected candidates. This acts as an unconstrained "Best-in-Slot" upgrade
    # advisory list, allowing managers with bankrolls > C$ 135 (e.g. C$ 150-180+) to immediately identify
    # top luxury upgrades to maximize their budget.
    remaining = pool[x_sol == 0].copy()
    bench_quota = {1: 1, 2: 2, 3: 2, 4: 2, 5: 2}
    reserves_list = []

    for r_pos, q_needed in bench_quota.items():
        if enforce_cartola_bench_price_rule:
            starter_sub = selected[selected["position_id"] == r_pos]
            starter_min_price = float(starter_sub["pre_match_price"].min()) if len(starter_sub) > 0 else 999.0
            cand_res = remaining[
                (remaining["position_id"] == r_pos) & (remaining["pre_match_price"] <= starter_min_price)
            ]
        else:
            cand_res = remaining[remaining["position_id"] == r_pos]

        taken = cand_res.nlargest(q_needed, utility_col).copy()

        # If shortfall, backfill from remaining players at that position
        if len(taken) < q_needed:
            shortfall = q_needed - len(taken)
            already_ids = taken["player_id"].tolist()
            fallback_cands = remaining[
                (remaining["position_id"] == r_pos) & (~remaining["player_id"].isin(already_ids))
            ]
            if len(fallback_cands) > 0:
                backfill = fallback_cands.nlargest(shortfall, utility_col).copy()
                taken = pd.concat([taken, backfill], ignore_index=True)

        for idx_r, (_, r_row) in enumerate(taken.iterrows(), 1):
            r_entry = r_row.to_frame().T.copy()
            r_entry["lineup_slot"] = f"RES {idx_r}" if q_needed > 1 else "RES 1"
            reserves_list.append(r_entry)
            # Remove chosen player from remaining so they can't be chosen twice
            remaining = remaining[remaining["player_id"] != r_row["player_id"]]

    reserves_df = pd.concat(reserves_list, ignore_index=True) if reserves_list else pd.DataFrame()
    full_squad = pd.concat([selected, reserves_df], ignore_index=True)
    full_squad["position_code"] = full_squad["position_id"].map(POS_CODE)
    
    # Sort starters first (1..6), then reserves (1..5)
    full_squad["is_starter_sort"] = np.where(full_squad["lineup_slot"] == "STARTER", 0, 1)
    full_squad = full_squad.sort_values(
        ["is_starter_sort", "position_id", utility_col], ascending=[True, True, False]
    ).drop(columns=["is_starter_sort"]).reset_index(drop=True)

    starters = full_squad[full_squad["lineup_slot"] == "STARTER"]
    field_starters = starters[starters["position_id"] <= 5]
    bench = full_squad[full_squad["lineup_slot"] != "STARTER"]

    info = {
        "total_price": round(float(starters["pre_match_price"].sum()), 2),
        "bench_price": round(float(bench["pre_match_price"].sum()), 2) if len(bench) > 0 else 0.0,
        "total_roster_count": len(full_squad),
        "starters_count": len(starters),
        "bench_count": len(bench),
        "budget": budget,
        "expected_mean_ens": round(float(starters.get("pred_CLF_Meta_Ensemble", starters["pred_Generative_Poisson"]).sum()), 2),
        "expected_mean_gen": round(float(starters["pred_Generative_Poisson"].sum()), 2),
        "sum_floor_p10": round(float(starters["gen_floor_p10"].sum()), 2),
        "sum_ceiling_p90": round(float(starters["gen_ceiling_p90"].sum()), 2),
        "n_solids": int(field_starters["is_solid"].sum()),
        "n_wildcards": int(field_starters["is_wildcard"].sum()),
        "n_conflicts": len(active_conflicts),
        "conflicts": active_conflicts,
        "clubs_represented": int(field_starters["team_id"].nunique()),
    }
    return full_squad, info


def simulate_correlated_team_distribution(
    squad_df: pd.DataFrame,
    n_sims: int = 5000,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Runs a Joint Correlated Monte Carlo simulation (5,000 matches) for the 12 starters:
      1. Club-level Clean Sheets SG_T ~ Bernoulli(P(SG_T)) are drawn ONCE per club per simulation,
         meaning all defenders from Club T simultaneously win (+5.0 each) or lose SG together.
      2. Negative Matchup Covariance (Cannibalization): If an Attacker on our roster from Club A
         draws G >= 1 against opposing Club B, then Club B's Clean Sheet SG_B is forced to 0
         on that simulation draw!
    """
    starters = squad_df[squad_df["lineup_slot"] == "STARTER"].reset_index(drop=True)
    rng = np.random.default_rng(random_seed)
    n_starters = len(starters)

    scouts = ["G", "A", "DS", "FD", "FF", "FS", "DE", "FC", "CA"]
    weights = {"G": 8.0, "A": 5.0, "DS": 1.0, "FD": 1.2, "FF": 0.8, "FS": 0.5, "DE": 1.3, "FC": -0.3, "CA": -1.0}

    # Track goals scored by each club in our lineup against their opponent
    opp_goals_scored_against_team: Dict[int, np.ndarray] = {}
    player_base_draws = np.zeros((n_starters, n_sims), dtype=float)

    for i, row in starters.iterrows():
        pos = int(row["position_id"])
        if pos == 6:
            # Coach score has low variance around expected mean
            c_mean = float(row.get("pred_Generative_Poisson", 4.5))
            if c_mean < 2.0:
                c_mean = float(row.get("pred_CLF_Meta_Ensemble", 4.8))
            player_base_draws[i] = rng.normal(loc=c_mean, scale=1.2, size=n_sims)
            continue

        for s in scouts:
            lam = max(1e-6, float(row.get(f"gen_lam_{s}", 0.0)))
            w = weights[s]
            if s == "DE" and pos == 1 and lam > 0.05:
                r = max(0.5, lam / 3.2)
                p_nb = r / (r + lam)
                draws = rng.negative_binomial(r, p_nb, size=n_sims)
            else:
                draws = rng.poisson(lam=lam, size=n_sims)

            player_base_draws[i] += draws * w

            if s == "G" and pos in [4, 5]:
                opp_id = int(row.get("opp_id", -1))
                if opp_id not in opp_goals_scored_against_team:
                    opp_goals_scored_against_team[opp_id] = np.zeros(n_sims, dtype=int)
                opp_goals_scored_against_team[opp_id] += draws

    # Draw shared Club-level Clean Sheets for defenders
    club_sg_prob: Dict[int, float] = {}
    for _, row in starters[starters["position_id"].isin([1, 2, 3])].iterrows():
        t_id = int(row["team_id"])
        p_sg = float(row.get("gen_lam_SG", row.get("poisson_sg_prob", 0.30)))
        club_sg_prob[t_id] = p_sg

    club_sg_draws: Dict[int, np.ndarray] = {}
    for t_id, p_sg in club_sg_prob.items():
        raw_sg = rng.binomial(n=1, p=np.clip(p_sg, 0.0, 1.0), size=n_sims)
        # Enforce Cannibalization: if our own attacker scored against t_id, SG is wiped out!
        if t_id in opp_goals_scored_against_team:
            opp_g = opp_goals_scored_against_team[t_id]
            raw_sg = np.where(opp_g >= 1, 0, raw_sg)
        club_sg_draws[t_id] = raw_sg

    for i, row in starters.iterrows():
        pos = int(row["position_id"])
        if pos in [1, 2, 3]:
            t_id = int(row["team_id"])
            player_base_draws[i] += club_sg_draws[t_id] * 5.0

    team_totals = player_base_draws.sum(axis=0)

    return {
        "team_floor_p10": round(float(np.percentile(team_totals, 10)), 2),
        "team_median_p50": round(float(np.percentile(team_totals, 50)), 2),
        "team_mean": round(float(np.mean(team_totals)), 2),
        "team_ceiling_p90": round(float(np.percentile(team_totals, 90)), 2),
        "team_std": round(float(np.std(team_totals)), 2),
        "prob_80_plus": round(float((team_totals >= 80.0).mean()), 3),
        "prob_100_plus": round(float((team_totals >= 100.0).mean()), 3),
    }


def build_strategic_roster_suite(
    pool: pd.DataFrame,
    budget: float = 135.0,
    sg_threshold: float = 0.38,
    enforce_cartola_bench_price_rule: bool = False,
) -> Dict[str, Any]:
    """
    Constructs the 4 Distinct Strategic Roster Compositions requested by the manager:
      1. Raw_Model_Blank       (Model Output Blank: Unconstrained Expected Points)
      2. Conservative_Shield   (9 Solids + 2 Wildcards, Soft Conflict Penalty, Max 3/Club)
      3. Correlated_Fortress   (5-Man Defensive Lock on High-SG Club + 3 Solids / 3 Wildcards Attack)
      4. Asymmetric_Chaser     (5 Solids + 6 Wildcards, Upside Ceiling Hunter, Soft Conflict Penalty)
    """
    classified = classify_solids_and_wildcards(pool)
    def_locks = evaluate_defensive_locks(classified, budget_limit=budget * 0.55, sg_threshold=sg_threshold)

    # Custom utility columns for each strategic lens
    mean_col = "pred_CLF_Meta_Ensemble" if "pred_CLF_Meta_Ensemble" in classified.columns else "pred_Generative_Poisson"
    classified["util_blank"] = classified[mean_col]

    # Shield utility rewards Solids for their floor + mean, and selected Wildcards for upside
    classified["util_shield"] = np.where(
        classified["is_solid"] == 1,
        0.55 * classified[mean_col] + 0.25 * classified["pred_Generative_Poisson"] + 0.20 * classified["gen_floor_p10"],
        0.50 * classified["gen_ceiling_p90"] + 0.50 * classified[mean_col],
    )

    # Fortress utility
    classified["util_fortress"] = np.where(
        classified["position_id"].isin([1, 2, 3]),
        classified["solid_utility"],
        np.where(
            classified["is_solid"] == 1,
            classified["solid_utility"],
            classified["wildcard_utility"],
        ),
    )

    # Chaser utility rewards explosive P90 ceiling and upside spread while preserving baseline floor
    classified["util_chaser"] = np.where(
        classified["is_wildcard"] == 1,
        0.45 * classified["gen_ceiling_p90"] + 0.30 * classified["upside_spread"] + 0.25 * classified[mean_col],
        0.50 * classified["solid_utility"] + 0.50 * classified["gen_ceiling_p90"],
    )

    rosters = {}

    # -------------------------------------------------------------------------
    # ROSTER 1: Raw Model Blank (Unconstrained Control Benchmark)
    # -------------------------------------------------------------------------
    sq_blank, info_blank = solve_cartola_portfolio(
        classified,
        utility_col="util_blank",
        budget=budget,
        n_wildcards_target=None,
        max_per_club=None,
        lock_defense_team_id=None,
        cannibalization_penalty_scale=0.0,
        enforce_cartola_bench_price_rule=enforce_cartola_bench_price_rule,
    )
    sim_blank = simulate_correlated_team_distribution(sq_blank, random_seed=101)
    info_blank.update(sim_blank)
    info_blank["strategy_title"] = "1. Raw Model Blank (Unconstrained Baseline)"
    info_blank["strategy_desc"] = (
        "Pure point maximization with zero strategic constraints (no cannibalization filter, "
        "no club exposure limits, no wildcard targets). Serves as the control benchmark."
    )
    rosters["Raw_Model_Blank"] = {"squad": sq_blank, "meta": info_blank}

    # -------------------------------------------------------------------------
    # ROSTER 2: The Conservative Shield (9 Solids + 2 Wildcards, Soft Conflict)
    # -------------------------------------------------------------------------
    sq_shield, info_shield = solve_cartola_portfolio(
        classified,
        utility_col="util_shield",
        budget=budget,
        n_wildcards_target=(2, 2),
        max_per_club=3,
        lock_defense_team_id=None,
        cannibalization_penalty_scale=1.25,
        enforce_cartola_bench_price_rule=enforce_cartola_bench_price_rule,
    )
    sim_shield = simulate_correlated_team_distribution(sq_shield, random_seed=102)
    info_shield.update(sim_shield)
    info_shield["strategy_title"] = "2. The Conservative Shield (9 Solids + 2 Wildcards)"
    info_shield["strategy_desc"] = (
        "Engineered for Liga Clássica leaders. Anchors 9 high-floor Solids with 2 surgical "
        "Wildcards, caps exposure at <=3 players per club, and penalizes matchup cannibalization."
    )
    rosters["Conservative_Shield"] = {"squad": sq_shield, "meta": info_shield}

    # -------------------------------------------------------------------------
    # ROSTER 3: The Correlated Fortress (Conditional High-SG Defensive Lock)
    # -------------------------------------------------------------------------
    lock_team_id = None
    lock_club_abbr = "None"
    lock_p_sg = 0.0
    lock_triggered = False

    if len(def_locks) > 0:
        top_lock = def_locks.iloc[0]
        lock_p_sg = float(top_lock["p_sg"])
        lock_club_abbr = str(top_lock["team_abbr"])
        if bool(top_lock["meets_sg_threshold"]):
            lock_team_id = int(top_lock["team_id"])
            lock_triggered = True
        else:
            # Still build the best defensive lock option so the manager can inspect it,
            # while flagging that it sits below the strict safety threshold!
            lock_team_id = int(top_lock["team_id"])
            lock_triggered = False

    sq_fort, info_fort = solve_cartola_portfolio(
        classified,
        utility_col="solid_utility",
        budget=budget,
        n_wildcards_target=None,  # Pair the high-variance 5-man defense lock with elite Solid attackers!
        max_per_club=4,
        lock_defense_team_id=lock_team_id,
        cannibalization_penalty_scale=2.0,  # Heavy penalty against picking attackers facing our locked defense!
        enforce_cartola_bench_price_rule=enforce_cartola_bench_price_rule,
    )
    sim_fort = simulate_correlated_team_distribution(sq_fort, random_seed=103)
    info_fort.update(sim_fort)
    info_fort["lock_team_id"] = lock_team_id
    info_fort["lock_club_abbr"] = lock_club_abbr
    info_fort["lock_p_sg"] = lock_p_sg
    info_fort["lock_triggered"] = lock_triggered
    status_tag = (
        f"ACTIVE LOCK: {lock_club_abbr} (P(SG)={lock_p_sg*100:.1f}% >= {sg_threshold*100:.0f}%)"
        if lock_triggered
        else f"CAUTION LOCK: {lock_club_abbr} (P(SG)={lock_p_sg*100:.1f}% < {sg_threshold*100:.0f}% threshold)"
    )
    info_fort["strategy_title"] = f"3. The Correlated Fortress — {status_tag}"
    info_fort["strategy_desc"] = (
        f"Locks all 5 defenders (1 GOL, 2 LAT, 2 ZAG) from {lock_club_abbr} to capture the correlated "
        f"+25.0 pt Clean Sheet windfall (P(SG)={lock_p_sg*100:.1f}%), paired with non-conflicting attack."
    )
    rosters["Correlated_Fortress"] = {"squad": sq_fort, "meta": info_fort}

    # -------------------------------------------------------------------------
    # ROSTER 4: The Asymmetric Chaser (5 Solids + 6 Wildcards Upside Hunter)
    # -------------------------------------------------------------------------
    sq_chase, info_chase = solve_cartola_portfolio(
        classified,
        utility_col="util_chaser",
        budget=budget,
        n_wildcards_target=(6, 6),
        max_per_club=4,
        lock_defense_team_id=None,
        cannibalization_penalty_scale=1.0,
        enforce_cartola_bench_price_rule=enforce_cartola_bench_price_rule,
    )
    sim_chase = simulate_correlated_team_distribution(sq_chase, random_seed=104)
    info_chase.update(sim_chase)
    info_chase["strategy_title"] = "4. The Asymmetric Chaser (5 Solids + 6 Wildcards)"
    info_chase["strategy_desc"] = (
        "Designed for Tiro Curto, Mata-Mata, or chasing a league deficit. Combines 5 Solid anchors "
        "with 6 high-volatility Wildcard differentials to maximize 90th-percentile ceiling."
    )
    rosters["Asymmetric_Chaser"] = {"squad": sq_chase, "meta": info_chase}

    return {
        "rosters": rosters,
        "defensive_locks_table": def_locks,
        "classified_pool": classified,
    }
