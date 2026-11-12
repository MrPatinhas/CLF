"""
src/generative_specialist.py — CLF Generative Scout Poisson/NegBin Specialist
==============================================================================
Implements an explainable generative rate engine that models individual football
events (Goals, Assists, Tackles, Shots, Saves, Fouls, Clean Sheets) and runs
vectorized Monte Carlo simulations to extract exact floor (P10) and ceiling (P90)
distributions alongside an itemized scorecard.

Core Components:
  - 9 Poisson-Deviance LightGBM regressors predicting λ_s for each scout
  - 1 Bernoulli LightGBM classifier predicting P(SG)
  - Vectorized 10,000-draw Monte Carlo simulator per candidate
  - Itemized scorecard breakdown (exact points receipt)
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
DATA_DIR = CLF_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

SCOUTS = ["G", "A", "DS", "FD", "FF", "FS", "DE", "FC", "CA"]

SCOUT_WEIGHTS = {
    "G": 8.0,
    "A": 5.0,
    "SG": 5.0,
    "DS": 1.0,
    "FD": 1.2,
    "FF": 0.8,
    "FS": 0.5,
    "DE": 1.3,
    "FC": -0.3,
    "CA": -1.0,
}

SCOUT_LABELS = {
    "G": "Gols",
    "A": "Assistências",
    "SG": "Saldo de Gols",
    "DS": "Desarmes",
    "FD": "Finalizações Defendidas",
    "FF": "Finalizações Fora",
    "FS": "Faltas Sofridas",
    "DE": "Defesas (GOL)",
    "FC": "Faltas Cometidas",
    "CA": "Cartões Amarelos",
}

BASE_FEATURES = [
    "is_home",
    "position_id",
    "pre_match_price",
    "price_rel_to_team",
    "price_rel_to_pos",
    "player_dual_eb_score",
    "player_dual_eb_floor",
    "eb_team_pts_mean",
    "opp_eb_team_pts_mean",
    "expected_team_gf",
    "expected_opp_gf",
    "poisson_sg_prob",
    "rounds_since_last_played",
    "season_games_prior",
]

SG_FEATURES = [
    "is_home",
    "poisson_sg_prob",
    "expected_opp_gf",
    "expected_team_gf",
    "team_vs_opp_xg_diff",
    "eb_ga",
    "roll5_ga",
    "eb_clean_sheet",
    "roll5_clean_sheet",
    "opp_eb_gf",
    "opp_roll5_gf",
    "pre_match_price",
    "eb_team_avg_price",
    "opp_eb_team_pts_mean",
]


def get_scout_features(scout: str, available_cols: List[str]) -> List[str]:
    cols = [c for c in BASE_FEATURES if c in available_cols]
    if f"player_roll5_{scout}" in available_cols:
        cols.append(f"player_roll5_{scout}")
    if f"player_ewma_{scout}" in available_cols:
        cols.append(f"player_ewma_{scout}")
    if f"opp_eb_conceded_{scout}" in available_cols:
        cols.append(f"opp_eb_conceded_{scout}")
    if f"opp_roll5_conceded_{scout}" in available_cols:
        cols.append(f"opp_roll5_conceded_{scout}")
    return cols


def train_generative_specialist(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Trains Poisson-deviance models for each scoring scout and binary SG classifier.
    """
    t0 = time.perf_counter()
    print("-> Training Generative Scout Specialist (9 Poisson Models + 1 SG Classifier)...")
    models = {}
    feature_map = {}

    available = train_df.columns.tolist()

    # 1. Train Poisson models for all 9 core scouts
    for s in SCOUTS:
        f_cols = get_scout_features(s, available)
        feature_map[s] = f_cols
        y_tr = train_df[s].clip(lower=0.0).fillna(0.0)
        y_va = val_df[s].clip(lower=0.0).fillna(0.0)

        m = lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=160,
            learning_rate=0.03,
            num_leaves=18,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=5.0,
            random_state=42 + len(s),
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            train_df[f_cols],
            y_tr,
            eval_set=[(val_df[f_cols], y_va)],
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        models[s] = m

    # 2. Train SG Classifier for defenders/goalkeepers
    tr_def = train_df[train_df["position_id"].isin([1, 2, 3])]
    va_def = val_df[val_df["position_id"].isin([1, 2, 3])]
    sg_cols = [c for c in SG_FEATURES if c in available]
    feature_map["SG"] = sg_cols

    sg_clf = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=80,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    sg_clf.fit(
        tr_def[sg_cols],
        (tr_def["SG"] >= 1.0).astype(int),
        eval_set=[(va_def[sg_cols], (va_def["SG"] >= 1.0).astype(int))],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    models["SG"] = sg_clf

    artifact = {
        "models": models,
        "feature_map": feature_map,
        "scouts": SCOUTS,
        "weights": SCOUT_WEIGHTS,
        "labels": SCOUT_LABELS,
    }
    print(f"-> Generative Specialist trained in {time.perf_counter() - t0:.1f}s.")
    return artifact


def predict_scout_lambdas(artifact: Dict[str, Any], df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Predicts expected rates λ_s for each scout and P(SG) for each player.
    Returns (lambdas_df, expected_scores).
    """
    models = artifact["models"]
    feature_map = artifact["feature_map"]

    lambdas = {}
    for s in SCOUTS:
        m = models[s]
        f_cols = feature_map[s]
        X = df[f_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Ensure rates are strictly non-negative
        pred_lam = np.clip(m.predict(X), 0.0, None)
        lambdas[s] = pred_lam

    # Clean Sheet Probability
    sg_cols = feature_map["SG"]
    X_sg = df[sg_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_sg_raw = models["SG"].predict_proba(X_sg)[:, 1]
    # SG only applies to GOL (1), LAT (2), ZAG (3)
    is_def = df["position_id"].isin([1, 2, 3]).values
    p_sg = np.where(is_def, p_sg_raw, 0.0)
    lambdas["SG"] = p_sg

    lam_df = pd.DataFrame(lambdas, index=df.index)

    # Compute analytical expected score
    exp_score = np.zeros(len(df))
    for s in SCOUTS:
        exp_score += lam_df[s].values * SCOUT_WEIGHTS[s]
    exp_score += p_sg * SCOUT_WEIGHTS["SG"]

    # Coach (position 6) adjustment: Coach score is team per-player mean + SG contribution
    coach_mask = (df["position_id"] == 6).values
    if coach_mask.any():
        base_c = df.loc[coach_mask, "eb_team_pts_mean"].values if "eb_team_pts_mean" in df.columns else np.full(coach_mask.sum(), 4.2)
        sg_c = df.loc[coach_mask, "poisson_sg_prob"].values if "poisson_sg_prob" in df.columns else np.full(coach_mask.sum(), 0.30)
        # If eb_team_pts_mean is team sum (>15), divide by 11; if already per-player average (<15), use directly
        base_c_norm = np.where(base_c > 15.0, base_c / 11.0, base_c)
        exp_score[coach_mask] = np.clip(base_c_norm + 1.8 * sg_c, 2.5, 8.5)

    return lam_df, np.round(exp_score, 4)


def simulate_monte_carlo(
    lam_df: pd.DataFrame,
    pos_ids: np.ndarray,
    n_sims: int = 5000,
    random_seed: int = 42,
    exp_scores: np.ndarray = None,
) -> pd.DataFrame:
    """
    Runs 2D-vectorized Monte Carlo simulation across all players in parallel chunks.
    Computes Floor (P10), Median (P50), Ceiling (P90), P(Mito >= 8), P(SuperMito >= 12).
    """
    rng = np.random.default_rng(random_seed)
    n_players = len(lam_df)
    pos_arr = np.asarray(pos_ids, dtype=int)

    floor_p10 = np.zeros(n_players, dtype=float)
    median_p50 = np.zeros(n_players, dtype=float)
    ceiling_p90 = np.zeros(n_players, dtype=float)
    std_arr = np.zeros(n_players, dtype=float)
    prob_mito = np.zeros(n_players, dtype=float)
    prob_super = np.zeros(n_players, dtype=float)

    chunk_size = 2500
    for start in range(0, n_players, chunk_size):
        end = min(start + chunk_size, n_players)
        m = end - start
        sub_pos = pos_arr[start:end]
        sim_scores = np.zeros((m, n_sims), dtype=float)

        for s in SCOUTS:
            lam_vec = np.clip(lam_df[s].iloc[start:end].values, 1e-6, None)[:, None]
            w = SCOUT_WEIGHTS[s]
            if s == "DE":
                # Overdispersed Gamma-Poisson (Negative Binomial equivalent) for GK saves
                gk_mask = (sub_pos == 1) & (lam_vec[:, 0] > 0.01)
                draws = rng.poisson(lam=lam_vec, size=(m, n_sims))
                if gk_mask.any():
                    lam_gk = lam_vec[gk_mask, 0]
                    r_gk = np.maximum(0.5, lam_gk / 3.2)[:, None]
                    scale_gk = (lam_gk / r_gk[:, 0])[:, None]
                    gamma_rate = rng.gamma(shape=r_gk, scale=scale_gk, size=( gk_mask.sum(), n_sims))
                    draws[gk_mask] = rng.poisson(lam=gamma_rate)
            else:
                draws = rng.poisson(lam=lam_vec, size=(m, n_sims))
            sim_scores += draws * w

        # Clean sheet Bernoulli for GOL, LAT, ZAG
        def_mask = np.isin(sub_pos, [1, 2, 3])
        if def_mask.any():
            p_sg_vec = np.clip(lam_df["SG"].iloc[start:end].values[def_mask], 0.0, 1.0)[:, None]
            sg_draws = rng.binomial(n=1, p=p_sg_vec, size=(def_mask.sum(), n_sims))
            sim_scores[def_mask] += sg_draws * SCOUT_WEIGHTS["SG"]

        # Coaches (position 6)
        coach_mask = (sub_pos == 6)
        if coach_mask.any():
            if exp_scores is not None:
                c_mu = np.asarray(exp_scores[start:end])[coach_mask][:, None]
            else:
                c_mu = np.full((coach_mask.sum(), 1), 4.8)
            sim_scores[coach_mask] = rng.normal(loc=c_mu, scale=1.35, size=(coach_mask.sum(), n_sims))

        floor_p10[start:end] = np.percentile(sim_scores, 10, axis=1)
        median_p50[start:end] = np.percentile(sim_scores, 50, axis=1)
        ceiling_p90[start:end] = np.percentile(sim_scores, 90, axis=1)
        std_arr[start:end] = np.std(sim_scores, axis=1)
        prob_mito[start:end] = (sim_scores >= 8.0).mean(axis=1)
        prob_super[start:end] = (sim_scores >= 12.0).mean(axis=1)

    return pd.DataFrame({
        "gen_floor_p10": np.round(floor_p10, 2),
        "gen_median_p50": np.round(median_p50, 2),
        "gen_ceiling_p90": np.round(ceiling_p90, 2),
        "gen_std": np.round(std_arr, 2),
        "gen_prob_mito": np.round(prob_mito, 3),
        "gen_prob_supermito": np.round(prob_super, 3),
    }, index=lam_df.index)


def generate_itemized_receipt(
    player_row: pd.Series,
    lam_row: pd.Series,
) -> List[Dict[str, Any]]:
    """
    Generates a readable, auditable line-item grocery receipt for a player.
    """
    receipt = []
    pos = player_row.get("position_id", 0)

    for s in SCOUTS:
        lam = float(lam_row.get(s, 0.0))
        w = SCOUT_WEIGHTS[s]
        contrib = lam * w
        if abs(contrib) >= 0.02 or s in ["G", "A", "DS"]:
            receipt.append({
                "scout": s,
                "label": SCOUT_LABELS.get(s, s),
                "expected_count": round(lam, 2),
                "weight": w,
                "points_contrib": round(contrib, 2),
            })

    if pos in [1, 2, 3]:
        p_sg = float(lam_row.get("SG", 0.0))
        receipt.append({
            "scout": "SG",
            "label": SCOUT_LABELS["SG"],
            "expected_count": f"{round(p_sg * 100, 1)}%",
            "weight": SCOUT_WEIGHTS["SG"],
            "points_contrib": round(p_sg * 5.0, 2),
        })

    receipt.sort(key=lambda x: abs(x["points_contrib"]), reverse=True)
    return receipt
