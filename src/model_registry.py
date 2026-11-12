"""
src/model_registry.py — CLF Model Registry & Persistence Module
================================================================
Implements serialization, deserialization, fast loading, and retraining for
all CLF specialist models and the Meta-Ensemble stack.

Serialized Artifacts (stored in data/models/):
  - market_price.joblib        Ridge regressors for pre-match price calibration (per position)
  - position_ridge.joblib      L2 Ridge regressors + scalers (per position)
  - positional_lgbm.joblib     6 LightGBM regressors tuned per position
  - residual_xgboost.joblib    6 XGBoost regressors predicting de-meaned genius delta
  - decomposed_hybrid.joblib   Clean-sheet classifier + 6 floor regressors + 5 G/A regressors
  - neural_mlp.joblib          Multi-layer Perceptron regressor + StandardScaler
  - meta_ensemble_weights.json Convex weights for the Grand Meta-Ensemble
  - registry_metadata.json     Timestamp, training seasons, feature count, validation metrics
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import generative_specialist as gs

warnings.filterwarnings("ignore")

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
DATA_DIR = CLF_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

FEATURE_COLS = [
    # Match & Context
    "is_home",
    "position_id",
    "pos_mean_anchor",
    "career_games_prior",
    "season_games_prior",
    "rounds_since_last_played",
    "played_last_round",
    # Market Wisdom (Strictly Pre-Match t-1)
    "pre_match_price",
    "prev_price_variation",
    "price_rel_to_team",
    "price_rel_to_pos",
    # Player Baselines & Dual-Memory Shrinkage
    "player_career_eb_score",
    "player_career_eb_floor",
    "player_career_eb_ga",
    "player_offseason_prior",
    "player_dual_eb_score",
    "player_dual_eb_floor",
    "player_genius_delta",
    "player_floor_delta",
    # Player Short/Medium Rolling & EWMA Windows
    "player_roll3_base_norm_score",
    "player_roll5_base_norm_score",
    "player_roll10_base_norm_score",
    "player_ewma_base_norm_score",
    "player_roll3_floor_Component",
    "player_roll5_floor_Component",
    "player_roll10_floor_Component",
    "player_ewma_floor_Component",
    "player_roll3_ga_Component",
    "player_roll5_ga_Component",
    "player_roll10_ga_Component",
    "player_ewma_ga_Component",
    "player_roll5_G",
    "player_ewma_G",
    "player_roll5_A",
    "player_ewma_A",
    "player_roll3_DS",
    "player_roll5_DS",
    "player_roll10_DS",
    "player_ewma_DS",
    "player_roll3_FD",
    "player_roll5_FD",
    "player_roll10_FD",
    "player_ewma_FD",
    "player_roll5_FF",
    "player_ewma_FF",
    "player_roll5_FS",
    "player_ewma_FS",
    "player_roll5_DE",
    "player_ewma_DE",
    "player_roll5_FC",
    "player_ewma_FC",
    "player_roll5_SG",
    "player_ewma_SG",
    # Player Volatility & Quantile Envelopes
    "player_roll10_std",
    "player_roll10_ceiling_q85",
    "player_roll10_floor_q25",
    # Own Team Form & Strength
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
    # Opponent Team Conceded & Vulnerability
    "opp_eb_gf",
    "opp_eb_ga",
    "opp_eb_clean_sheet",
    "opp_eb_team_pts_mean",
    "opp_eb_conceded_pts",
    "opp_eb_conceded_floor",
    "opp_eb_conceded_G",
    "opp_eb_conceded_A",
    "opp_eb_conceded_DS",
    "opp_eb_conceded_FD",
    "opp_roll5_gf",
    "opp_roll5_ga",
    "opp_roll5_conceded_pts",
    "opp_roll10_conceded_pts",
    "opp_roll5_conceded_floor",
    "opp_roll5_conceded_G",
    "opp_roll5_conceded_A",
    "opp_roll10_conceded_A",
    "opp_roll5_conceded_DS",
    "opp_roll10_conceded_DS",
    "opp_roll5_conceded_FD",
    # Opponent Positional Vulnerability
    "opp_pos_conceded_roll5_pos_pts_mean",
    "opp_pos_conceded_roll10_pos_pts_mean",
    "opp_pos_conceded_roll5_pos_floor_mean",
    "opp_pos_conceded_roll10_pos_floor_mean",
    "opp_pos_conceded_roll5_pos_DS_mean",
    "opp_pos_conceded_roll5_pos_FD_mean",
    "opp_pos_conceded_roll5_pos_DE_mean",
    # Tactical & Matchup Physics Interactions
    "team_vs_opp_xg_diff",
    "expected_team_gf",
    "expected_opp_gf",
    "poisson_sg_prob",
    "expected_sg_points",
    "ix_player_ga_vs_opp_conceded",
    "ix_player_ds_vs_opp_pos_ds",
    "ix_gk_de_vs_opp_attack",
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

DEFAULT_ENSEMBLE_WEIGHTS = {
    "Positional_LightGBM": 0.30,
    "Residual_XGBoost": 0.30,
    "Decomposed_Hybrid": 0.25,
    "Neural_MLP": 0.15,
}

REQUIRED_FILES = [
    "market_price.joblib",
    "position_ridge.joblib",
    "positional_lgbm.joblib",
    "residual_xgboost.joblib",
    "decomposed_hybrid.joblib",
    "neural_mlp.joblib",
    "generative_specialist.joblib",
    "meta_ensemble_weights.json",
    "registry_metadata.json",
]


def models_exist(models_dir: Path = MODELS_DIR) -> bool:
    """Checks whether all serialized model files exist in the target directory."""
    if not models_dir.exists():
        return False
    return all((models_dir / f).exists() for f in REQUIRED_FILES)


def train_and_save_models(
    feature_store_path: Path = DATA_DIR / "clf_feature_store.parquet",
    models_dir: Path = MODELS_DIR,
    train_max_season: int = 2024,
    val_season: int = 2025,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """
    Trains all specialists on historical data (2014..train_max_season) with validation
    on val_season, then serializes the trained models to models_dir.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("=" * 72)
    print("CLF Model Registry: Training & Serializing Model Zoo")
    print(f"Train Seasons: 2014-{train_max_season} | Early-Stopping Val: {val_season}")
    print(f"Target Directory: {models_dir}")
    print("=" * 72)

    if not feature_store_path.exists():
        raise FileNotFoundError(
            f"Feature store not found at {feature_store_path}. Run 01_build_feature_store.py first!"
        )

    df = pd.read_parquet(feature_store_path)
    for c in FEATURE_COLS:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_df = df[df["season"] <= train_max_season].copy()
    val_df = df[df["season"] == val_season].copy()

    print(f"Loaded {len(df):,} total rows -> Train: {len(train_df):,} | Val: {len(val_df):,}")

    models = {}

    # 1. Market Price Proxy
    print("1/5 Training Market Price Calibration per position...")
    market_price_models = {}
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        m_price = Ridge(alpha=10.0)
        m_price.fit(
            tr_p[["pre_match_price", "price_rel_to_team", "price_rel_to_pos", "is_home"]],
            tr_p["target_score"],
        )
        market_price_models[pos_id] = m_price
    joblib.dump(market_price_models, models_dir / "market_price.joblib")
    models["market_price"] = market_price_models

    # 2. Position Ridge
    print("2/5 Training Position Ridge per position...")
    position_ridge_models = {}
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr_p[FEATURE_COLS])
        m_ridge = Ridge(alpha=250.0)
        m_ridge.fit(X_tr, tr_p["target_score"])
        position_ridge_models[pos_id] = {"model": m_ridge, "scaler": scaler}
    joblib.dump(position_ridge_models, models_dir / "position_ridge.joblib")
    models["position_ridge"] = position_ridge_models

    # 3. Positional LightGBM
    print("3/5 Training Positional LightGBM per position...")
    pos_lgbm_models = {}
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        va_p = val_df[val_df["position_id"] == pos_id]
        m_pos = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.025,
            num_leaves=23 if pos_id not in [1, 6] else 15,
            min_child_samples=60 if pos_id not in [1, 6] else 30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=3.0,
            reg_lambda=8.0,
            random_state=42 + pos_id,
            n_jobs=-1,
            verbose=-1,
        )
        m_pos.fit(
            tr_p[FEATURE_COLS],
            tr_p["target_score"],
            eval_set=[(va_p[FEATURE_COLS], va_p["target_score"])],
            callbacks=[lgb.early_stopping(35, verbose=False)],
        )
        pos_lgbm_models[pos_id] = m_pos
    joblib.dump(pos_lgbm_models, models_dir / "positional_lgbm.joblib")
    models["positional_lgbm"] = pos_lgbm_models

    # 4. Residual XGBoost
    print("4/5 Training Residual XGBoost (de-meaned target) per position...")
    res_xgb_models = {}
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        va_p = val_df[val_df["position_id"] == pos_id]
        m_xgb = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=3.0,
            reg_lambda=10.0,
            random_state=100 + pos_id,
            n_jobs=-1,
            early_stopping_rounds=35,
            eval_metric="rmse",
        )
        m_xgb.fit(
            tr_p[FEATURE_COLS],
            tr_p["target_residual"],
            eval_set=[(va_p[FEATURE_COLS], va_p["target_residual"])],
            verbose=False,
        )
        res_xgb_models[pos_id] = m_xgb
    joblib.dump(res_xgb_models, models_dir / "residual_xgboost.joblib")
    models["residual_xgboost"] = res_xgb_models

    # 5. Decomposed Hybrid (SG Classifier + Floor + G/A)
    print("5/5 Training Decomposed Hybrid GBDT (SG + Floor + G/A)...")
    tr_def = train_df[train_df["position_id"].isin([1, 2, 3])]
    va_def = val_df[val_df["position_id"].isin([1, 2, 3])]
    sg_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=100,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    y_tr_sg = (tr_def["SG"] >= 1.0).astype(int)
    y_va_sg = (va_def["SG"] >= 1.0).astype(int)
    sg_clf.fit(
        tr_def[SG_FEATURES],
        y_tr_sg,
        eval_set=[(va_def[SG_FEATURES], y_va_sg)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    floor_models = {}
    ga_models = {}
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        va_p = val_df[val_df["position_id"] == pos_id]

        m_floor = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.025,
            num_leaves=20,
            min_child_samples=60,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=3.0,
            reg_lambda=8.0,
            random_state=200 + pos_id,
            n_jobs=-1,
            verbose=-1,
        )
        m_floor.fit(
            tr_p[FEATURE_COLS],
            tr_p["floor_Component"],
            eval_set=[(va_p[FEATURE_COLS], va_p["floor_Component"])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        floor_models[pos_id] = m_floor

        if pos_id != 6:
            m_ga = lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.02,
                num_leaves=15,
                min_child_samples=80,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=5.0,
                reg_lambda=12.0,
                random_state=300 + pos_id,
                n_jobs=-1,
                verbose=-1,
            )
            m_ga.fit(
                tr_p[FEATURE_COLS],
                tr_p["ga_Component"],
                eval_set=[(va_p[FEATURE_COLS], va_p["ga_Component"])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            ga_models[pos_id] = m_ga

    decomposed_hybrid = {
        "sg_clf": sg_clf,
        "floor_models": floor_models,
        "ga_models": ga_models,
        "sg_features": SG_FEATURES,
    }
    joblib.dump(decomposed_hybrid, models_dir / "decomposed_hybrid.joblib")
    models["decomposed_hybrid"] = decomposed_hybrid

    # 6. Neural MLP
    print("6/5 Training Neural Entity MLP on de-meaned residuals...")
    scaler_nn = StandardScaler()
    pos_dummies_tr = pd.get_dummies(train_df["position_id"], prefix="pos", dtype=float)
    for p in range(1, 7):
        col_name = f"pos_{p}"
        if col_name not in pos_dummies_tr.columns:
            pos_dummies_tr[col_name] = 0.0
    pos_dummies_tr = pos_dummies_tr[[f"pos_{p}" for p in range(1, 7)]]

    X_tr_nn = np.hstack([scaler_nn.fit_transform(train_df[FEATURE_COLS]), pos_dummies_tr.values])
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        alpha=2.5,
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=35,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42,
    )
    mlp.fit(X_tr_nn, train_df["target_residual"])
    neural_mlp = {"mlp": mlp, "scaler_nn": scaler_nn}
    joblib.dump(neural_mlp, models_dir / "neural_mlp.joblib")
    models["neural_mlp"] = neural_mlp

    # 7. Generative Poisson Specialist
    print("7/5 Training Generative Scout Specialist (Poisson / NegBin + Monte Carlo)...")
    gen_artifact = gs.train_generative_specialist(train_df, val_df)
    joblib.dump(gen_artifact, models_dir / "generative_specialist.joblib")
    models["generative_specialist"] = gen_artifact

    # Save Meta-Ensemble Weights & Metadata
    weights = DEFAULT_ENSEMBLE_WEIGHTS
    with open(models_dir / "meta_ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    metadata = {
        "train_seasons": f"2014-{train_max_season}",
        "val_season": val_season,
        "feature_count": len(FEATURE_COLS),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": [
            "Market_Price",
            "Position_Ridge",
            "Positional_LightGBM",
            "Residual_XGBoost",
            "Decomposed_Hybrid",
            "Neural_MLP",
            "Generative_Poisson",
            "CLF_Meta_Ensemble",
        ],
    }
    with open(models_dir / "registry_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"All models trained and saved to {models_dir} in {elapsed:.1f}s.")
    return models, weights, metadata


def load_models(
    models_dir: Path = MODELS_DIR,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """
    Loads all trained model artifacts from models_dir in < 250ms.
    """
    t0 = time.perf_counter()
    if not models_exist(models_dir):
        raise FileNotFoundError(
            f"Model artifacts missing from {models_dir}. Call train_and_save_models() first!"
        )

    models = {
        "market_price": joblib.load(models_dir / "market_price.joblib"),
        "position_ridge": joblib.load(models_dir / "position_ridge.joblib"),
        "positional_lgbm": joblib.load(models_dir / "positional_lgbm.joblib"),
        "residual_xgboost": joblib.load(models_dir / "residual_xgboost.joblib"),
        "decomposed_hybrid": joblib.load(models_dir / "decomposed_hybrid.joblib"),
        "neural_mlp": joblib.load(models_dir / "neural_mlp.joblib"),
        "generative_specialist": joblib.load(models_dir / "generative_specialist.joblib"),
    }

    with open(models_dir / "meta_ensemble_weights.json", "r", encoding="utf-8") as f:
        weights = json.load(f)

    with open(models_dir / "registry_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    elapsed = time.perf_counter() - t0
    return models, weights, metadata


def ensure_models(
    force_retrain: bool = False,
    models_dir: Path = MODELS_DIR,
    feature_store_path: Path = DATA_DIR / "clf_feature_store.parquet",
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """
    Returns the loaded model zoo. If models exist and force_retrain is False,
    loads them instantaneously from disk. Otherwise trains and saves them.
    """
    if not force_retrain and models_exist(models_dir):
        t0 = time.perf_counter()
        models, weights, metadata = load_models(models_dir)
        print(f"[Model Registry] Pre-trained models loaded from {models_dir.name}/ in {time.perf_counter() - t0:.2f}s "
              f"(Trained {metadata.get('trained_at', 'earlier')} on {metadata.get('train_seasons')}).")
        return models, weights, metadata

    print("[Model Registry] Models not found or --retrain requested. Training fresh model zoo...")
    return train_and_save_models(feature_store_path, models_dir)


def predict_all_specialists(
    models: Dict[str, Any],
    weights: Dict[str, float],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs inference for all baselines, specialists, and the Meta-Ensemble
    on a DataFrame containing populated FEATURE_COLS.
    """
    out = df.copy()

    # Pre-clean infinite / NaN values
    for c in FEATURE_COLS:
        if c in out.columns:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 1. Baseline A: Naive_Roll5
    if "player_roll5_base_norm_score" in out.columns:
        out["pred_Naive_Roll5"] = out["player_roll5_base_norm_score"].values
    else:
        out["pred_Naive_Roll5"] = 0.0

    # 2. Baseline B: Market Price Proxy
    price_preds = np.zeros(len(out))
    m_price_dict = models["market_price"]
    price_features = ["pre_match_price", "price_rel_to_team", "price_rel_to_pos", "is_home"]
    for pos_id, m_price in m_price_dict.items():
        mask = (out["position_id"] == pos_id).values
        if mask.any():
            price_preds[mask] = m_price.predict(out.loc[mask, price_features])
    out["pred_Market_Price"] = price_preds

    # 3. Baseline C: Dual_Memory_EB
    if "player_dual_eb_score" in out.columns:
        out["pred_Dual_Memory_EB"] = out["player_dual_eb_score"].values
    else:
        out["pred_Dual_Memory_EB"] = 0.0

    # 4. Position Ridge
    ridge_preds = np.zeros(len(out))
    ridge_dict = models["position_ridge"]
    for pos_id, obj in ridge_dict.items():
        mask = (out["position_id"] == pos_id).values
        if mask.any():
            X_sc = obj["scaler"].transform(out.loc[mask, FEATURE_COLS])
            ridge_preds[mask] = obj["model"].predict(X_sc)
    out["pred_Position_Ridge"] = ridge_preds

    # 5. Positional LightGBM
    pos_lgb_preds = np.zeros(len(out))
    pos_lgb_dict = models["positional_lgbm"]
    for pos_id, m_lgb in pos_lgb_dict.items():
        mask = (out["position_id"] == pos_id).values
        if mask.any():
            pos_lgb_preds[mask] = m_lgb.predict(out.loc[mask, FEATURE_COLS])
    out["pred_Positional_LightGBM"] = pos_lgb_preds

    # 6. Residual XGBoost (Predicts genius delta + dual EB)
    res_xgb_preds = np.zeros(len(out))
    res_xgb_dict = models["residual_xgboost"]
    dual_eb = out["player_dual_eb_score"].values if "player_dual_eb_score" in out.columns else np.zeros(len(out))
    for pos_id, m_xgb in res_xgb_dict.items():
        mask = (out["position_id"] == pos_id).values
        if mask.any():
            pred_delta = m_xgb.predict(out.loc[mask, FEATURE_COLS])
            res_xgb_preds[mask] = dual_eb[mask] + pred_delta
    out["pred_Residual_XGBoost"] = res_xgb_preds

    # 7. Decomposed Hybrid (SG Probability + Floor + G/A)
    decomp_dict = models["decomposed_hybrid"]
    sg_clf = decomp_dict["sg_clf"]
    floor_models = decomp_dict["floor_models"]
    ga_models = decomp_dict["ga_models"]
    sg_cols = decomp_dict.get("sg_features", SG_FEATURES)

    decomp_preds = np.zeros(len(out))
    ev_sg_prob = np.where(
        out["position_id"].isin([1, 2, 3]),
        sg_clf.predict_proba(out[sg_cols])[:, 1],
        0.0,
    )
    out["pred_sg_prob"] = np.round(ev_sg_prob, 4)

    for pos_id in range(1, 7):
        mask = (out["position_id"] == pos_id).values
        if not mask.any():
            continue
        m_floor = floor_models[pos_id]
        p_floor = m_floor.predict(out.loc[mask, FEATURE_COLS])

        if pos_id == 6:
            decomp_preds[mask] = p_floor
        else:
            m_ga = ga_models[pos_id]
            p_ga = m_ga.predict(out.loc[mask, FEATURE_COLS])
            p_sg = 5.0 * ev_sg_prob[mask] if pos_id in [1, 2, 3] else 0.0
            decomp_preds[mask] = p_floor + p_ga + p_sg
    out["pred_Decomposed_Hybrid"] = decomp_preds

    # 8. Neural MLP
    neural_dict = models["neural_mlp"]
    mlp = neural_dict["mlp"]
    scaler_nn = neural_dict["scaler_nn"]

    pos_dummies = pd.get_dummies(out["position_id"], prefix="pos", dtype=float)
    for p in range(1, 7):
        col_name = f"pos_{p}"
        if col_name not in pos_dummies.columns:
            pos_dummies[col_name] = 0.0
    pos_dummies = pos_dummies[[f"pos_{p}" for p in range(1, 7)]]

    X_nn = np.hstack([scaler_nn.transform(out[FEATURE_COLS]), pos_dummies.values])
    pred_nn_delta = mlp.predict(X_nn)
    out["pred_Neural_MLP"] = dual_eb + pred_nn_delta

    # 9. Generative Scout Poisson / NegBin Specialist & Monte Carlo Simulation
    if "generative_specialist" in models:
        gen_art = models["generative_specialist"]
        lam_df, exp_scores = gs.predict_scout_lambdas(gen_art, out)
        out["pred_Generative_Poisson"] = exp_scores
        pos_arr = out["position_id"].values if "position_id" in out.columns else np.zeros(len(out), dtype=int)
        mc_df = gs.simulate_monte_carlo(lam_df, pos_arr, n_sims=5000, exp_scores=exp_scores)
        for col in mc_df.columns:
            out[col] = mc_df[col].values
        for s in gs.SCOUTS:
            out[f"gen_lam_{s}"] = np.round(lam_df[s].values, 3)
        out["gen_lam_SG"] = np.round(lam_df["SG"].values, 3)
    else:
        out["pred_Generative_Poisson"] = 0.0

    # 10. Meta-Ensemble Stack
    out["pred_CLF_Meta_Ensemble"] = (
        weights.get("Positional_LightGBM", 0.30) * out["pred_Positional_LightGBM"]
        + weights.get("Residual_XGBoost", 0.30) * out["pred_Residual_XGBoost"]
        + weights.get("Decomposed_Hybrid", 0.25) * out["pred_Decomposed_Hybrid"]
        + weights.get("Neural_MLP", 0.15) * out["pred_Neural_MLP"]
    )

    pred_cols = [
        "pred_Naive_Roll5",
        "pred_Market_Price",
        "pred_Dual_Memory_EB",
        "pred_Position_Ridge",
        "pred_Positional_LightGBM",
        "pred_Residual_XGBoost",
        "pred_Decomposed_Hybrid",
        "pred_Neural_MLP",
        "pred_Generative_Poisson",
        "pred_CLF_Meta_Ensemble",
    ]
    for c in pred_cols:
        out[c] = np.round(out[c], 4)

    return out
