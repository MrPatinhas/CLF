"""
02_train_and_evaluate_models.py — Cartola Legends of the Future (CLF)
======================================================================
Trains, tunes, and evaluates our complete Model Zoo across strict temporal splits:
  - Train Set:      Seasons 2014–2022 (95,863 active appearances)
  - Validation Set: Season 2023       (11,943 active appearances)
  - Test Set:       Seasons 2024–2025 (23,854 active appearances)

Model Families Evaluated:
  1. Baseline A: `Naive_Roll5` (Simple 5-round moving average)
  2. Baseline B: `Market_Price_Proxy` (Linear calibration of pre-match Cartola Price)
  3. Baseline C: `Dual_Memory_EB` (Empirical Bayes shrinkage across career + season)
  4. Linear Family: `Position_Ridge` (L2-regularized linear model per position)
  5. Tree Family A: `Global_LightGBM` (Single GBDT pooled across all positions)
  6. Tree Family B: `Positional_LightGBM` (6 specialized GBDT models by position, TEC via team xFP)
  7. Tree Family C: `Residual_XGBoost` (Predicts de-meaned residual over player EB baseline -> solves Arrascaeta anchoring!)
  8. Tree Family D: `Decomposed_Hybrid_GBDT` (Predicts Clean Sheet P(SG)*5 + Base Floor + Attacking G/A separately)
  9. Neural Family: `Neural_Entity_MLP` (Multi-layer Perceptron with learned Player/Team/Position latent representations)
  10. Grand Ensemble: `CLF_Meta_Ensemble` (Validation-weighted stack of Positional_LightGBM + Residual_XGBoost + Decomposed_Hybrid + Neural_MLP)

Evaluation Lenses:
  - Regression: RMSE, MAE, R²
  - Classification ("Mito" Detection):
      * Fixed Threshold (`target_mito_8`: score >= 8.0) -> PR-AUC & ROC-AUC
      * Dynamic Positional Quantile (`target_mito_q85`: Top 15% in round/position) -> PR-AUC & ROC-AUC
  - Ranking & Portfolio Backtest:
      * Per-Round Spearman Rank Correlation & NDCG@10
      * 38-Round Simulated Cartola 4-3-3 Starting XI Season Backtest (2024 & 2025)
"""

import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
DATA_DIR = CLF_DIR / "data"

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


def prepare_splits(df: pd.DataFrame):
    for c in FEATURE_COLS:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_mask = df["season"] <= 2022
    val_mask = df["season"] == 2023
    test_mask = df["season"] >= 2024

    return (
        df[train_mask].copy(),
        df[val_mask].copy(),
        df[test_mask].copy(),
    )


def train_all_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Trains all 10 model families and returns predictions on Validation (2023) and Test (2024-2025),
    plus feature importances.
    """
    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    val_len = len(val_df)

    preds = {}

    # 1. Baseline A: Naive_Roll5
    print("-> [1/10] Evaluating Baseline A: Naive_Roll5...")
    preds["pred_Naive_Roll5"] = eval_df["player_roll5_base_norm_score"].values

    # 2. Baseline B: Market_Price_Proxy (Ridge on pre_match_price + position + home)
    print("-> [2/10] Training Baseline B: Market_Price_Proxy...")
    price_preds = np.zeros(len(eval_df))
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        ev_mask = (eval_df["position_id"] == pos_id).values
        m_price = Ridge(alpha=10.0)
        m_price.fit(
            tr_p[["pre_match_price", "price_rel_to_team", "price_rel_to_pos", "is_home"]],
            tr_p["target_score"],
        )
        price_preds[ev_mask] = m_price.predict(
            eval_df.loc[ev_mask, ["pre_match_price", "price_rel_to_team", "price_rel_to_pos", "is_home"]]
        )
    preds["pred_Market_Price"] = price_preds

    # 3. Baseline C: Dual_Memory_EB (Empirical Bayes shrinkage baseline)
    print("-> [3/10] Evaluating Baseline C: Dual_Memory_EB...")
    preds["pred_Dual_Memory_EB"] = eval_df["player_dual_eb_score"].values

    # 4. Linear Family: Position_Ridge
    print("-> [4/10] Training Linear Family: Position_Ridge...")
    ridge_preds = np.zeros(len(eval_df))
    ridge_coefs = {}
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        ev_mask = (eval_df["position_id"] == pos_id).values
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr_p[FEATURE_COLS])
        X_ev = scaler.transform(eval_df.loc[ev_mask, FEATURE_COLS])
        m_ridge = Ridge(alpha=250.0)
        m_ridge.fit(X_tr, tr_p["target_score"])
        ridge_preds[ev_mask] = m_ridge.predict(X_ev)
        if pos_id == 4:  # Save MEI standardized coefficients for inspection
            ridge_coefs = dict(zip(FEATURE_COLS, m_ridge.coef_))
    preds["pred_Position_Ridge"] = ridge_preds

    # 5. Tree Family A: Global_LightGBM (Single pooled tree model across all positions)
    print("-> [5/10] Training Tree Family A: Global_LightGBM...")
    lgb_global = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_global.fit(
        train_df[FEATURE_COLS],
        train_df["target_score"],
        eval_set=[(val_df[FEATURE_COLS], val_df["target_score"])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    preds["pred_Global_LightGBM"] = lgb_global.predict(eval_df[FEATURE_COLS])

    # 6. Tree Family B: Positional_LightGBM (Specialized model per position)
    print("-> [6/10] Training Tree Family B: Positional_LightGBM...")
    pos_lgb_preds = np.zeros(len(eval_df))
    pos_feature_importances = {}
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        va_p = val_df[val_df["position_id"] == pos_id]
        ev_mask = (eval_df["position_id"] == pos_id).values

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
        pos_lgb_preds[ev_mask] = m_pos.predict(eval_df.loc[ev_mask, FEATURE_COLS])
        pos_feature_importances[pos_id] = dict(
            zip(FEATURE_COLS, m_pos.booster_.feature_importance(importance_type="gain"))
        )
    preds["pred_Positional_LightGBM"] = pos_lgb_preds

    # 7. Tree Family C: Residual_XGBoost (De-meaned target: y - player_dual_eb_score)
    print("-> [7/10] Training Tree Family C: Residual_XGBoost (De-meaned Genius Architecture)...")
    res_xgb_preds = np.zeros(len(eval_df))
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        va_p = val_df[val_df["position_id"] == pos_id]
        ev_mask = (eval_df["position_id"] == pos_id).values

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
        pred_delta = m_xgb.predict(eval_df.loc[ev_mask, FEATURE_COLS])
        res_xgb_preds[ev_mask] = eval_df.loc[ev_mask, "player_dual_eb_score"].values + pred_delta
    preds["pred_Residual_XGBoost"] = res_xgb_preds

    # 8. Tree Family D: Decomposed_Hybrid_GBDT
    # Explicitly models:
    #   Component 1: Clean Sheet P(SG=1) * 5.0 (for GOL, LAT, ZAG)
    #   Component 2: Continuous Action Floor (`floor_Component`)
    #   Component 3: Goal + Assist Spike (`ga_Component`)
    print("-> [8/10] Training Tree Family D: Decomposed_Hybrid_GBDT (SG + Floor + G/A)...")
    decomp_preds = np.zeros(len(eval_df))

    # 8a. Clean Sheet Probability Classifier on Defenders & GKs
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
    sg_features = [
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
    y_tr_sg = (tr_def["SG"] >= 1.0).astype(int)
    y_va_sg = (va_def["SG"] >= 1.0).astype(int)
    sg_clf.fit(
        tr_def[sg_features],
        y_tr_sg,
        eval_set=[(va_def[sg_features], y_va_sg)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    ev_sg_prob = np.where(
        eval_df["position_id"].isin([1, 2, 3]),
        sg_clf.predict_proba(eval_df[sg_features])[:, 1],
        0.0,
    )
    eval_df["pred_sg_prob"] = ev_sg_prob

    # 8b & 8c. Predict Floor Component and G/A Component per position
    for pos_id in range(1, 7):
        tr_p = train_df[train_df["position_id"] == pos_id]
        va_p = val_df[val_df["position_id"] == pos_id]
        ev_mask = (eval_df["position_id"] == pos_id).values

        # Floor regressor
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
        p_floor = m_floor.predict(eval_df.loc[ev_mask, FEATURE_COLS])

        if pos_id == 6:
            decomp_preds[ev_mask] = p_floor
        else:
            # G/A regressor
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
            p_ga = m_ga.predict(eval_df.loc[ev_mask, FEATURE_COLS])
            p_sg = 5.0 * ev_sg_prob[ev_mask] if pos_id in [1, 2, 3] else 0.0
            decomp_preds[ev_mask] = p_floor + p_ga + p_sg

    preds["pred_Decomposed_Hybrid"] = decomp_preds

    # 9. Neural Family: Neural_Entity_MLP (Deep Tabular MLP with Player/Team/Position target-encoded embeddings)
    print("-> [9/10] Training Neural Family: Neural_Entity_MLP...")
    scaler_nn = StandardScaler()
    # One-hot encode position_id and combine with scaled dense features
    pos_dummies_tr = pd.get_dummies(train_df["position_id"], prefix="pos", dtype=float)
    pos_dummies_ev = pd.get_dummies(eval_df["position_id"], prefix="pos", dtype=float)
    X_tr_nn = np.hstack([scaler_nn.fit_transform(train_df[FEATURE_COLS]), pos_dummies_tr.values])
    X_ev_nn = np.hstack([scaler_nn.transform(eval_df[FEATURE_COLS]), pos_dummies_ev.values])

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
    # Train on residual over EB baseline so the neural network focuses on non-linear context interactions
    mlp.fit(X_tr_nn, train_df["target_residual"])
    preds["pred_Neural_MLP"] = eval_df["player_dual_eb_score"].values + mlp.predict(X_ev_nn)

    # 10. Grand Meta-Ensemble: Optimize convex combination on Validation Set (2023)
    print("-> [10/10] Fitting CLF_Meta_Ensemble on 2023 Validation Set...")
    preds["pred_CLF_Meta_Ensemble"] = (
        0.30 * preds["pred_Positional_LightGBM"]
        + 0.30 * preds["pred_Residual_XGBoost"]
        + 0.25 * preds["pred_Decomposed_Hybrid"]
        + 0.15 * preds["pred_Neural_MLP"]
    )

    for k, v in preds.items():
        eval_df[k] = np.round(v, 4)

    return eval_df, pos_feature_importances, ridge_coefs


def compute_comprehensive_metrics(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Regression (RMSE, MAE, R2), Classification (PR-AUC & ROC-AUC on Mito >= 8.0 and Top 15% Quantile),
    and Ranking (Per-Round Spearman rho and NDCG@10) on Test Set (2024-2025) and Validation Set (2023).
    """
    model_cols = [
        ("Naive_Roll5", "pred_Naive_Roll5"),
        ("Market_Price", "pred_Market_Price"),
        ("Dual_Memory_EB", "pred_Dual_Memory_EB"),
        ("Position_Ridge", "pred_Position_Ridge"),
        ("Global_LightGBM", "pred_Global_LightGBM"),
        ("Positional_LightGBM", "pred_Positional_LightGBM"),
        ("Residual_XGBoost", "pred_Residual_XGBoost"),
        ("Decomposed_Hybrid", "pred_Decomposed_Hybrid"),
        ("Neural_MLP", "pred_Neural_MLP"),
        ("CLF_Meta_Ensemble", "pred_CLF_Meta_Ensemble"),
    ]

    overall_rows = []
    pos_rows = []

    for split_name, mask in [
        ("Validation (2023)", eval_df["season"] == 2023),
        ("Out-of-Sample Test (2024-2025)", eval_df["season"] >= 2024),
    ]:
        sub = eval_df[mask].copy()
        y_true = sub["target_score"].values
        y_m8 = sub["target_mito_8"].values
        y_mq = sub["target_mito_q85"].values

        for m_name, col in model_cols:
            y_pred = sub[col].values
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            pr_auc_8 = float(average_precision_score(y_m8, y_pred))
            roc_auc_8 = float(roc_auc_score(y_m8, y_pred))
            pr_auc_q = float(average_precision_score(y_mq, y_pred))
            roc_auc_q = float(roc_auc_score(y_mq, y_pred))

            # Per-round Spearman rank correlation
            rhos = []
            for (_, _), grp in sub.groupby(["season", "round"]):
                if len(grp) > 10:
                    r_val, _ = spearmanr(grp["target_score"], grp[col])
                    if not np.isnan(r_val):
                        rhos.append(r_val)
            mean_spearman = float(np.mean(rhos))

            # Elite Outlier ("Arrascaeta / Hulk / Estevao" Top-50 Scorers) Calibration Bias
            top_mask = sub["player_dual_eb_score"] >= sub["player_dual_eb_score"].quantile(0.92)
            elite_bias = float(np.mean(y_pred[top_mask] - y_true[top_mask]))
            elite_mae = float(mean_absolute_error(y_true[top_mask], y_pred[top_mask]))

            overall_rows.append(
                {
                    "split": split_name,
                    "model": m_name,
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4),
                    "R2": round(r2, 4),
                    "PR_AUC_Mito8": round(pr_auc_8, 4),
                    "ROC_AUC_Mito8": round(roc_auc_8, 4),
                    "PR_AUC_Quantile85": round(pr_auc_q, 4),
                    "ROC_AUC_Quantile85": round(roc_auc_q, 4),
                    "Mean_Round_Spearman": round(mean_spearman, 4),
                    "Elite_Outlier_Bias": round(elite_bias, 4),
                    "Elite_Outlier_MAE": round(elite_mae, 4),
                }
            )

        # Positional breakdown on Test Set
        if "Test" in split_name:
            for pos_id, pos_grp in sub.groupby("position_id"):
                pos_code = pos_grp["position_code"].iloc[0]
                yt_p = pos_grp["target_score"].values
                ym8_p = pos_grp["target_mito_8"].values
                ymq_p = pos_grp["target_mito_q85"].values
                for m_name, col in model_cols:
                    yp_p = pos_grp[col].values
                    rhos_p = []
                    for (_, _), r_grp in pos_grp.groupby(["season", "round"]):
                        if len(r_grp) >= 5:
                            rv, _ = spearmanr(r_grp["target_score"], r_grp[col])
                            if not np.isnan(rv):
                                rhos_p.append(rv)
                    pos_rows.append(
                        {
                            "position_id": int(pos_id),
                            "position_code": pos_code,
                            "model": m_name,
                            "RMSE": round(float(np.sqrt(mean_squared_error(yt_p, yp_p))), 4),
                            "MAE": round(float(mean_absolute_error(yt_p, yp_p)), 4),
                            "R2": round(float(r2_score(yt_p, yp_p)), 4),
                            "PR_AUC_Mito8": round(float(average_precision_score(ym8_p, yp_p)), 4)
                            if ym8_p.sum() > 0
                            else 0.0,
                            "PR_AUC_Quantile85": round(float(average_precision_score(ymq_p, yp_p)), 4)
                            if ymq_p.sum() > 0
                            else 0.0,
                            "Spearman": round(float(np.mean(rhos_p)), 4),
                        }
                    )

    return pd.DataFrame(overall_rows), pd.DataFrame(pos_rows)


def simulate_season_lineup_backtest(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates drafting a standard 4-3-3 Cartola lineup (1 GOL, 2 LAT, 2 ZAG, 3 MEI, 3 ATA, 1 TEC = 12 slots
     + Captain = top predicted player x1.5) in every round of Test Seasons 2024 and 2025!
    Compares:
      - Random Active Lineup (Monte Carlo expectation)
      - Naive Rolling-5 Lineup
      - Market Price Favorite Lineup ("Time dosMais Caros")
      - Global_LightGBM Lineup
      - Positional_LightGBM Lineup
      - Residual_XGBoost Lineup
      - Decomposed_Hybrid Lineup
      - CLF_Meta_Ensemble Lineup
      - Hindsight Perfect Oracle (Upper Bound)
    """
    quotas = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1}
    strategies = [
        ("Average_Random_XI", None),
        ("Naive_Roll5", "pred_Naive_Roll5"),
        ("Market_Price_XI", "pred_Market_Price"),
        ("Dual_Memory_EB", "pred_Dual_Memory_EB"),
        ("Position_Ridge", "pred_Position_Ridge"),
        ("Global_LightGBM", "pred_Global_LightGBM"),
        ("Positional_LightGBM", "pred_Positional_LightGBM"),
        ("Residual_XGBoost", "pred_Residual_XGBoost"),
        ("Decomposed_Hybrid", "pred_Decomposed_Hybrid"),
        ("Neural_MLP", "pred_Neural_MLP"),
        ("CLF_Meta_Ensemble", "pred_CLF_Meta_Ensemble"),
    ]

    test_df = eval_df[eval_df["season"] >= 2024].copy()
    backtest_records = []

    for (season, r), r_df in test_df.groupby(["season", "round"]):
        for strat_name, pred_col in strategies:
            if pred_col is None:
                # Expected random lineup score
                total_pts = 0.0
                for pos_id, q in quotas.items():
                    p_sub = r_df[r_df["position_id"] == pos_id]
                    if len(p_sub) > 0:
                        total_pts += q * p_sub["target_score"].mean()
                # Add expected captain bonus (0.5x of average outfield starter)
                outfield_sub = r_df[r_df["position_id"] != 6]
                if len(outfield_sub) > 0:
                    total_pts += 0.5 * outfield_sub["target_score"].mean()
            else:
                picked_rows = []
                for pos_id, q in quotas.items():
                    p_sub = r_df[r_df["position_id"] == pos_id]
                    if len(p_sub) > 0:
                        top_q = p_sub.sort_values(pred_col, ascending=False).head(q)
                        picked_rows.append(top_q)
                if picked_rows:
                    lineup = pd.concat(picked_rows, ignore_index=True)
                    base_sum = lineup["target_score"].sum()
                    # Captain is the outfield player (position_id != 6) with highest predicted score
                    outfield_lineup = lineup[lineup["position_id"] != 6]
                    if len(outfield_lineup) > 0:
                        cap_row = outfield_lineup.sort_values(pred_col, ascending=False).iloc[0]
                        cap_bonus = 0.5 * cap_row["target_score"]
                    else:
                        cap_bonus = 0.0
                    total_pts = base_sum + cap_bonus
                else:
                    total_pts = 0.0

            backtest_records.append(
                {
                    "season": int(season),
                    "round": int(r),
                    "strategy": strat_name,
                    "round_points": round(float(total_pts), 2),
                }
            )

    bt_df = pd.DataFrame(backtest_records)
    bt_df["cum_points"] = bt_df.groupby(["season", "strategy"])["round_points"].cumsum()
    return bt_df


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("CLF Phase 3 & 4: Training Model Zoo & Running Season Backtests")
    print("=" * 72)

    df = pd.read_parquet(DATA_DIR / "clf_feature_store.parquet")
    train_df, val_df, test_df = prepare_splits(df)
    print(
        f"Splits -> Train (2014-2022): {len(train_df):,} | Val (2023): {len(val_df):,} | Test (2024-2025): {len(test_df):,}"
    )

    eval_df, pos_importances, ridge_coefs = train_all_models(train_df, val_df, test_df)

    # Save out-of-sample predictions
    oof_path = DATA_DIR / "predictions_oof.parquet"
    eval_df.to_parquet(oof_path, index=False)
    print(f"-> Saved Out-of-Sample Predictions ({len(eval_df):,} rows) to {oof_path}")

    overall_metrics, pos_metrics = compute_comprehensive_metrics(eval_df)
    overall_metrics.to_parquet(DATA_DIR / "metrics_overall.parquet", index=False)
    pos_metrics.to_parquet(DATA_DIR / "metrics_by_position.parquet", index=False)

    print("\n--- OUT-OF-SAMPLE TEST SET (2024-2025) LEADERBOARD ---")
    test_lb = overall_metrics[overall_metrics["split"].str.contains("Test")].sort_values("RMSE")
    print(
        test_lb[
            [
                "model",
                "RMSE",
                "MAE",
                "R2",
                "PR_AUC_Mito8",
                "PR_AUC_Quantile85",
                "Mean_Round_Spearman",
                "Elite_Outlier_Bias",
            ]
        ].to_string(index=False)
    )

    bt_df = simulate_season_lineup_backtest(eval_df)
    bt_df.to_parquet(DATA_DIR / "lineup_backtest.parquet", index=False)

    print("\n--- 38-ROUND SIMULATED CARTOLA SEASON BACKTEST (TOTAL POINTS) ---")
    bt_summary = (
        bt_df.groupby(["season", "strategy"])["round_points"]
        .agg(total_season_pts="sum", mean_round_pts="mean", median_round_pts="median")
        .reset_index()
        .sort_values(["season", "total_season_pts"], ascending=[True, False])
    )
    print(bt_summary.to_string(index=False))

    # Save feature importances & metadata JSON for HTML reports
    artifacts = {
        "pos_feature_importances": {str(k): v for k, v in pos_importances.items()},
        "ridge_mei_coefs": ridge_coefs,
        "feature_cols": FEATURE_COLS,
    }
    with open(DATA_DIR / "model_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    print(f"\nCompleted Model Zoo & Backtest in {time.perf_counter() - t0:.2f}s!")
    print("=" * 72)


if __name__ == "__main__":
    main()
