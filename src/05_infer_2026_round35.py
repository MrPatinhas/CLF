"""
05_infer_2026_round35.py — CLF Phase 6b: Feature engine + specialist inference for 2026 Round 35
=================================================================================================
Steps:
  1. Merge CLT history (2014-2025) with cleaned 2026 rounds 1-34 (from 04_ingest_2026.py).
  2. Append one PLACEHOLDER row per rostered player for round 35 (zero scouts, target unknown).
     Every feature in the engine is a shift(1) lag, so a round-35 row's features are built only
     from rounds <= 34. The placeholder's zeros could only ever leak into round 36+, which doesn't exist.
  3. Run the *unchanged* feature functions from 01_build_feature_store.py.
  4. Retrain the model zoo from 02_train_and_evaluate_models.py with:
        train = 2014-2024 | early-stopping/val = 2025 | predict = 2026 (R1-34 for an honest check, R35 for inference)
  5. Report a 2026 R1-34 out-of-sample check, then 15 suggestions per specialist for Round 35.

Outputs:
  * data/inference_2026_r35.parquet       all round-35 candidates with every model's prediction
  * data/oos_2026_metrics.csv             2026 R1-34 out-of-sample metrics
  * reports/04_round35_inference.html     interactive report (per-specialist squads, consensus, metrics)
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.stdout.reconfigure(encoding="utf-8")

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
SRC_DIR = CLF_DIR / "src"
DATA_DIR = CLF_DIR / "data"
REPORTS_DIR = CLF_DIR / "reports"
CLT_DATA_DIR = Path(r"C:\Users\thiag\Documents\CLT\data")

SEASON, TARGET_ROUND = 2026, 35
STATUS_MAP = {2: "Dúvida", 3: "Suspenso", 5: "Contundido", 6: "Nulo", 7: "Provável"}

SPECIALISTS = {
    "Positional_LightGBM": "pred_Positional_LightGBM",
    "Residual_XGBoost": "pred_Residual_XGBoost",
    "Decomposed_Hybrid": "pred_Decomposed_Hybrid",
    "Neural_MLP": "pred_Neural_MLP",
}
ENSEMBLE_COL = "pred_CLF_Meta_Ensemble"

# 15-player squad per specialist: a 4-3-3 + coach (12) plus one backup at GOL, LAT, ZAG.
SQUAD_QUOTA = {1: 2, 2: 3, 3: 3, 4: 3, 5: 3, 6: 1}
POS_CODE = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assemble_inputs():
    hist = pd.read_parquet(CLT_DATA_DIR / "normalized_scouts.parquet")
    s26 = pd.read_parquet(DATA_DIR / "season2026_scouts.parquet")
    roster = pd.read_parquet(DATA_DIR / "season2026_roster_r34.parquet")

    cols = hist.columns.tolist()
    s26_norm = s26[cols]

    # Round-35 placeholders: price after round 34 == market price before round 35.
    ph = pd.DataFrame({c: 0.0 for c in cols}, index=range(len(roster)))
    ph["player_id"] = roster["player_id"].values
    ph["season"] = SEASON
    ph["round"] = TARGET_ROUND
    ph["team_id"] = roster["team_id"].values
    ph["position_id"] = roster["position_id"].values
    ph["played"] = True
    ph["price"] = roster["price"].values
    ph["price_variation"] = 0.0
    ph = ph[cols].astype(hist.dtypes.to_dict())

    scouts = pd.concat([hist, s26_norm, ph], ignore_index=True)

    matches = pd.concat(
        [
            pd.read_parquet(CLT_DATA_DIR / "matches.parquet"),
            pd.read_parquet(DATA_DIR / "matches_2026.parquet").drop(columns=["is_inference_fixture"]),
        ],
        ignore_index=True,
    )
    matches["match_id"] = matches["match_id"].astype(str)

    players = pd.read_parquet(CLT_DATA_DIR / "players.parquet")
    new_ids = set(roster["player_id"]) | set(s26["player_id"])
    missing = sorted(new_ids - set(players["player_id"]))
    nick = s26.drop_duplicates("player_id", keep="last").set_index("player_id")["nickname"]
    players = pd.concat(
        [
            players,
            pd.DataFrame({
                "player_id": missing,
                "canonical_name": [nick.get(p, str(p)) for p in missing],
                "canonical_nickname": [nick.get(p, str(p)) for p in missing],
                "photo_url": None,
            }),
        ],
        ignore_index=True,
    )

    teams = pd.read_parquet(CLT_DATA_DIR / "teams.parquet")
    remo = teams[teams["team_id"] == 2140].copy()
    remo["team_id"] = 364
    teams = pd.concat([teams, remo], ignore_index=True)

    print(f"1. History {len(hist):,} rows + 2026 R1-34 {len(s26_norm):,} rows + {len(ph)} round-35 placeholders. "
          f"New 2026 players added to dictionary: {len(missing)}")
    return scouts, matches, players, teams, roster


def build_features(fs, scouts, matches, players, teams):
    t = time.perf_counter()
    active = scouts[scouts["played"]].copy().reset_index(drop=True)
    active = fs.sanitize_cumulative_bleed_anomalies(active)
    aligned = fs.align_matches_to_cartola_rounds(matches, active)
    cube, pos_def = fs.build_team_and_positional_cube(active, aligned)
    feats = fs.build_player_features(active, players, teams, cube, pos_def)
    fs.verify_zero_leakage(feats)
    print(f"2. Feature engine: {feats.shape[0]:,} rows x {feats.shape[1]} cols in {time.perf_counter() - t:.1f}s")
    return feats


def verify_round35(feats, matches):
    r35 = feats[(feats["season"] == SEASON) & (feats["round"] == TARGET_ROUND)]
    fx = matches[(matches["season"] == SEASON) & (matches["round"] == TARGET_ROUND)]
    expected = {**dict(zip(fx["home_team_id"], fx["away_team_id"])), **dict(zip(fx["away_team_id"], fx["home_team_id"]))}
    got = r35.groupby("team_id")["opp_id"].first().to_dict()
    bad = {k: (v, expected.get(k)) for k, v in got.items() if expected.get(k) != v}
    assert not bad, f"Round-35 opponent mismatch: {bad}"
    home_expected = set(fx["home_team_id"])
    assert (r35["is_home"] == r35["team_id"].isin(home_expected).astype(int)).all(), "Home flag mismatch!"

    # Lag check: a regular starter's round-35 roll3 must equal the mean of his last 3 2026 appearances.
    hist26 = feats[(feats["season"] == SEASON) & (feats["round"] < TARGET_ROUND)]
    pid = hist26["player_id"].value_counts().index[0]
    last3 = hist26[hist26["player_id"] == pid].sort_values("round")["target_score"].tail(3).mean()
    r3 = r35.loc[r35["player_id"] == pid, "player_roll3_base_norm_score"].iloc[0]
    assert np.isclose(last3, r3, atol=1e-4), f"Round-35 lag check failed: {last3} vs {r3}"
    print(f"3. [QA PASS] Round-35 opponents & home flags match the official fixtures; lag check OK (player {pid}).")


def train_and_predict(mz, feats):
    for c in mz.FEATURE_COLS:
        feats[c] = feats[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_df = feats[feats["season"] <= 2024].copy()
    val_df = feats[feats["season"] == 2025].copy()
    test_df = feats[feats["season"] == SEASON].copy()
    print(f"4. Retraining zoo: train 2014-2024 ({len(train_df):,}) | val 2025 ({len(val_df):,}) | "
          f"predict 2026 ({len(test_df):,} incl. {int((test_df['round'] == TARGET_ROUND).sum())} R35 candidates)")
    eval_df, _, _ = mz.train_all_models(train_df, val_df, test_df)
    return eval_df[eval_df["season"] == SEASON].copy()


def oos_metrics_2026(pred26):
    sub = pred26[pred26["round"] < TARGET_ROUND]
    rows = []
    for name, col in [("Naive_Roll5", "pred_Naive_Roll5"), ("Market_Price", "pred_Market_Price"),
                      ("Dual_Memory_EB", "pred_Dual_Memory_EB"), *SPECIALISTS.items(),
                      ("CLF_Meta_Ensemble", ENSEMBLE_COL)]:
        rhos = [spearmanr(g["target_score"], g[col])[0] for _, g in sub.groupby("round") if len(g) > 10]
        rows.append({
            "model": name,
            "RMSE": round(float(np.sqrt(mean_squared_error(sub["target_score"], sub[col]))), 4),
            "MAE": round(float(mean_absolute_error(sub["target_score"], sub[col])), 4),
            "Mean_Round_Spearman": round(float(np.nanmean(rhos)), 4),
        })
    out = pd.DataFrame(rows).sort_values("RMSE")
    out.to_csv(DATA_DIR / "oos_2026_metrics.csv", index=False)
    print(f"\n5. Honest out-of-sample check on 2026 R1-34 (N={len(sub):,}; models never saw 2026):")
    print(out.to_string(index=False))
    return out


def candidate_pool(pred26, roster):
    r35 = pred26[pred26["round"] == TARGET_ROUND].copy()
    r35 = r35.merge(roster[["player_id", "status_id"]], on="player_id", how="left")
    r35["status"] = r35["status_id"].map(STATUS_MAP).fillna("?")
    r35["last_played_round"] = TARGET_ROUND - r35["rounds_since_last_played"].where(r35["season_games_prior"] > 0)
    r35["position_code"] = r35["position_id"].map(POS_CODE)
    r35["name"] = r35["canonical_nickname"].fillna(r35["canonical_name"]).fillna(r35["player_id"].astype(str))
    # Require >= 5 appearances in 2026 AND active appearance in the last 3 rounds (rounds 32..34)
    # so long-benched reserves (e.g. players who last played in Round 18) don't pollute top picks!
    pool = r35[
        (r35["status_id"] != 5)
        & (r35["season_games_prior"] >= 5)
        & (r35["rounds_since_last_played"] <= 3)
    ].copy()
    print(f"\n6. Round-35 pool: {len(r35)} rostered → {len(pool)} active rotation candidates "
          f"(dropped Contundido, <5 games in 2026, or inactive for >3 rounds).")
    return r35, pool


def squad(pool, col):
    picks = []
    for pos, q in SQUAD_QUOTA.items():
        picks.append(pool[pool["position_id"] == pos].nlargest(q, col))
    return pd.concat(picks, ignore_index=True)


def print_squads(pool):
    squads = {}
    show = ["position_code", "name", "team_abbr", "opp_abbr", "is_home", "status", "pre_match_price",
            "season_games_prior", "last_played_round"]
    for name, col in {**SPECIALISTS, "CLF_Meta_Ensemble": ENSEMBLE_COL}.items():
        sq = squad(pool, col)
        squads[name] = sq
        print(f"\n--- {name}: 15 suggestions for Round 35 ---")
        tbl = sq[show + [col]].rename(columns={col: "pred_pts", "pre_match_price": "price",
                                                 "season_games_prior": "games26", "last_played_round": "last_rd"})
        tbl["pred_pts"] = tbl["pred_pts"].round(2)
        tbl["last_rd"] = tbl["last_rd"].astype("Int64")
        print(tbl.to_string(index=False))

    votes = pd.concat([s[["player_id"]].assign(model=m) for m, s in squads.items() if m in SPECIALISTS])
    consensus = votes.groupby("player_id")["model"].agg(n_votes="count", picked_by=lambda x: ", ".join(sorted(x)))
    return squads, consensus.reset_index()


def html_table(df, cols, rename):
    t = df[cols].rename(columns=rename).copy()
    head = "".join(f"<th>{c}</th>" for c in t.columns)
    body = ""
    for _, r in t.iterrows():
        cells = ""
        for c in t.columns:
            v = r[c]
            if isinstance(v, float):
                v = "" if np.isnan(v) else f"{v:.2f}"
            cells += f"<td>{v}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def build_html(squads, consensus, pool, metrics, gen_mod):
    cols = ["position_code", "name", "team_abbr", "opp_abbr", "is_home", "status", "pre_match_price",
            "season_games_prior", "last_played_round"]
    ren = {"position_code": "Pos", "name": "Player", "team_abbr": "Club", "opp_abbr": "Opp", "is_home": "Home",
           "status": "Status (pre-R34)", "pre_match_price": "Price C$", "season_games_prior": "Games 2026",
           "last_played_round": "Last Rd"}

    tabs_btn, tabs_body = "", ""
    for i, (name, sq) in enumerate(squads.items()):
        col = ENSEMBLE_COL if name == "CLF_Meta_Ensemble" else SPECIALISTS[name]
        sq = sq.merge(consensus, on="player_id", how="left")
        sq["Pred pts"] = sq[col]
        sq["Votes (of 4)"] = sq["n_votes"].fillna(0).astype(int)
        active = "active" if i == 0 else ""
        tabs_btn += f'<button class="tab-btn {active}" onclick="sw(\'t{i}\', this)">{name}</button>'
        tabs_body += (f'<div id="t{i}" class="tab-pane {active}"><div class="section-card"><h2>{name} — 15 suggestions</h2>'
                      + html_table(sq, cols + ["Pred pts", "Votes (of 4)"], ren) + "</div></div>")

    depth = ""
    for pos in [1, 2, 3, 4, 5, 6]:
        p = pool[pool["position_id"] == pos].nlargest(15, ENSEMBLE_COL).copy()
        for name, col in SPECIALISTS.items():
            p[name] = p[col]
        p["Ensemble"] = p[ENSEMBLE_COL]
        depth += f"<h3 style='color:#fff'>{POS_CODE[pos]} — top 15 by ensemble</h3>" + html_table(
            p, ["name", "team_abbr", "opp_abbr", "is_home", "status", "pre_match_price"] + list(SPECIALISTS) + ["Ensemble"], ren)

    mt = html_table(metrics, ["model", "RMSE", "MAE", "Mean_Round_Spearman"], {})

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>CLF Report 04 — 2026 Round 35 Inference</title>{gen_mod.SHARED_CSS}
<style>.tab-btn{{background:var(--card);border:1px solid var(--border);color:var(--muted);padding:10px 16px;border-radius:10px;cursor:pointer;font-weight:800;margin:0 6px 8px 0}}
.tab-btn.active{{color:#fff;border-color:var(--accent);background:var(--card-alt)}}.tab-pane{{display:none}}.tab-pane.active{{display:block}}</style></head>
<body>{gen_mod.nav_bar('r35')}<div class="container">
<div class="hero"><h1>🔮 Report 04: 2026 Round 35 — Specialist Suggestions</h1>
<p>Models retrained on 2014–2024, early-stopped on 2025, and applied to 2026. Predictions are <strong>conditional on the player taking the pitch</strong> (the models were trained only on active appearances), which is why each specialist offers 15 names to cover lineup/injury surprises.</p></div>
<div class="pill-box gold"><div class="pill-title">⚠️ Read before using</div><ul>
<li><strong>Status is one round stale.</strong> The <code>rodada-34</code> snapshot status lines up with who played in round 34, so it most likely reflects the pre-round-34 market. Only <em>Contundido</em> players were removed; check Cartola's live status before the market closes.</li>
<li><strong>Ensemble weights (0.30/0.30/0.25/0.15) are fixed by hand</strong>, not fitted on validation data.</li>
<li>Squad quota per model: 2 GOL, 3 LAT, 3 ZAG, 3 MEI, 3 ATA, 1 TEC = 15.</li></ul></div>
<div class="section-card"><h2>Honest check: 2026 rounds 1–34 (models never saw 2026)</h2>{mt}</div>
<div>{tabs_btn}</div>{tabs_body}
<div class="section-card"><h2>Positional depth charts (top 15 per position, all specialists side by side)</h2>{depth}</div>
</div><script>function sw(id,b){{document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active'));document.getElementById(id).classList.add('active');b.classList.add('active');}}</script></body></html>"""
    out = REPORTS_DIR / "04_round35_inference.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n-> Saved {out}")


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("CLF Phase 6b: 2026 Round-35 feature engine + specialist inference")
    print("=" * 72)
    fs = load_module("clf_fs", "01_build_feature_store.py")
    mz = load_module("clf_mz", "02_train_and_evaluate_models.py")
    gen = load_module("clf_gen", "03_generate_html_suite.py")

    scouts, matches, players, teams, roster = assemble_inputs()
    feats = build_features(fs, scouts, matches, players, teams)
    verify_round35(feats, matches)

    pred26 = train_and_predict(mz, feats)
    metrics = oos_metrics_2026(pred26)
    r35, pool = candidate_pool(pred26, roster)
    r35.to_parquet(DATA_DIR / "inference_2026_r35.parquet", index=False)

    squads, consensus = print_squads(pool)
    build_html(squads, consensus, pool, metrics, gen)
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
