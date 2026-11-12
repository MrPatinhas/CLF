"""
04_ingest_2026.py — CLF Phase 6a: Raw acquisition + cleaning of the live 2026 season
=====================================================================================
Sources:
  * henriquepgomide/caRtola  data/01_raw/2026/rodada-{1..34}.csv   (cumulative scouts)
  * williamorim/brasileirao  data-raw/csv/matches.csv               (2026 fixtures)

Outputs (all inside CLF/data — CLT data is never modified):
  * raw/2026/rodada-N.csv                 cached raw snapshots
  * season2026_scouts.parquet             per-round de-accumulated scouts + base_norm_score
  * season2026_roster_r34.parquet         latest roster snapshot (status, price) = candidate pool
  * matches_2026.parquet                  played 2026 fixtures + round-35 fixtures (placeholder 0-0)

Learnable pill — why the round-35 fixtures come from `id_match` order, not dates:
  The schedule source keeps official round order in `id_match`, while dates move when
  games get rescheduled (e.g. Mirassol x Atlético-MG moved to Nov 14). Chunking the
  season's id order into blocks of 10 gives the official round; we assert that the
  round-35 block covers all 20 clubs exactly once.
"""

import importlib.util
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
DATA_DIR = CLF_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "2026"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASON = 2026
LAST_ROUND = 34
TARGET_ROUND = 35
BASE_RAW = "https://raw.githubusercontent.com/henriquepgomide/caRtola/master/data/01_raw/2026"
MATCHES_URL = "https://raw.githubusercontent.com/williamorim/brasileirao/master/data-raw/csv/matches.csv"

# Same 17 universal scouts + weights used by CLT's base_norm_score (app/engine/data_prep.py)
BASE_WEIGHTS = {
    "G": 8.0, "A": 5.0, "SG": 5.0, "DP": 7.0, "DE": 1.3, "DS": 1.0,
    "FT": 3.0, "FD": 1.2, "FF": 0.8, "FS": 0.5,
    "GC": -3.0, "CV": -3.0, "CA": -1.0, "GS": -1.0, "FC": -0.3,
    "I": -0.1, "PP": -4.0,
}
SCOUTS = list(BASE_WEIGHTS.keys())

# Cartola uses club id 364 for Remo in 2026 (CLT canonical table has 2140).
CARTOLA_ID_OVERRIDES = {2140: 364}


def load_clt_name_lookup():
    """Reuse CLT's club-name resolver without executing its pipeline."""
    path = Path(r"C:\Users\thiag\Documents\CLT\pipeline\01_build_teams_and_matches.py")
    spec = importlib.util.spec_from_file_location("clt_teams", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_team_id


def fetch_round_csvs() -> pd.DataFrame:
    frames = []
    for r in range(1, LAST_ROUND + 1):
        local = RAW_DIR / f"rodada-{r}.csv"
        if not local.exists():
            urllib.request.urlretrieve(f"{BASE_RAW}/rodada-{r}.csv", local)
        df = pd.read_csv(local, low_memory=False)
        df["round"] = r
        frames.append(df)
    print(f"1. Loaded {len(frames)} round snapshots (rodada-1..{LAST_ROUND}).")
    return pd.concat(frames, ignore_index=True)


def clean_scouts(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "player_id": raw["atletas.atleta_id"].astype(int),
        "season": SEASON,
        "round": raw["round"].astype(int),
        "team_id": raw["atletas.clube_id"].astype(int),
        "position_id": raw["atletas.posicao_id"].astype(int),
        "played": raw["atletas.entrou_em_campo"].fillna(False).astype(bool),
        "points": pd.to_numeric(raw["atletas.pontos_num"], errors="coerce").fillna(0.0),
        "price": pd.to_numeric(raw["atletas.preco_num"], errors="coerce").fillna(0.0),
        "price_variation": pd.to_numeric(raw["atletas.variacao_num"], errors="coerce").fillna(0.0),
        "status_id": raw["atletas.status_id"],
        "nickname": raw["atletas.apelido"],
    })
    for s in SCOUTS:
        df[s] = pd.to_numeric(raw[s], errors="coerce").fillna(0.0) if s in raw.columns else 0.0

    # De-accumulate: 2026 snapshots are cumulative season-to-date (verified monotone).
    df = df.sort_values(["player_id", "round"]).reset_index(drop=True)
    deltas = df.groupby("player_id")[SCOUTS].diff().fillna(df[SCOUTS]).clip(lower=0.0)
    df[SCOUTS] = deltas

    # Non-playing rows must carry zero scouts (a de-accumulation of a non-appearance is 0 anyway).
    df.loc[~df["played"], SCOUTS] = 0.0

    scout_score = sum(df[s] * w for s, w in BASE_WEIGHTS.items())
    df["base_norm_score"] = np.where(
        ~df["played"], 0.0, np.where(df["position_id"] == 6, df["points"], scout_score)
    ).round(2)

    gap = (df["base_norm_score"] - df["points"]).abs()
    print(
        f"2. Cleaned {len(df):,} rows ({df['played'].sum():,} active). "
        f"Mean |base_norm_score - points| on active outfield rows = "
        f"{gap[df['played'] & (df['position_id'] != 6)].mean():.3f} "
        f"(rows with gap > 6.5 → sanitized later: {(gap[df['played']] > 6.5).sum()})"
    )
    return df


def build_matches_2026(get_team_id) -> pd.DataFrame:
    m = pd.read_csv(MATCHES_URL)
    s = m[m["season"] == SEASON].sort_values("id_match").reset_index(drop=True)
    s["round"] = (np.arange(len(s)) // 10) + 1
    s["is_played"] = s["score"].astype(str).str.contains(r"\d", regex=True)

    def parse(score):
        try:
            h, a = str(score).lower().split("x")
            return int(h), int(a)
        except Exception:
            return np.nan, np.nan

    sc = s["score"].apply(parse)
    s["home_score"] = [x[0] for x in sc]
    s["away_score"] = [x[1] for x in sc]

    def resolve(name):
        tid = get_team_id(name)
        return CARTOLA_ID_OVERRIDES.get(tid, tid)

    s["home_team_id"] = s["home"].apply(resolve)
    s["away_team_id"] = s["away"].apply(resolve)
    assert s[["home_team_id", "away_team_id"]].notna().all().all(), "Unresolved 2026 club names!"

    r35 = s[s["round"] == TARGET_ROUND]
    teams_r35 = pd.concat([r35["home_team_id"], r35["away_team_id"]])
    assert len(r35) == 10 and teams_r35.nunique() == 20, "Round-35 block is not a valid 10-game round!"

    keep = s[s["is_played"] | (s["round"] == TARGET_ROUND)].copy()
    is_r35 = keep["round"] == TARGET_ROUND
    # Placeholder 0-0 for round 35 so fixture alignment has numeric values. Never used as a
    # feature for round 35 itself (all team features are shift(1) lags).
    keep.loc[is_r35, ["home_score", "away_score"]] = 0
    keep["is_inference_fixture"] = is_r35

    out = keep.rename(columns={"id_match": "match_id", "home": "home_team", "away": "away_team"})[
        ["match_id", "season", "round", "date", "home_team_id", "away_team_id",
         "home_team", "away_team", "home_score", "away_score", "is_inference_fixture"]
    ]
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    print(f"3. 2026 fixtures: {int(s['is_played'].sum())} played + 10 round-35 fixtures:")
    for _, row in r35.iterrows():
        print(f"     R35  {row['home']:>15} x {row['away']:<15} ({row['date']})")
    return out


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("CLF Phase 6a: Ingest & clean 2026 (rounds 1-34) + round-35 fixtures")
    print("=" * 72)
    raw = fetch_round_csvs()
    scouts = clean_scouts(raw)
    scouts.to_parquet(DATA_DIR / "season2026_scouts.parquet", index=False)

    roster = scouts[scouts["round"] == LAST_ROUND][
        ["player_id", "team_id", "position_id", "price", "status_id", "nickname"]
    ].copy()
    roster.to_parquet(DATA_DIR / "season2026_roster_r34.parquet", index=False)
    print(f"   Roster snapshot (rodada-{LAST_ROUND}): {len(roster)} players across {roster['team_id'].nunique()} clubs.")

    matches = build_matches_2026(load_clt_name_lookup())
    matches.to_parquet(DATA_DIR / "matches_2026.parquet", index=False)
    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
