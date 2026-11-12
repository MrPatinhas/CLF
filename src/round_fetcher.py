"""
src/round_fetcher.py — CLF Automated Round Fetcher & Data Ingestion Pipeline
=============================================================================
Provides idempotent downloading, caching, validation, and preprocessing of
scout snapshots and fixtures for any requested season and round.

Key Capabilities:
  - Idempotent caching: checks local data/raw/{season}/rodada-{r}.csv before fetching
  - Graceful errors: raises MissingScoutsError or MissingFixturesError with actionable messages
  - Training Grounds detection: warns when a requested round falls within historical train/val
  - Automatic round detection: inspects local cache to find the next unplayed round
"""

import importlib.util
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
DATA_DIR = CLF_DIR / "data"
RAW_BASE_DIR = DATA_DIR / "raw"
CLT_DATA_DIR = Path(r"C:\Users\thiag\Documents\CLT\data")

CARTOLA_BASE_URL = "https://raw.githubusercontent.com/henriquepgomide/caRtola/master/data/01_raw"
MATCHES_URL = "https://raw.githubusercontent.com/williamorim/brasileirao/master/data-raw/csv/matches.csv"

CARTOLA_ID_OVERRIDES = {2140: 364}  # Remo in 2026

SCOUTS = ["G", "A", "SG", "DP", "DE", "DS", "FT", "FD", "FF", "FS", "GC", "CV", "CA", "PP", "GS", "FC", "I"]

BASE_WEIGHTS = {
    "G": 8.0, "A": 5.0, "SG": 5.0, "DP": 7.0, "DE": 1.3, "DS": 1.0,
    "FT": 3.0, "FD": 1.2, "FF": 0.8, "FS": 0.5,
    "GC": -3.0, "CV": -3.0, "CA": -1.0, "PP": -4.0, "GS": -1.0, "FC": -0.3, "I": -0.1,
}

STATUS_MAP = {2: "Dúvida", 3: "Suspenso", 5: "Contundido", 6: "Nulo", 7: "Provável"}
POS_CODE = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}


class MissingScoutsError(FileNotFoundError):
    """Raised when required scout snapshots cannot be located locally or downloaded."""
    pass


class MissingFixturesError(ValueError):
    """Raised when fixtures for the target round are incomplete or unavailable."""
    pass


def load_clt_name_resolver():
    """Loads CLT's canonical club name resolver without running its pipeline."""
    path = Path(r"C:\Users\thiag\Documents\CLT\pipeline\01_build_teams_and_matches.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("clt_teams", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_team_id


def check_remote_round_exists(season: int, r: int) -> bool:
    """Sends a fast HTTP HEAD request (~150ms) to check if round r is published on GitHub."""
    url = f"{CARTOLA_BASE_URL}/{season}/rodada-{r}.csv"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def auto_detect_season_and_round() -> Tuple[int, int]:
    """
    Auto-detects the target season and round:
      1. Finds highest locally cached round for 2026 (e.g. 34).
      2. Sends a quick HTTP HEAD check to Henrique's GitHub repo to see if newer
         rounds (e.g. 35) have been published.
      3. Returns (season, target_round) where target_round = latest_published + 1.
    """
    season = 2026
    latest_local = 34
    raw_season = RAW_BASE_DIR / str(season)
    if raw_season.exists():
        csvs = list(raw_season.glob("rodada-*.csv"))
        if csvs:
            rounds = [int(p.stem.split("-")[1]) for p in csvs if p.stem.split("-")[1].isdigit()]
            if rounds:
                latest_local = max(rounds)

    # Probe GitHub for any newly published rounds beyond local cache
    candidate_r = latest_local + 1
    while candidate_r <= 38:
        if check_remote_round_exists(season, candidate_r):
            latest_local = candidate_r
            candidate_r += 1
        else:
            break

    target_round = min(latest_local + 1, 38)
    return season, target_round


def check_training_grounds(season: int, target_round: int) -> Dict[str, str]:
    """
    Determines whether the requested round falls in training, validation, or live grounds.
    """
    if season <= 2024:
        return {
            "status": "HISTORICAL_TRAINING",
            "is_future": False,
            "message": f"Season {season} Round {target_round} is within HISTORICAL TRAINING GROUNDS (2014-2024).",
        }
    elif season == 2025:
        return {
            "status": "HISTORICAL_VAL",
            "is_future": False,
            "message": f"Season {season} Round {target_round} is within VALIDATION GROUNDS (2025).",
        }
    elif season == 2026 and target_round <= 34:
        return {
            "status": "HISTORICAL_PLAYED",
            "is_future": False,
            "message": f"Season {season} Round {target_round} was ALREADY PLAYED in live 2026. Running backtest audit.",
        }
    else:
        return {
            "status": "LIVE_FUTURE",
            "is_future": True,
            "message": f"Season {season} Round {target_round} is an UNPLAYED FUTURE ROUND. Running forward inference.",
        }


def download_round_snapshot_if_missing(season: int, r: int) -> Path:
    """
    Idempotently returns local path to rodada-{r}.csv, downloading from caRtola if missing.
    """
    season_raw_dir = RAW_BASE_DIR / str(season)
    season_raw_dir.mkdir(parents=True, exist_ok=True)
    local_file = season_raw_dir / f"rodada-{r}.csv"

    if local_file.exists():
        return local_file

    url = f"{CARTOLA_BASE_URL}/{season}/rodada-{r}.csv"
    try:
        urllib.request.urlretrieve(url, local_file)
        print(f"   [RoundFetcher] Downloaded: {url} -> {local_file.name}")
        return local_file
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise MissingScoutsError(
                f"\n[MissingScoutsError] Round {r} scouts for Season {season} are not available on GitHub (404 Not Found).\n"
                f"URL attempted: {url}\n"
                f"To predict round {r + 1}, scouts through round {r} must be published first."
            ) from e
        raise e
    except Exception as e:
        raise MissingScoutsError(f"Failed to download {url}: {e}") from e


def fetch_scouts_up_to_round(season: int, target_round: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensures all round snapshots 1..(target_round - 1) are present locally and cleaned.
    Returns (cleaned_scouts_df, latest_roster_snapshot).
    """
    last_played = target_round - 1
    if last_played < 1:
        raise ValueError(f"Target round must be >= 2 to compute pre-match features (got {target_round}).")

    # If season is <= 2025, we can use the pristine historical archive in CLT
    if season <= 2025:
        hist_scouts = pd.read_parquet(CLT_DATA_DIR / "normalized_scouts.parquet")
        sub = hist_scouts[(hist_scouts["season"] == season) & (hist_scouts["round"] <= last_played)].copy()
        roster_sub = sub[sub["round"] == last_played].copy()
        roster = pd.DataFrame({
            "player_id": roster_sub["player_id"],
            "team_id": roster_sub["team_id"],
            "position_id": roster_sub["position_id"],
            "price": roster_sub["price"],
            "status_id": 7,  # Default to Provável for historical roster
            "nickname": roster_sub["player_id"].astype(str),
        })
        return sub, roster

    # Live season (e.g. 2026): ensure each round 1..last_played is cached locally
    frames = []
    print(f"[RoundFetcher] Validating local cache for Season {season} Rounds 1..{last_played}...")
    for r in range(1, last_played + 1):
        p = download_round_snapshot_if_missing(season, r)
        df_r = pd.read_csv(p, low_memory=False)
        df_r["round"] = r
        frames.append(df_r)

    raw = pd.concat(frames, ignore_index=True)

    # Clean and de-accumulate scouts
    df = pd.DataFrame({
        "player_id": raw["atletas.atleta_id"].astype(int),
        "season": season,
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

    # De-accumulate cumulative scouts
    df = df.sort_values(["player_id", "round"]).reset_index(drop=True)
    deltas = df.groupby("player_id")[SCOUTS].diff().fillna(df[SCOUTS]).clip(lower=0.0)
    df[SCOUTS] = deltas
    df.loc[~df["played"], SCOUTS] = 0.0

    scout_score = sum(df[s] * w for s, w in BASE_WEIGHTS.items())
    df["base_norm_score"] = np.where(
        ~df["played"], 0.0, np.where(df["position_id"] == 6, df["points"], scout_score)
    ).round(2)

    # Roster snapshot at round R-1
    roster_raw = df[df["round"] == last_played]
    roster = roster_raw[["player_id", "team_id", "position_id", "price", "status_id", "nickname"]].copy()

    return df, roster


def load_fixtures_for_season(season: int, target_round: int) -> pd.DataFrame:
    """
    Loads match fixtures for the requested season, ensuring target_round contains
    exactly 10 matches across 20 distinct clubs.
    """
    if season <= 2025:
        matches = pd.read_parquet(CLT_DATA_DIR / "matches.parquet")
        sub_m = matches[matches["season"] == season].copy()
        r_matches = sub_m[sub_m["round"] == target_round]
        if len(r_matches) != 10:
            raise MissingFixturesError(
                f"Historical season {season} round {target_round} has {len(r_matches)} matches (expected 10)."
            )
        return sub_m

    # For 2026: check local cached matches_2026.parquet
    cached_2026 = DATA_DIR / "matches_2026.parquet"
    if cached_2026.exists():
        m_2026 = pd.read_parquet(cached_2026)
        r_matches = m_2026[m_2026["round"] == target_round]
        teams_r = pd.concat([r_matches["home_team_id"], r_matches["away_team_id"]])
        if len(r_matches) == 10 and teams_r.nunique() == 20:
            return m_2026

    # Otherwise fetch live schedule
    resolver = load_clt_name_resolver()
    print(f"[RoundFetcher] Fetching official fixture calendar from {MATCHES_URL}...")
    m = pd.read_csv(MATCHES_URL)
    s = m[m["season"] == season].sort_values("id_match").reset_index(drop=True)
    if len(s) == 0:
        raise MissingFixturesError(f"No fixtures found for season {season} in {MATCHES_URL}")

    s["round"] = (np.arange(len(s)) // 10) + 1
    s["is_played"] = s["score"].astype(str).str.contains(r"\d", regex=True)

    def parse_score(score):
        try:
            h, a = str(score).lower().split("x")
            return int(h), int(a)
        except Exception:
            return np.nan, np.nan

    sc = s["score"].apply(parse_score)
    s["home_score"] = [x[0] for x in sc]
    s["away_score"] = [x[1] for x in sc]

    def resolve(name):
        tid = resolver(name) if resolver else None
        return CARTOLA_ID_OVERRIDES.get(tid, tid)

    s["home_team_id"] = s["home"].apply(resolve)
    s["away_team_id"] = s["away"].apply(resolve)

    r_target = s[s["round"] == target_round]
    teams_target = pd.concat([r_target["home_team_id"], r_target["away_team_id"]])
    if len(r_target) != 10 or teams_target.nunique() != 20:
        raise MissingFixturesError(
            f"Season {season} Round {target_round} does not have a complete 10-match fixture set! "
            f"Found {len(r_target)} matches and {teams_target.nunique()} unique teams."
        )

    keep = s[s["is_played"] | (s["round"] == target_round)].copy()
    is_target = keep["round"] == target_round
    keep.loc[is_target, ["home_score", "away_score"]] = 0
    keep["is_inference_fixture"] = is_target

    out = keep.rename(columns={"id_match": "match_id", "home": "home_team", "away": "away_team"})[
        ["match_id", "season", "round", "date", "home_team_id", "away_team_id",
         "home_team", "away_team", "home_score", "away_score", "is_inference_fixture"]
    ]
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    out.to_parquet(DATA_DIR / f"matches_{season}.parquet", index=False)
    return out


def prepare_inference_feature_dataset(
    season: int,
    target_round: int,
    feature_store_module,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines historical scouts + season scouts up to (target_round - 1) + placeholder
    rows for target_round, executes the feature pipeline, and returns the target round features.
    """
    t0 = time.perf_counter()
    hist_scouts = pd.read_parquet(CLT_DATA_DIR / "normalized_scouts.parquet")
    cols = hist_scouts.columns.tolist()

    season_scouts, roster = fetch_scouts_up_to_round(season, target_round)

    if season <= 2025:
        # Target round is inside CLT historical bounds
        prior_hist = hist_scouts[
            (hist_scouts["season"] < season)
            | ((hist_scouts["season"] == season) & (hist_scouts["round"] < target_round))
        ].copy()
    else:
        prior_hist = hist_scouts.copy()

    # Create target round placeholders from roster snapshot
    ph = pd.DataFrame({c: 0.0 for c in cols}, index=range(len(roster)))
    ph["player_id"] = roster["player_id"].values
    ph["season"] = season
    ph["round"] = target_round
    ph["team_id"] = roster["team_id"].values
    ph["position_id"] = roster["position_id"].values
    ph["played"] = True
    ph["price"] = roster["price"].values
    ph["price_variation"] = 0.0
    ph = ph[cols].astype(hist_scouts.dtypes.to_dict())

    if season <= 2025:
        all_scouts = pd.concat([prior_hist, ph], ignore_index=True)
    else:
        all_scouts = pd.concat([prior_hist, season_scouts[cols], ph], ignore_index=True)

    matches_season = load_fixtures_for_season(season, target_round)
    hist_matches = pd.read_parquet(CLT_DATA_DIR / "matches.parquet")

    if season <= 2025:
        all_matches = hist_matches[
            (hist_matches["season"] < season)
            | ((hist_matches["season"] == season) & (hist_matches["round"] <= target_round))
        ].copy()
    else:
        all_matches = pd.concat(
            [hist_matches, matches_season.drop(columns=["is_inference_fixture"], errors="ignore")],
            ignore_index=True,
        )
    all_matches["match_id"] = all_matches["match_id"].astype(str)

    # Resolve players and teams dictionaries
    players = pd.read_parquet(CLT_DATA_DIR / "players.parquet")
    new_ids = set(roster["player_id"]) | set(season_scouts["player_id"])
    missing = sorted(new_ids - set(players["player_id"]))
    if missing:
        nick = season_scouts.drop_duplicates("player_id", keep="last").set_index("player_id")["nickname"]
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

    # Run feature engine
    active = all_scouts[all_scouts["played"]].copy().reset_index(drop=True)
    active = feature_store_module.sanitize_cumulative_bleed_anomalies(active)
    aligned = feature_store_module.align_matches_to_cartola_rounds(all_matches, active)
    cube, pos_def = feature_store_module.build_team_and_positional_cube(active, aligned)
    feats = feature_store_module.build_player_features(active, players, teams, cube, pos_def)
    feature_store_module.verify_zero_leakage(feats)

    target_feats = feats[(feats["season"] == season) & (feats["round"] == target_round)].copy()
    print(f"[RoundFetcher] Built pre-match features for {len(target_feats)} candidates in {time.perf_counter() - t0:.1f}s.")
    return target_feats, roster
