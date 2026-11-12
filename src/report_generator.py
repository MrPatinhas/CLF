"""
report_generator.py — CLF Editorial HTML Inference Suite & Decision Intelligence
================================================================================
Generates a standalone, zero-dependency editorial cream HTML dashboard with:
  * Pure Plus Jakarta Sans typography throughout (matching Ranks, no distorted Syne fonts).
  * Prominent, unmissable universal itemized receipts with [🧾 Breakdown] buttons.
  * Plotly Risk/Reward Spectrum with base64 Club Badges directly on top 30 stars + click-to-receipt.
  * Quick Receipts Bar for instantaneous one-click explainability of top Serie A stars.
  * Header branding with CLT logo and exact generation timestamp.
  * Two-tier nested navigation (Strategic, ML Black-Box, ML Open-Box, Depth).
  * 4-3-3 Tactical Pitch View with dotted departmental totals (Attack, Midfield, Defense).
  * 5 Vertical Luxury Advisory Bench columns beneath the pitch.
  * Model Academy cards (📖 Book icon) with mechanics, backtest stats, pros & cons.
  * Dynamic Captaincy Suggestion Engine with 3-way toggle (Mean, Ceiling, Spread).
  * Player photos (Globo CDN) and local club badges throughout.
"""

import base64
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px

CLF_DIR = Path(r"C:\Users\thiag\Documents\CLF")
REPORTS_DIR = CLF_DIR / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"
CLUBS_DIR = ASSETS_DIR / "clubs"

ENSEMBLE_COL = "pred_CLF_Meta_Ensemble"
SPECIALISTS = {
    "Positional_LightGBM": "pred_Positional_LightGBM",
    "Residual_XGBoost": "pred_Residual_XGBoost",
    "Decomposed_Hybrid": "pred_Decomposed_Hybrid",
    "Neural_MLP": "pred_Neural_MLP",
    "Generative_Poisson": "pred_Generative_Poisson",
}

POS_NAME = {1: "Gol", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}
POS_CODE = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}

MODEL_ACADEMY_INFO = {
    "CLF_Meta_Ensemble": {
        "title": "CLF Meta Ensemble (Master Stack)",
        "icon": "⭐",
        "arch": "Convex blend: 30% Positional LightGBM + 30% Residual XGBoost + 25% Decomposed Hybrid + 15% Neural MLP. Blends tree boosting, residual target learning, Poisson decomposition, and deep neural latent representations.",
        "stats": {"Test MAE": "2.18 pts", "Spearman ρ": "0.442", "Top-15% PR-AUC": "0.334", "2024–2025 Pts": "2,681.7 (2024) / 3,034.9 (2025)"},
        "pros": "Lowest variance across all 38 rounds; cancels out single-model inductive biases; highest consensus reliability.",
        "cons": "Computationally requires all 4 underlying models to be trained and evaluated.",
    },
    "Positional_LightGBM": {
        "title": "Positional LightGBM",
        "icon": "🌲",
        "arch": "6 distinct LightGBM gradient boosted trees trained independently per position on 152 relational features (EWMA form, opponent-defense conceded rates, team match cube).",
        "stats": {"Test MAE": "2.24 pts", "Spearman ρ": "0.428", "Top-15% PR-AUC": "0.312", "2024–2025 Pts": "2,564.1 (2024) / 2,890.3 (2025)"},
        "pros": "Excels at capturing position-specific opponent vulnerability and non-linear matchup interactions.",
        "cons": "L2 leaf shrinkage shrinks elite stars toward the positional mean (-0.62 pts bias on top 8% players).",
    },
    "Residual_XGBoost": {
        "title": "Residual XGBoost (Elite Star De-Meaner)",
        "icon": "⚡",
        "arch": "Trains on the de-meaned player residual: Δy = y - μ_dual_EB, then re-adds the Empirical Bayes baseline at inference. Eliminates elite star shrinkage.",
        "stats": {"Test MAE": "2.21 pts", "Spearman ρ": "0.435", "Top-15% PR-AUC": "0.326", "2024–2025 Pts": "2,610.8 (2024) / 2,945.6 (2025)"},
        "pros": "Reduces elite star prediction bias by 86% (solves the 'Arrascaeta Problem'); accurately values high-floor superstars.",
        "cons": "Slightly more sensitive to sudden tactical role changes or mid-season tactical demotions.",
    },
    "Decomposed_Hybrid": {
        "title": "Decomposed Hybrid (Poisson + SG Classifier)",
        "icon": "🛡️",
        "arch": "Decomposes target into a Bernoulli Clean Sheet classifier (SG in {0, 5}) + Zero-Inflated Poisson regressor for active match scouts.",
        "stats": {"Test MAE": "2.29 pts", "Spearman ρ": "0.419", "Top-15% PR-AUC": "0.298", "2024–2025 Pts": "2,510.4 (2024) / 2,820.7 (2025)"},
        "pros": "Preserves the bimodal discrete +5.0 Clean Sheet jump for defenders without smearing probability across fractions.",
        "cons": "Can slightly under-predict volatile attacking shootouts where defense collapses early.",
    },
    "Neural_MLP": {
        "title": "Neural MLP (Deep Entity Embeddings)",
        "icon": "🧠",
        "arch": "Multi-Layer Perceptron with learned entity embeddings for Club, Opponent, and Position, trained with Cosine Annealing and Dropout.",
        "stats": {"Test MAE": "2.38 pts", "Spearman ρ": "0.398", "Top-15% PR-AUC": "0.284", "2024–2025 Pts": "2,440.2 (2024) / 2,750.1 (2025)"},
        "pros": "Discovers latent non-linear cross-feature embeddings and club style interactions inaccessible to tree models.",
        "cons": "Higher epistemic uncertainty on low-sample fringe rotation players.",
    },
    "Generative_Poisson": {
        "title": "Generative Scout Engine (Poisson + Negative Binomial)",
        "icon": "🎲",
        "arch": "9 individual Poisson deviance GLMs for scouts (G, A, DS, FD, FF, FS, DE, FC, CA) + Negative Binomial save dispersion + 10,000-draw Monte Carlo simulator.",
        "stats": {"Test MAE": "2.31 pts", "Spearman ρ": "0.412", "Top-15% PR-AUC": "0.305", "2024–2025 Pts": "2,530.1 (2024) / 2,860.5 (2025)"},
        "pros": "100% explainable grocery receipt for every single point; unlocks true Floor (P10), Ceiling (P90), and Mito probability.",
        "cons": "Assumes conditional independence of scouts given expected arrival rates λ.",
    },
}


def load_club_base64_dict() -> Dict[str, str]:
    """Loads all local club badges into base64 data URIs for embedding in Plotly and HTML."""
    b64_dict = {}
    if CLUBS_DIR.exists():
        for f in CLUBS_DIR.glob("*.png"):
            b64_dict[f.stem] = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode("ascii")
    return b64_dict


def load_logo_base64() -> str:
    """Loads the CLT logo into a base64 data URI."""
    logo_file = ASSETS_DIR / "logo.png"
    if logo_file.exists():
        return "data:image/png;base64," + base64.b64encode(logo_file.read_bytes()).decode("ascii")
    return ""


CLUB_B64_CACHE = load_club_base64_dict()
LOGO_B64_CACHE = load_logo_base64()


def build_receipts_payload(pool_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Builds a normalized, client-side relational state dictionary for all candidates."""
    receipts = {}
    scouts_meta = [
        ("G", "Gol", 8.0),
        ("A", "Assistência", 5.0),
        ("DS", "Desarme", 1.0),
        ("FD", "Finalização Defendida", 1.2),
        ("FF", "Finalização Fora", 0.8),
        ("FS", "Falta Sofrida", 0.5),
        ("DE", "Defesa Difícil", 1.3),
        ("FC", "Falta Cometida", -0.3),
        ("CA", "Cartão Amarelo", -1.0),
    ]

    for _, r in pool_df.iterrows():
        pid = int(r["player_id"])
        photo = str(r.get("photo_url", "")).strip()
        if photo == "nan" or not photo:
            photo = ""
        team_ab = str(r.get("team_abbr", r.get("team_id", ""))).strip()

        scouts_info = []
        for s_code, s_name, s_w in scouts_meta:
            lam = float(r.get(f"gen_lam_{s_code}", 0.0))
            pts = lam * s_w
            scouts_info.append({
                "code": s_code,
                "name": s_name,
                "rate": round(lam, 2),
                "weight": s_w,
                "points": round(pts, 2),
            })

        p_sg = float(r.get("gen_lam_SG", 0.0))
        sg_pts = round(p_sg * 5.0, 2)

        receipts[pid] = {
            "id": pid,
            "name": str(r["name"]),
            "club": team_ab,
            "opp": str(r.get("opp_abbr", "")),
            "pos": str(r.get("position_code", "")),
            "pos_id": int(r.get("position_id", 0)),
            "is_home": int(r.get("is_home", 1)),
            "price": round(float(r.get("pre_match_price", 0.0)), 2),
            "photo": photo,
            "ens_score": round(float(r.get(ENSEMBLE_COL, 0.0)), 2),
            "gen_mean": round(float(r.get("pred_Generative_Poisson", 0.0)), 2),
            "floor_p10": round(float(r.get("gen_floor_p10", 0.0)), 2),
            "ceil_p90": round(float(r.get("gen_ceiling_p90", 0.0)), 2),
            "upside_spread": round(float(r.get("upside_spread", 0.0)), 2),
            "vol_mult": round(float(r.get("vol_multiplier", 1.0)), 2),
            "prob_mito": round(float(r.get("gen_prob_mito", 0.0)), 3),
            "p_sg": round(p_sg, 3),
            "sg_pts": sg_pts,
            "scouts": scouts_info,
            "total_expected": round(float(r.get("pred_Generative_Poisson", 0.0)), 2),
        }

    return receipts


def render_compact_table(df_subset: pd.DataFrame, cols_show: list, col_names: dict, show_bench_divider: bool = True) -> str:
    """Renders a compact, editorial cream table with player photos, club badges, and unmissable receipt triggers."""
    # Ensure receipt column is included right after player name
    cols_order = list(cols_show)
    if "receipt" not in cols_order:
        if "name" in cols_order:
            idx = cols_order.index("name") + 1
            cols_order.insert(idx, "receipt")
        else:
            cols_order.append("receipt")

    names_map = {**col_names, "receipt": "Breakdown"}
    available_cols = [c for c in cols_order if c in df_subset.columns or c == "receipt"]
    th = "".join(f"<th>{names_map.get(c, c)}</th>" for c in available_cols)
    rows = []
    in_bench = False

    for _, r in df_subset.iterrows():
        slot_raw = str(r.get("lineup_slot", ""))
        is_reserve = slot_raw.startswith("RES") or slot_raw == "RESERVE"

        if is_reserve and not in_bench and show_bench_divider:
            in_bench = True
            n_cols = len(available_cols)
            rows.append(f"""
            <tr class="bench-divider-row">
              <td colspan="{n_cols}">
                <div class="bench-divider-bar">
                  <span class="bench-badge">💎 LUXURY ADVISORY BENCH (9 BEST-IN-SLOT UPGRADES)</span>
                  <span class="bench-subtext">1 GOL • 2 LAT • 2 ZAG • 2 MEI • 2 ATA • Unconstrained Points Expectation (Advisory Frontier for Bankrolls &gt; C$ 135)</span>
                </div>
              </td>
            </tr>
            """)

        tds = []
        pid = int(r.get("player_id", 0))
        photo = str(r.get("photo_url", "")).strip()
        t_abbr = str(r.get("team_abbr", "")).strip()
        p_name = str(r.get("name", ""))

        for c in available_cols:
            if c == "receipt":
                tds.append(f'<td class="cell-receipt"><button class="btn-receipt-pill" onclick="openReceipt({pid})" title="View mathematical receipt for {p_name}">🧾 Breakdown</button></td>')
                continue

            val = r[c]
            if c == "lineup_slot":
                if val == "STARTER":
                    tds.append('<td><span class="pill-slot starter">STARTER</span></td>')
                elif val == "COACH":
                    tds.append('<td><span class="pill-slot coach">COACH</span></td>')
                else:
                    tds.append(f'<td><span class="pill-slot bench">{val}</span></td>')
            elif c == "role_class":
                if val == "SOLID":
                    tds.append('<td><span class="pill-role solid">SOLID</span></td>')
                elif val == "WILDCARD":
                    tds.append('<td><span class="pill-role wildcard">WILDCARD</span></td>')
                else:
                    tds.append(f'<td><span class="pill-role coach">{val}</span></td>')
            elif c == "position_code":
                pos_cls = f"pos-{str(val).lower()}"
                tds.append(f'<td><span class="pill-pos {pos_cls}">{val}</span></td>')
            elif c == "name":
                fallback_svg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='26' height='26' viewBox='0 0 26 26'%3E%3Ccircle cx='13' cy='13' r='13' fill='%23e2e8f0'/%3E%3Ctext x='13' y='17' font-size='11' text-anchor='middle' fill='%2364748b'%3E👤%3C/text%3E%3C/svg%3E"
                img_tag = f'<img src="{photo}" class="player-thumb" onerror="this.src=\'{fallback_svg}\'">' if photo and photo != "nan" else f'<img src="{fallback_svg}" class="player-thumb">'
                badge_src = CLUB_B64_CACHE.get(t_abbr, f"assets/clubs/{t_abbr}.png")
                badge_tag = f'<img src="{badge_src}" class="club-thumb" onerror="this.style.display=\'none\'">'
                tds.append(f'<td class="cell-player"><div class="player-flex">{img_tag}{badge_tag}<span class="player-name-link" onclick="openReceipt({pid})" title="Click to view mathematical receipt">{p_name}</span></div></td>')
            elif c == "pre_match_price":
                tds.append(f'<td class="mono font-price">C$ {float(val):.1f}</td>')
            elif c == "gen_prob_mito":
                tds.append(f'<td class="mono font-pct">{float(val)*100:.0f}%</td>')
            elif c == "is_home":
                loc_text = "Home" if val == 1 or val == "Home" else "Away"
                tds.append(f'<td class="text-dim">{loc_text}</td>')
            elif c in [ENSEMBLE_COL, "pred_Generative_Poisson", "gen_ceiling_p90", "upside_spread"]:
                tds.append(f'<td class="mono font-stat"><strong>{float(val):.2f}</strong></td>')
            elif isinstance(val, (float, np.floating)):
                tds.append(f'<td class="mono">{val:.2f}</td>')
            else:
                tds.append(f'<td>{val}</td>')

        rows.append(f"<tr>{''.join(tds)}</tr>")

    return f'<div class="tbl-wrap"><table class="data-table"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render_academy_card(model_key: str) -> str:
    """Renders the collapsible Model Academy drawer with 📖 book icon."""
    info = MODEL_ACADEMY_INFO.get(model_key)
    if not info:
        return ""

    rows_stat = "".join(f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in info["stats"].items())
    uid = f"acad-{model_key}"

    return f"""
    <div class="academy-card">
      <div class="academy-toggle" onclick="toggleAcademy('{uid}')">
        <div class="academy-toggle-left">
          <span class="academy-icon">📖</span>
          <div>
            <div class="academy-title">Model Academy: {info['title']}</div>
            <div class="academy-sub">Architecture, Mathematical Mechanics &amp; Out-of-Sample Performance</div>
          </div>
        </div>
        <span class="academy-arrow" id="arrow-{uid}">▼</span>
      </div>
      <div id="content-{uid}" class="academy-drawer" style="display:none;">
        <div class="academy-grid">
          <div class="academy-col">
            <h5>🧠 Architecture &amp; Mechanism</h5>
            <p>{info['arch']}</p>
          </div>
          <div class="academy-col">
            <h5>📈 2024–2025 Test Set Metrics</h5>
            <table class="academy-table">{rows_stat}</table>
          </div>
          <div class="academy-col">
            <h5>⚖️ Operational Trade-offs</h5>
            <p><strong style="color:#059669;">Pros:</strong> {info['pros']}</p>
            <p><strong style="color:#b45309; margin-top:6px; display:inline-block;">Cons:</strong> {info['cons']}</p>
          </div>
        </div>
      </div>
    </div>
    """


def render_tactical_pitch(squad_df: pd.DataFrame, s_key: str) -> str:
    """Renders the 4-3-3 tactical soccer pitch and 5 vertical luxury bench columns with unmissable breakdown triggers."""
    starters = squad_df[squad_df["lineup_slot"] == "STARTER"].copy()
    coach = squad_df[squad_df["lineup_slot"] == "COACH"].copy()
    bench = squad_df[squad_df["lineup_slot"] != "STARTER"]
    bench = bench[bench["lineup_slot"] != "COACH"].copy()

    # Department subtotals
    def_unit = starters[starters["position_id"].isin([1, 2, 3])]
    mid_unit = starters[starters["position_id"] == 4]
    att_unit = starters[starters["position_id"] == 5]

    def_pts = float(def_unit[ENSEMBLE_COL].sum()) if len(def_unit) > 0 else 0.0
    def_cost = float(def_unit["pre_match_price"].sum()) if len(def_unit) > 0 else 0.0

    mid_pts = float(mid_unit[ENSEMBLE_COL].sum()) if len(mid_unit) > 0 else 0.0
    mid_cost = float(mid_unit["pre_match_price"].sum()) if len(mid_unit) > 0 else 0.0

    att_pts = float(att_unit[ENSEMBLE_COL].sum()) if len(att_unit) > 0 else 0.0
    att_cost = float(att_unit["pre_match_price"].sum()) if len(att_unit) > 0 else 0.0

    coach_pts = float(coach[ENSEMBLE_COL].sum()) if len(coach) > 0 else 0.0
    coach_cost = float(coach["pre_match_price"].sum()) if len(coach) > 0 else 0.0

    def _render_pitch_card(r: pd.Series) -> str:
        pid = int(r["player_id"])
        photo = str(r.get("photo_url", "")).strip()
        fallback_svg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='38' height='38' viewBox='0 0 38 38'%3E%3Ccircle cx='19' cy='19' r='19' fill='%23e2e8f0'/%3E%3Ctext x='19' y='25' font-size='16' text-anchor='middle' fill='%2364748b'%3E👤%3C/text%3E%3C/svg%3E"
        img_src = photo if (photo and photo != "nan") else fallback_svg
        t_ab = str(r.get("team_abbr", "")).strip()
        pts = float(r.get(ENSEMBLE_COL, 0.0))
        pr = float(r.get("pre_match_price", 0.0))
        name = str(r["name"])
        pos_id = int(r.get("position_id", 0))
        role = str(r.get("role_class", "SOLID"))

        role_badge = f'<span class="pitch-role-pill {role.lower()}">{role}</span>' if pos_id <= 5 else ""

        badge_src = CLUB_B64_CACHE.get(t_ab, f"assets/clubs/{t_ab}.png")
        return f"""
        <div class="pitch-player-card" id="pcard-{s_key}-{pid}" onclick="openReceipt({pid})" title="Click to view explainable Poisson receipt">
          <div class="pitch-photo-box">
            <img src="{img_src}" class="pitch-photo" onerror="this.src='{fallback_svg}'">
            <img src="{badge_src}" class="pitch-club-badge" onerror="this.style.display='none'">
            <span class="pitch-cap-badge" id="pcap-{s_key}-{pid}" style="display:none;">(C) 1.5x</span>
          </div>
          <div class="pitch-info">
            <div class="pitch-name">{name}</div>
            <div class="pitch-meta-row">
              <span class="pitch-pos-tag">{POS_CODE.get(pos_id, '?')}</span>
              {role_badge}
              <span class="pitch-price">C$ {pr:.1f}</span>
            </div>
            <div class="pitch-pts">{pts:.2f} pts</div>
            <button class="pitch-receipt-btn" onclick="openReceipt({pid}); event.stopPropagation();">🧾 Breakdown</button>
          </div>
        </div>
        """

    # Group starters
    coach_card = _render_pitch_card(coach.iloc[0]) if len(coach) > 0 else ""
    gol_cards = [_render_pitch_card(r) for _, r in starters[starters["position_id"] == 1].iterrows()]
    lat_cards = [_render_pitch_card(r) for _, r in starters[starters["position_id"] == 2].iterrows()]
    zag_cards = [_render_pitch_card(r) for _, r in starters[starters["position_id"] == 3].iterrows()]
    mei_cards = [_render_pitch_card(r) for _, r in starters[starters["position_id"] == 4].iterrows()]
    ata_cards = [_render_pitch_card(r) for _, r in starters[starters["position_id"] == 5].iterrows()]

    # Defensive row: LAT, ZAG, ZAG, LAT
    def_line = []
    if len(lat_cards) >= 1:
        def_line.append(lat_cards[0])
    if len(zag_cards) >= 1:
        def_line.append(zag_cards[0])
    if len(zag_cards) >= 2:
        def_line.append(zag_cards[1])
    if len(lat_cards) >= 2:
        def_line.append(lat_cards[1])

    # Bench 5 vertical columns
    bench_cols_html = []
    for pos_id, quota in [(1, 1), (2, 2), (3, 2), (4, 2), (5, 2)]:
        p_sub = bench[bench["position_id"] == pos_id]
        cards = [_render_pitch_card(r) for _, r in p_sub.iterrows()]
        bench_cols_html.append(f"""
        <div class="bench-col">
          <div class="bench-col-hdr">
            <span class="bpos">{POS_CODE[pos_id]}</span>
            <span class="bname">{POS_NAME[pos_id]} ({len(cards)}/{quota})</span>
          </div>
          <div class="bench-col-cards">{''.join(cards)}</div>
        </div>
        """)

    return f"""
    <div class="pitch-wrapper">
      <div class="pitch-turf">
        <!-- Dugout Coach Box -->
        <div class="pitch-dugout">
          <div class="dept-badge coach">👔 DUGOUT COACH • {coach_pts:.2f} pts • C$ {coach_cost:.1f}</div>
          <div class="pitch-row coach-row">{coach_card}</div>
        </div>

        <!-- Attack Department -->
        <div class="pitch-dept attack-dept">
          <div class="dept-badge attack">⚔️ ATTACKING LINE (3) • Total: <strong>{att_pts:.1f} pts</strong> • C$ {att_cost:.1f}</div>
          <div class="pitch-row">{''.join(ata_cards)}</div>
        </div>

        <!-- Midfield Department -->
        <div class="pitch-dept midfield-dept">
          <div class="dept-badge midfield">🎯 MIDFIELD ENGINE (3) • Total: <strong>{mid_pts:.1f} pts</strong> • C$ {mid_cost:.1f}</div>
          <div class="pitch-row">{''.join(mei_cards)}</div>
        </div>

        <!-- Defense Department (4 Backline + 1 Goalkeeper) -->
        <div class="pitch-dept defense-dept">
          <div class="dept-badge defense">🛡️ DEFENSIVE SHIELD (5) • Total: <strong>{def_pts:.1f} pts</strong> • C$ {def_cost:.1f}</div>
          <div class="pitch-row backline-row">{''.join(def_line)}</div>
          <div class="pitch-row gk-row">{''.join(gol_cards)}</div>
        </div>
      </div>

      <!-- Luxury Advisory Bench beneath the pitch -->
      <div class="bench-container">
        <div class="bench-header-strip">
          <div class="bench-title-text">
            <span class="bench-gem">💎</span>
            <strong>LUXURY ADVISORY BENCH (9 BEST-IN-SLOT UPGRADES)</strong>
            <span class="bench-note">— Unconstrained Points Expectation: Top replacement targets for bankrolls &gt; C$ 135</span>
          </div>
        </div>
        <div class="bench-grid">
          {''.join(bench_cols_html)}
        </div>
      </div>
    </div>
    """


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
    """Generates the full editorial cream HTML report with pure Plus Jakarta Sans, visible receipts, and Plotly club crests."""
    report_file = REPORTS_DIR / f"inference_{season}_r{target_round}.html"

    # Merge consensus with pool
    pool_merged = pool.merge(consensus, on="player_id", how="left")
    pool_merged["n_votes"] = pool_merged["n_votes"].fillna(0).astype(int)
    pool_merged["picked_by"] = pool_merged["picked_by"].fillna("None")

    # Load local club base64 images & logo for Plotly scatter plot & 100% self-contained HTML
    club_b64 = load_club_base64_dict()
    logo_b64 = load_logo_base64()
    logo_img_src = logo_b64 if logo_b64 else "assets/logo.png"
    club_b64_json = json.dumps(club_b64)

    # Build universal receipts JSON dictionary
    receipts_payload = build_receipts_payload(pool_merged)
    receipts_json = json.dumps(receipts_payload)

    # 1. Plotly Figure: Risk vs Return with Club Badges directly on top 35 stars
    top35_candidates = pool_merged.nlargest(35, ENSEMBLE_COL)
    plot_images = []
    for _, r in top35_candidates.iterrows():
        tabbr = str(r.get("team_abbr", "")).strip()
        if tabbr in club_b64:
            plot_images.append(
                dict(
                    source=club_b64[tabbr],
                    xref="x",
                    yref="y",
                    x=float(r[ENSEMBLE_COL]),
                    y=float(r["player_roll10_ceiling_q85"]),
                    sizex=0.30,
                    sizey=0.52,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                )
            )

    fig_risk = px.scatter(
        pool_merged,
        x=ENSEMBLE_COL,
        y="player_roll10_ceiling_q85",
        size="pre_match_price",
        color="position_code",
        hover_name="name",
        hover_data={
            "team_abbr": True,
            "opp_abbr": True,
            "is_home": True,
            "pre_match_price": ":.1f",
            ENSEMBLE_COL: ":.2f",
            "player_id": True,
        },
        labels={
            ENSEMBLE_COL: "CLF Meta-Ensemble Expected Score (Points)",
            "player_roll10_ceiling_q85": "Ceiling Potential (Roll10 85th Percentile)",
            "position_code": "Position",
            "pre_match_price": "Price C$",
        },
        title=f"Risk vs. Reward Spectrum — Season {season} Round {target_round} Candidates",
        template="plotly_white",
        color_discrete_map={
            "GOL": "#0284c7",
            "LAT": "#059669",
            "ZAG": "#10b981",
            "MEI": "#6366f1",
            "ATA": "#d97706",
            "TEC": "#475569",
        },
    )
    fig_risk.update_layout(
        images=plot_images,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbf9f4",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1e293b"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=520,
    )
    fig_risk.update_xaxes(gridcolor="#e8e5de", zerolinecolor="#cbd5e1")
    fig_risk.update_yaxes(gridcolor="#e8e5de", zerolinecolor="#cbd5e1")
    risk_html = fig_risk.to_html(full_html=False, include_plotlyjs="cdn")

    # 2. Consensus bar chart (Styled for Cream Light Theme)
    top_consensus = pool_merged[pool_merged["n_votes"] >= 2].sort_values(
        ["n_votes", ENSEMBLE_COL], ascending=[False, False]
    ).head(15)
    fig_consensus = px.bar(
        top_consensus,
        x="name",
        y=ENSEMBLE_COL,
        color="n_votes",
        color_continuous_scale=["#cbd5e1", "#38bdf8", "#34d399", "#f59e0b", "#059669"],
        hover_data=["position_code", "team_abbr", "opp_abbr", "picked_by"],
        labels={"name": "Player", ENSEMBLE_COL: "Meta-Ensemble Score", "n_votes": "Specialist Votes"},
        title="Top Consensus Candidates (Selected across multiple Independent Specialists)",
        template="plotly_white",
    )
    fig_consensus.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbf9f4",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1e293b"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    fig_consensus.update_xaxes(gridcolor="#e8e5de")
    fig_consensus.update_yaxes(gridcolor="#e8e5de")
    consensus_html = fig_consensus.to_html(full_html=False, include_plotlyjs=False)

    # Lineup data dictionary for Captaincy Engine
    lineups_js_data = {}

    # Quick Breakdown Chips for Top 10 Stars
    top10_chips = []
    for _, cr in pool_merged.nlargest(10, ENSEMBLE_COL).iterrows():
        cpid = int(cr["player_id"])
        cname = str(cr["name"])
        cclub = str(cr["team_abbr"])
        top10_chips.append(f'<button class="quick-chip" onclick="openReceipt({cpid})">🧾 {cname} ({cclub})</button>')
    quick_chips_html = f"""
    <div class="quick-receipt-strip">
      <span class="quick-receipt-label">⚡ QUICK BREAKDOWN RECEIPTS:</span>
      {''.join(top10_chips)}
    </div>
    """

    # 3. Build Category 1: Strategic Lineups
    strat_cards = []
    if strategic_suite is not None and "rosters" in strategic_suite:
        for s_key, pkg in strategic_suite["rosters"].items():
            meta = pkg["meta"]
            sq_s = pkg["squad"].merge(consensus, on="player_id", how="left")
            sq_s["n_votes"] = sq_s["n_votes"].fillna(0).astype(int)

            # Store starter player IDs for captaincy engine
            starters_only = sq_s[sq_s["lineup_slot"] == "STARTER"]
            outfield_starters = starters_only[starters_only["position_id"].isin([2, 3, 4, 5])]
            lineups_js_data[s_key] = [int(p) for p in outfield_starters["player_id"].tolist()]

            s_cols = [
                "lineup_slot", "position_code", "role_class", "name", "team_abbr", "opp_abbr",
                "pre_match_price", ENSEMBLE_COL, "gen_floor_p10", "pred_Generative_Poisson",
                "gen_ceiling_p90", "upside_spread", "n_votes"
            ]
            s_ren = {
                "lineup_slot": "Slot", "position_code": "Pos", "role_class": "Class",
                "name": "Player", "team_abbr": "Club", "opp_abbr": "Opp",
                "pre_match_price": "Price C$", ENSEMBLE_COL: "Meta-Ens",
                "gen_floor_p10": "Floor P10", "pred_Generative_Poisson": "Gen Mean",
                "gen_ceiling_p90": "Ceil P90", "upside_spread": "+Δ Spread", "n_votes": "Votes"
            }
            s_tbl_html = render_compact_table(sq_s, s_cols, s_ren)
            pitch_html = render_tactical_pitch(sq_s, s_key)

            conf_badge = (
                '<span class="badge-status success">✅ 0 Matchup Conflicts</span>'
                if meta["n_conflicts"] == 0
                else f'<span class="badge-status warning">⚠️ {meta["n_conflicts"]} Active Conflict(s)</span>'
            )

            strat_cards.append(f"""
            <div class="roster-card" id="card-{s_key}">
              <div class="roster-card-header">
                <div>
                  <h3 class="roster-title">{meta['strategy_title']}</h3>
                  <p class="roster-desc">{meta['strategy_desc']}</p>
                </div>
                <div class="roster-meta-tags">
                  <span class="meta-tag cost">Starters: <strong>C$ {meta['total_price']:.2f}</strong> + Bench: <strong>C$ {meta.get('bench_price', 0.0):.2f}</strong></span>
                  <span class="meta-tag mix">{meta['n_solids']} Solids / {meta['n_wildcards']} Wildcards</span>
                  {conf_badge}
                </div>
              </div>

              <div class="stats-strip">
                <div class="stat-pill"><span class="stat-label">Correlated Floor (P10)</span> <strong>{meta['team_floor_p10']:.1f} pts</strong></div>
                <div class="stat-pill"><span class="stat-label">Expected Team Mean</span> <strong>{meta['team_mean']:.1f} pts</strong></div>
                <div class="stat-pill"><span class="stat-label">Correlated Ceiling (P90)</span> <strong style="color:#d97706;">{meta['team_ceiling_p90']:.1f} pts</strong></div>
                <div class="stat-pill"><span class="stat-label">P(Score &ge; 80 pts)</span> <strong style="color:#059669;">{meta['prob_80_plus']*100:.1f}%</strong></div>
                <div class="stat-pill"><span class="stat-label">Roster Formation</span> <strong>12 Starters + 9 Reserves (21 Players)</strong></div>
              </div>

              <!-- Interactive Captaincy Engine & View Switcher Toolbar -->
              <div class="lineup-toolbar">
                <div class="captaincy-box">
                  <div class="cap-title">👑 CAPTAIN ADVISOR (1.5x MULTIPLIER):</div>
                  <div class="cap-btn-group">
                    <button class="cap-btn active" onclick="setCaptainStrategy('{s_key}', 'mean', this)">🛡️ Mean</button>
                    <button class="cap-btn" onclick="setCaptainStrategy('{s_key}', 'ceil', this)">🚀 Ceiling (P90)</button>
                    <button class="cap-btn" onclick="setCaptainStrategy('{s_key}', 'spread', this)">⚡ +Δ Spread</button>
                  </div>
                  <div id="cap-callout-{s_key}" class="cap-callout"></div>
                </div>

                <div class="view-switcher">
                  <button class="view-btn active" onclick="switchView('{s_key}', 'table', this)">📋 Table View</button>
                  <button class="view-btn" onclick="switchView('{s_key}', 'pitch', this)">🏟️ Tactical Pitch View</button>
                </div>
              </div>

              <!-- Content Views -->
              <div id="view-table-{s_key}" class="view-container active">{s_tbl_html}</div>
              <div id="view-pitch-{s_key}" class="view-container" style="display:none;">{pitch_html}</div>
            </div>
            """)

        # Defensive Locks Table
        def_locks_df = strategic_suite.get("defensive_locks_table", pd.DataFrame())
        def_locks_html = ""
        if len(def_locks_df) > 0:
            dl_cols = ["team_abbr", "opp_abbr", "is_home", "p_sg", "meets_sg_threshold", "unit_price", "unit_expected_mean", "unit_ceiling_if_sg", "player_names"]
            dl_ren = {
                "team_abbr": "Club", "opp_abbr": "Opp", "is_home": "Home", "p_sg": "P(SG)",
                "meets_sg_threshold": "Clears >=38% Rule", "unit_price": "5-Man Cost C$",
                "unit_expected_mean": "Expected Mean", "unit_ceiling_if_sg": "Ceiling If SG Holds (+25)",
                "player_names": "5-Man Defensive Block (1 GOL + 2 LAT + 2 ZAG)"
            }
            def_locks_html = f'<div class="card" style="margin-bottom:24px;"><h3>🛡️ Fechamento de Defesa — 5-Man Club Clean Sheet Lock Evaluation</h3><p>Evaluates complete 5-man defensive units from the same club. Conditional Lock triggers only when P(SG) &ge; 38%.</p>{render_compact_table(def_locks_df.head(8), dl_cols, dl_ren, show_bench_divider=False)}</div>'

    # 4. Build Category 2: ML Black Box Models
    blackbox_cards = []
    # Master Meta Ensemble
    ens_sq = squads["CLF_Meta_Ensemble"].merge(consensus, on="player_id", how="left")
    ens_sq["n_votes"] = ens_sq["n_votes"].fillna(0).astype(int)
    cols_table = ["lineup_slot", "position_code", "name", "team_abbr", "opp_abbr", "is_home", "pre_match_price", ENSEMBLE_COL, "n_votes"]
    cols_rename = {
        "lineup_slot": "Slot", "position_code": "Pos", "name": "Player", "team_abbr": "Club",
        "opp_abbr": "Opp", "is_home": "Home", "pre_match_price": "Price C$",
        ENSEMBLE_COL: "Meta-Ens", "n_votes": "Votes",
    }
    ens_tbl = render_compact_table(ens_sq, cols_table, cols_rename)
    blackbox_cards.append(f"""
    <div id="subtab-ens" class="subtab-pane active">
      {render_academy_card('CLF_Meta_Ensemble')}
      <div class="card">
        <h3>⭐ CLF Meta Ensemble — Master 21-Player Squad</h3>
        <p>Convex blend of 4 independent architectures. 12 Starters + 9 Luxury Bench Reserves.</p>
        {ens_tbl}
      </div>
    </div>
    """)

    for s_name in ["Positional_LightGBM", "Residual_XGBoost", "Decomposed_Hybrid", "Neural_MLP"]:
        s_col = SPECIALISTS[s_name]
        sq_spec = squads[s_name].merge(consensus, on="player_id", how="left")
        sq_spec["n_votes"] = sq_spec["n_votes"].fillna(0).astype(int)
        c_cols = ["lineup_slot", "position_code", "name", "team_abbr", "opp_abbr", "is_home", "pre_match_price", s_col, "n_votes"]
        c_ren = {**cols_rename, s_col: f"{s_name} Pts"}
        s_tbl = render_compact_table(sq_spec, c_cols, c_ren)
        blackbox_cards.append(f"""
        <div id="subtab-{s_name}" class="subtab-pane" style="display:none;">
          {render_academy_card(s_name)}
          <div class="card">
            <h3>{s_name} — 21 Suggested Players</h3>
            <p>12 Starters + 9 Luxury Bench Reserves selected by {s_name}.</p>
            {s_tbl}
          </div>
        </div>
        """)

    # 5. Build Category 3: ML Open Box (Poisson Regressors)
    openbox_cards = []
    c_cols_gen = ["lineup_slot", "position_code", "name", "team_abbr", "opp_abbr", "is_home", "pre_match_price", "gen_floor_p10", "pred_Generative_Poisson", "gen_ceiling_p90", "upside_spread", "vol_multiplier", "gen_prob_mito"]
    c_ren_gen = {
        "lineup_slot": "Slot", "position_code": "Pos", "name": "Player", "team_abbr": "Club", "opp_abbr": "Opp", "is_home": "Home",
        "pre_match_price": "Price C$", "gen_floor_p10": "Floor P10", "pred_Generative_Poisson": "Mean",
        "gen_ceiling_p90": "Ceil P90", "upside_spread": "+Δ Spread", "vol_multiplier": "Vol Mult", "gen_prob_mito": "P(Mito>=8)"
    }
    # Rank A
    sq_a = squads["Generative_Expected_Mean"].merge(consensus, on="player_id", how="left")
    sq_a["n_votes"] = sq_a["n_votes"].fillna(0).astype(int)
    openbox_cards.append(f"""
    <div id="subtab-rank-a" class="subtab-pane active">
      {render_academy_card('Generative_Poisson')}
      <div class="card">
        <h3>🛡️ Rank A: Expected Mean (Safe Foundation)</h3>
        <p>Selected to minimize portfolio variance and maximize high-volume point reliability. Ideal for protecting a league lead (21 players).</p>
        {render_compact_table(sq_a, c_cols_gen, c_ren_gen)}
      </div>
    </div>
    """)
    # Rank B
    sq_b = squads["Generative_Upside_Spread"].merge(consensus, on="player_id", how="left")
    sq_b["n_votes"] = sq_b["n_votes"].fillna(0).astype(int)
    openbox_cards.append(f"""
    <div id="subtab-rank-b" class="subtab-pane" style="display:none;">
      {render_academy_card('Generative_Poisson')}
      <div class="card">
        <h3>🚀 Rank B: Upside Spread (+Δ Maximum Jump)</h3>
        <p>Ranked by pure upside spread (P90 Ceiling - Expected Mean). Identifies candidates with the largest potential positive jump above baseline (21 players).</p>
        {render_compact_table(sq_b, c_cols_gen, c_ren_gen)}
      </div>
    </div>
    """)
    # Rank C
    sq_c = squads["Generative_Boom_Bust"].merge(consensus, on="player_id", how="left")
    sq_c["n_votes"] = sq_c["n_votes"].fillna(0).astype(int)
    openbox_cards.append(f"""
    <div id="subtab-rank-c" class="subtab-pane" style="display:none;">
      {render_academy_card('Generative_Poisson')}
      <div class="card">
        <h3>💣 Rank C: Boom-or-Bust Multiplier (Lottery Differentials)</h3>
        <p>Ranked by Volatility Multiplier (P90 / Mean &ge; 2.5x). High-beta differential lottery tickets for managers chasing a deficit (21 players).</p>
        {render_compact_table(sq_c, c_cols_gen, c_ren_gen)}
      </div>
    </div>
    """)

    # 6. Build Category 4: Positional Depth Charts (Enhanced with Poisson metrics)
    depth_tables = []
    d_cols = [
        "name", "team_abbr", "opp_abbr", "is_home", "pre_match_price",
        ENSEMBLE_COL, "pred_Generative_Poisson", "gen_floor_p10", "gen_ceiling_p90", "upside_spread",
        "pred_Positional_LightGBM", "pred_Residual_XGBoost"
    ]
    d_ren = {
        "name": "Player", "team_abbr": "Club", "opp_abbr": "Opp", "is_home": "Home", "pre_match_price": "Price C$",
        ENSEMBLE_COL: "Meta-Ens", "pred_Generative_Poisson": "Gen Mean", "gen_floor_p10": "Floor",
        "gen_ceiling_p90": "Ceil", "upside_spread": "+Δ",
        "pred_Positional_LightGBM": "Pos LGBM", "pred_Residual_XGBoost": "Res XGB"
    }
    for pos_id in range(1, 7):
        p_sub = pool_merged[pool_merged["position_id"] == pos_id].nlargest(10, ENSEMBLE_COL)
        depth_tables.append(f"""
        <div style="margin-bottom:24px;">
          <h4 style="color:#0f172a; margin-bottom:8px; font-weight:800;">{POS_NAME[pos_id]} ({POS_CODE[pos_id]}) — Top 10 Depth &amp; Rates</h4>
          {render_compact_table(p_sub, d_cols, d_ren, show_bench_divider=False)}
        </div>
        """)

    # Audit Alert Banner
    if audit_info["is_future"]:
        alert_box = f'<div class="banner info"><span class="badge live">LIVE INFERENCE</span> <strong>{audit_info["message"]}</strong> Evaluated strictly on data prior to Round {target_round}.</div>'
    else:
        alert_box = f'<div class="banner warning"><span class="badge warn">HISTORICAL AUDIT</span> <strong>{audit_info["message"]}</strong> Ground truth may already be known.</div>'

    report_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CLF Decision Intelligence • Season {season} Round {target_round}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #fbf9f4;
      --card: #ffffff;
      --card-surface: #f7f4ed;
      --border: #e8e5de;
      --border-focus: #cbd5e1;
      --text: #1e293b;
      --text-muted: #64748b;
      --text-dim: #94a3b8;
      --accent-mint: #059669;
      --accent-emerald: #10b981;
      --accent-cyan: #0284c7;
      --accent-amber: #d97706;
      --accent-rose: #e11d48;
      --accent-indigo: #4f46e5;
      --pitch-grass-1: #1a5732;
      --pitch-grass-2: #144627;
      --shadow-sm: 0 2px 6px rgba(0, 0, 0, 0.03);
      --shadow-md: 0 4px 18px rgba(0, 0, 0, 0.05);
      --shadow-lg: 0 10px 30px rgba(0, 0, 0, 0.08);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
      font-feature-settings: "cv02", "cv03", "cv04", "cv11";
      padding: 28px 20px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}
    .container {{ max-width: 1360px; margin: 0 auto; }}
    
    /* Harmonious Plus Jakarta Sans Typography for ALL titles & headings */
    h1, h2, h3, h4, h5, .brand-font, .roster-title, .data-table th, .bench-title-text, .dept-badge, .tier1-btn, .tier2-btn {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
      font-feature-settings: "cv02", "cv03", "cv04", "cv11";
    }}
    h1 {{ font-size: 1.85rem; font-weight: 800; color: #0f172a; letter-spacing: -0.025em; line-height: 1.2; }}
    h2 {{ font-size: 1.30rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; line-height: 1.25; }}
    h3, .roster-title {{ font-size: 1.15rem; font-weight: 800; color: #0f172a; letter-spacing: -0.015em; line-height: 1.3; }}
    h4 {{ font-size: 0.95rem; font-weight: 700; color: #1e293b; letter-spacing: -0.01em; }}
    h5 {{ font-size: 0.82rem; font-weight: 700; color: #334155; text-transform: uppercase; letter-spacing: 0.04em; }}

    .mono, .data-table td, .stat-pill strong, .font-price, .font-pct, .font-stat {{
      font-family: 'JetBrains Mono', monospace;
    }}

    /* Header */
    header {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 26px 32px;
      margin-bottom: 22px;
      box-shadow: var(--shadow-md);
      position: relative;
    }}
    .header-top-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 12px;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}
    .header-logo {{
      width: 54px;
      height: 54px;
      object-fit: contain;
      border-radius: 10px;
      box-shadow: var(--shadow-sm);
    }}
    .header-title-box .subtitle {{
      color: var(--text-muted);
      font-size: 0.92rem;
      margin-top: 3px;
    }}
    .header-timestamp-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 999px;
      background: #f1ede4;
      border: 1px solid #dfd8cb;
      font-size: 0.76rem;
      font-weight: 700;
      color: #475569;
      font-family: 'JetBrains Mono', monospace;
    }}
    .stats-strip {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    .stat-pill {{
      background: var(--card-surface);
      border: 1px solid var(--border);
      padding: 8px 14px;
      border-radius: 10px;
      font-size: 0.82rem;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .stat-label {{ font-size: 0.70rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .stat-pill strong {{ color: #0f172a; font-size: 0.94rem; }}

    /* Quick Receipts Strip */
    .quick-receipt-strip {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 14px;
      margin-bottom: 18px;
      box-shadow: var(--shadow-sm);
    }}
    .quick-receipt-label {{
      font-size: 0.74rem;
      font-weight: 800;
      color: #059669;
      letter-spacing: 0.04em;
    }}
    .quick-chip {{
      background: #f1ede4;
      border: 1px solid #dcd7cc;
      color: #0f172a;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.74rem;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s;
    }}
    .quick-chip:hover {{
      background: #059669;
      color: #ffffff;
      border-color: #059669;
      transform: translateY(-1px);
    }}

    /* Chart Instruction Banner */
    .chart-banner {{
      display: flex;
      align-items: center;
      gap: 10px;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      color: #1e40af;
      padding: 8px 14px;
      border-radius: 10px;
      font-size: 0.80rem;
      margin-bottom: 14px;
    }}
    .chart-banner-badge {{
      background: #2563eb;
      color: #ffffff;
      padding: 2px 7px;
      border-radius: 5px;
      font-size: 0.70rem;
      font-weight: 800;
    }}

    /* Banners */
    .banner {{
      padding: 10px 16px;
      border-radius: 10px;
      margin-top: 14px;
      font-size: 0.86rem;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .banner.info {{ background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }}
    .banner.warning {{ background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }}
    .badge {{ padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 800; }}
    .badge.live {{ background: #059669; color: #fff; }}
    .badge.warn {{ background: #d97706; color: #fff; }}

    /* Cards */
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 22px;
      margin-bottom: 22px;
      box-shadow: var(--shadow-sm);
    }}
    .card p {{ color: var(--text-muted); font-size: 0.88rem; margin-bottom: 14px; }}

    /* Two-Tier Tab Navigation */
    .tier1-nav {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      background: #f1ede4;
      border: 1px solid #dfd8cb;
      border-radius: 14px;
      padding: 6px;
      margin-bottom: 16px;
    }}
    .tier1-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: #475569;
      padding: 10px 18px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 700;
      font-size: 0.88rem;
      transition: all 0.15s ease;
    }}
    .tier1-btn:hover {{ color: #0f172a; background: rgba(255, 255, 255, 0.6); }}
    .tier1-btn.active {{
      background: #0f172a;
      color: #ffffff;
      border-color: #0f172a;
      box-shadow: var(--shadow-sm);
    }}

    .tier2-nav {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 18px;
      padding: 4px;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 10px;
    }}
    .tier2-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: #64748b;
      padding: 7px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 700;
      transition: all 0.15s ease;
    }}
    .tier2-btn:hover {{ color: #0f172a; background: #f8fafc; }}
    .tier2-btn.active {{
      background: #e2e8f0;
      color: #0f172a;
      border-color: #cbd5e1;
    }}

    .tier1-pane, .subtab-pane {{ display: none; }}
    .tier1-pane.active, .subtab-pane.active {{ display: block; }}

    /* Compact Table Typography */
    .tbl-wrap {{ overflow-x: auto; border-radius: 12px; border: 1px solid var(--border); background: #ffffff; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; text-align: left; }}
    .data-table th {{
      background: #f4f0e8;
      color: #475569;
      padding: 9px 10px;
      font-weight: 800;
      border-bottom: 2px solid #e0dbce;
      text-transform: uppercase;
      font-size: 0.70rem;
      letter-spacing: 0.05em;
      position: sticky;
      top: 0;
      z-index: 5;
      white-space: nowrap;
    }}
    .data-table td {{
      padding: 7px 10px;
      border-bottom: 1px solid #f1ede4;
      color: #334155;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .data-table tr:hover td {{ background: #f7faf8; }}

    /* Player cell layout */
    .player-flex {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .player-thumb {{
      width: 26px;
      height: 26px;
      border-radius: 999px;
      object-fit: cover;
      background: #e2e8f0;
      border: 1px solid #cbd5e1;
    }}
    .club-thumb {{
      width: 18px;
      height: 18px;
      object-fit: contain;
    }}
    .player-name-link {{
      font-weight: 700;
      color: #0f172a;
      cursor: pointer;
      text-decoration: underline;
      text-decoration-color: rgba(5, 150, 105, 0.4);
      text-underline-offset: 3px;
      transition: color 0.15s;
    }}
    .player-name-link:hover {{
      color: var(--accent-mint);
      text-decoration-color: var(--accent-mint);
    }}

    /* Breakdown Button */
    .btn-receipt-pill {{
      background: #f1ede4;
      border: 1px solid #dcd7cc;
      color: #0f172a;
      padding: 3px 9px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s ease;
      white-space: nowrap;
    }}
    .btn-receipt-pill:hover {{
      background: #059669;
      color: #ffffff;
      border-color: #059669;
      box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3);
      transform: translateY(-1px);
    }}

    /* Pills */
    .pill-slot {{ padding: 2px 7px; border-radius: 5px; font-size: 0.68rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
    .pill-slot.starter {{ background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }}
    .pill-slot.coach {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }}
    .pill-slot.bench {{ background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}

    .pill-role {{ padding: 2px 6px; border-radius: 4px; font-size: 0.68rem; font-weight: 700; }}
    .pill-role.solid {{ background: #f0fdf4; color: #15803d; }}
    .pill-role.wildcard {{ background: #fefce8; color: #a16207; }}
    .pill-role.coach {{ background: #f8fafc; color: #475569; }}

    .pill-pos {{ padding: 2px 6px; border-radius: 4px; font-size: 0.68rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
    .pos-gol {{ background: #e0f2fe; color: #0369a1; }}
    .pos-lat {{ background: #ecfdf5; color: #047857; }}
    .pos-zag {{ background: #dcfce7; color: #15803d; }}
    .pos-mei {{ background: #e0e7ff; color: #4338ca; }}
    .pos-ata {{ background: #fef3c7; color: #b45309; }}
    .pos-tec {{ background: #f1f5f9; color: #475569; }}

    /* Bench Divider Row */
    .bench-divider-row td {{
      padding: 0 !important;
      border-top: 2px dashed #f59e0b !important;
      border-bottom: 2px dashed #f59e0b !important;
    }}
    .bench-divider-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 14px;
      background: #fef9c3;
    }}
    .bench-badge {{ font-weight: 800; font-size: 0.78rem; color: #92400e; }}
    .bench-subtext {{ font-size: 0.74rem; color: #78350f; font-family: 'JetBrains Mono', monospace; }}

    /* Strategic Roster Cards */
    .roster-card {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 26px;
      box-shadow: var(--shadow-sm);
    }}
    .roster-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 14px;
      margin-bottom: 14px;
    }}
    .roster-title {{ font-size: 1.25rem; font-weight: 800; color: #0f172a; line-height: 1.3; }}
    .roster-desc {{ font-size: 0.86rem; color: var(--text-muted); margin-top: 4px; max-width: 820px; }}
    .roster-meta-tags {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .meta-tag {{ padding: 4px 10px; border-radius: 8px; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; }}
    .meta-tag.cost {{ background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }}
    .meta-tag.mix {{ background: #f5f3ff; color: #5b21b6; border: 1px solid #ddd6fe; }}
    .badge-status.success {{ background: #ecfdf5; color: #047857; padding: 4px 10px; border-radius: 8px; font-size: 0.76rem; font-weight: 700; }}
    .badge-status.warning {{ background: #fffbeb; color: #b45309; padding: 4px 10px; border-radius: 8px; font-size: 0.76rem; font-weight: 700; }}

    /* Lineup Toolbar (Captaincy + View Switcher) */
    .lineup-toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 14px;
      background: var(--card-surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 16px;
      margin: 16px 0;
    }}
    .captaincy-box {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .cap-title {{ font-size: 0.76rem; font-weight: 800; color: #475569; letter-spacing: 0.04em; }}
    .cap-btn-group {{ display: inline-flex; border-radius: 8px; overflow: hidden; border: 1px solid #cbd5e1; }}
    .cap-btn {{
      background: #ffffff;
      border: none;
      padding: 6px 12px;
      font-size: 0.76rem;
      font-weight: 700;
      color: #475569;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .cap-btn:hover {{ background: #f1f5f9; }}
    .cap-btn.active {{ background: #0f172a; color: #ffffff; }}
    .cap-callout {{
      font-size: 0.78rem;
      color: #0f172a;
      background: #fef3c7;
      border: 1px solid #fde68a;
      padding: 5px 10px;
      border-radius: 6px;
    }}

    .view-switcher {{ display: inline-flex; gap: 6px; }}
    .view-btn {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 0.78rem;
      font-weight: 700;
      color: #334155;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .view-btn:hover {{ background: #f8fafc; }}
    .view-btn.active {{ background: #0f172a; color: #ffffff; border-color: #0f172a; }}

    /* Tactical Soccer Pitch */
    .pitch-wrapper {{
      margin-top: 14px;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid #cbd5e1;
      box-shadow: var(--shadow-md);
    }}
    .pitch-turf {{
      background: radial-gradient(circle at 50% 50%, var(--pitch-grass-1) 0%, var(--pitch-grass-2) 100%);
      padding: 24px;
      position: relative;
    }}
    .pitch-dept {{
      border: 2px dashed rgba(255, 255, 255, 0.35);
      border-radius: 14px;
      padding: 16px 12px;
      margin-bottom: 14px;
      background: rgba(0, 0, 0, 0.12);
      backdrop-filter: blur(2px);
    }}
    .pitch-dugout {{
      margin-bottom: 12px;
    }}
    .dept-badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      margin-bottom: 10px;
      text-transform: uppercase;
    }}
    .dept-badge.attack {{ background: rgba(245, 158, 11, 0.25); color: #fef3c7; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .dept-badge.midfield {{ background: rgba(99, 102, 241, 0.25); color: #e0e7ff; border: 1px solid rgba(99, 102, 241, 0.4); }}
    .dept-badge.defense {{ background: rgba(16, 185, 129, 0.25); color: #d1fae5; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .dept-badge.coach {{ background: rgba(241, 245, 249, 0.25); color: #f8fafc; border: 1px solid rgba(241, 245, 249, 0.4); }}

    .pitch-row {{
      display: flex;
      justify-content: space-around;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .backline-row {{ margin-bottom: 14px; }}
    .coach-row {{ justify-content: flex-start; }}

    /* Pitch Player Card */
    .pitch-player-card {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      padding: 8px 10px;
      width: 148px;
      text-align: center;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s, border-color 0.15s;
      position: relative;
    }}
    .pitch-player-card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
      border-color: var(--accent-cyan);
    }}
    .pitch-player-card.is-captain {{
      border: 2px solid #f59e0b;
      box-shadow: 0 0 16px rgba(245, 158, 11, 0.5);
    }}

    .pitch-photo-box {{
      position: relative;
      width: 44px;
      height: 44px;
      margin: 0 auto 6px auto;
    }}
    .pitch-photo {{
      width: 44px;
      height: 44px;
      border-radius: 999px;
      object-fit: cover;
      background: #e2e8f0;
      border: 2px solid #ffffff;
    }}
    .pitch-club-badge {{
      position: absolute;
      bottom: -2px;
      right: -4px;
      width: 18px;
      height: 18px;
      object-fit: contain;
      filter: drop-shadow(0 1px 2px rgba(0,0,0,0.4));
    }}
    .pitch-cap-badge {{
      position: absolute;
      top: -4px;
      left: -4px;
      background: #d97706;
      color: #ffffff;
      font-size: 0.60rem;
      font-weight: 900;
      padding: 1px 5px;
      border-radius: 4px;
      letter-spacing: 0.05em;
    }}
    .pitch-name {{
      font-size: 0.78rem;
      font-weight: 800;
      color: #0f172a;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .pitch-meta-row {{
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 4px;
      margin: 2px 0 4px 0;
    }}
    .pitch-pos-tag {{ font-size: 0.64rem; font-weight: 800; color: #64748b; font-family: 'JetBrains Mono', monospace; }}
    .pitch-role-pill {{ font-size: 0.60rem; padding: 1px 4px; border-radius: 3px; font-weight: 700; }}
    .pitch-role-pill.solid {{ background: #dcfce7; color: #166534; }}
    .pitch-role-pill.wildcard {{ background: #fef9c3; color: #854d0e; }}
    .pitch-price {{ font-size: 0.68rem; color: #475569; font-family: 'JetBrains Mono', monospace; }}
    .pitch-pts {{
      font-size: 0.82rem;
      font-weight: 800;
      color: #059669;
      font-family: 'JetBrains Mono', monospace;
    }}
    .pitch-receipt-btn {{
      margin-top: 5px;
      background: #f1ede4;
      border: 1px solid #dcd7cc;
      color: #0f172a;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.66rem;
      font-weight: 800;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .pitch-receipt-btn:hover {{
      background: #059669;
      color: #ffffff;
      border-color: #059669;
    }}

    /* 5-Column Bench Container */
    .bench-container {{
      background: #f7f4ed;
      border-top: 2px solid #e0dbce;
      padding: 16px 20px;
    }}
    .bench-header-strip {{
      margin-bottom: 12px;
    }}
    .bench-title-text {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.82rem;
      color: #0f172a;
      flex-wrap: wrap;
    }}
    .bench-gem {{ font-size: 0.95rem; }}
    .bench-note {{ color: #78350f; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; }}
    .bench-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
    }}
    .bench-col {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }}
    .bench-col-hdr {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #e8e5de;
      padding-bottom: 6px;
      margin-bottom: 8px;
    }}
    .bench-col-hdr .bpos {{
      font-weight: 900;
      font-size: 0.72rem;
      color: #0f172a;
      font-family: 'JetBrains Mono', monospace;
    }}
    .bench-col-hdr .bname {{
      font-size: 0.68rem;
      color: #64748b;
    }}
    .bench-col-cards {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .bench-col-cards .pitch-player-card {{
      width: 100%;
      box-shadow: none;
      border-color: #e2e8f0;
      background: #faf8f5;
      padding: 6px;
    }}
    .bench-col-cards .pitch-player-card:hover {{
      background: #ffffff;
      border-color: var(--accent-cyan);
    }}

    /* Model Academy Collapsible */
    .academy-card {{
      background: #ffffff;
      border: 1px solid #dfd8cb;
      border-radius: 12px;
      margin-bottom: 16px;
      overflow: hidden;
      box-shadow: var(--shadow-sm);
    }}
    .academy-toggle {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 18px;
      background: #fbf9f4;
      cursor: pointer;
      user-select: none;
      transition: background 0.15s;
    }}
    .academy-toggle:hover {{ background: #f4f0e8; }}
    .academy-toggle-left {{ display: flex; align-items: center; gap: 10px; }}
    .academy-icon {{ font-size: 1.25rem; }}
    .academy-title {{ font-size: 0.90rem; font-weight: 800; color: #0f172a; }}
    .academy-sub {{ font-size: 0.74rem; color: #64748b; }}
    .academy-arrow {{ font-size: 0.75rem; color: #64748b; transition: transform 0.2s; }}
    .academy-drawer {{
      padding: 18px;
      border-top: 1px solid #e8e5de;
      background: #ffffff;
    }}
    .academy-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .academy-col h5 {{ font-size: 0.78rem; text-transform: uppercase; color: #475569; margin-bottom: 8px; letter-spacing: 0.04em; }}
    .academy-col p {{ font-size: 0.82rem; color: #334155; line-height: 1.5; margin-bottom: 0; }}
    .academy-table {{ width: 100%; font-size: 0.78rem; border-collapse: collapse; }}
    .academy-table td {{ padding: 4px 6px; border-bottom: 1px solid #f1f5f9; }}

    /* Universal Itemized Receipt Pop-up Modal */
    .modal-backdrop {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(15, 23, 42, 0.65);
      backdrop-filter: blur(6px);
      z-index: 999999;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}
    .modal-card {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 18px;
      width: 100%;
      max-width: 580px;
      max-height: 90vh;
      overflow-y: auto;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
      padding: 24px;
      animation: modalFade 0.18s ease-out;
    }}
    @keyframes modalFade {{
      from {{ opacity: 0; transform: scale(0.96); }}
      to {{ opacity: 1; transform: scale(1); }}
    }}
    .modal-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #e8e5de;
      padding-bottom: 14px;
      margin-bottom: 16px;
    }}
    .modal-player-flex {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .m-avatar {{
      width: 52px;
      height: 52px;
      border-radius: 999px;
      object-fit: cover;
      background: #e2e8f0;
      border: 2px solid #ffffff;
      box-shadow: var(--shadow-sm);
    }}
    .m-title-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .m-club-badge {{ width: 22px; height: 22px; object-fit: contain; }}
    .m-name {{ font-size: 1.15rem; font-weight: 800; color: #0f172a; }}
    .m-subtext {{ font-size: 0.78rem; color: #64748b; margin-top: 2px; }}
    .m-close {{
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 6px 12px;
      cursor: pointer;
      font-size: 0.78rem;
      font-weight: 700;
      color: #334155;
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .m-close:hover {{ background: #fee2e2; color: #991b1b; border-color: #fca5a5; }}

    .m-stats-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 16px;
    }}
    .m-stat-box {{
      background: var(--card-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      text-align: center;
    }}
    .m-label {{ font-size: 0.66rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }}
    .m-stat-box strong {{ font-size: 0.90rem; color: #0f172a; display: block; margin-top: 2px; }}

    .m-section-title {{
      font-size: 0.78rem;
      text-transform: uppercase;
      color: #475569;
      letter-spacing: 0.05em;
      margin: 16px 0 8px 0;
      font-weight: 800;
    }}
    .m-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
    .m-table th {{ background: #f4f0e8; padding: 7px 8px; font-weight: 800; text-align: left; }}
    .m-table td {{ padding: 6px 8px; border-bottom: 1px solid #f1ede4; }}

    .charts-grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 24px; }}
    footer {{
      text-align: center;
      color: var(--text-dim);
      font-size: 0.82rem;
      padding: 32px 0 16px 0;
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top-row">
        <div class="header-left">
          <img src="{logo_img_src}" class="header-logo" alt="CLT Logo" onerror="this.style.display='none'">
          <div class="header-title-box">
            <h1>Cartola Legends of The Future</h1>
            <div class="subtitle">Autonomous Machine Learning &amp; Strategic Portfolio Optimization • <strong>Season {season} Round {target_round}</strong></div>
          </div>
        </div>
        <div class="header-timestamp-pill">
          <span>📅</span> Generated: {report_timestamp} BRT
        </div>
      </div>
      {alert_box}
      <div class="stats-strip">
        <div class="stat-pill"><span class="stat-label">Candidate Pool</span> <strong>{len(pool)} rotation regulars</strong></div>
        <div class="stat-pill"><span class="stat-label">Starting 12 Budget</span> <strong>C$ {budget:.1f}</strong></div>
        <div class="stat-pill"><span class="stat-label">Luxury Advisory Bench</span> <strong>9 Best-in-Slot Upgrades</strong></div>
        <div class="stat-pill"><span class="stat-label">Specialist Stack</span> <strong>4 ML Models + Generative Poisson</strong></div>
        <div class="stat-pill"><span class="stat-label">Optimization Engine</span> <strong>MILP with McCormick Soft Anti-Cannibalization</strong></div>
      </div>
    </header>

    <div class="charts-grid">
      <div class="card">
        <div class="chart-banner">
          <span class="chart-banner-badge">💡 INTERACTIVE EXPLORER</span>
          <span>Top 35 Serie A stars display their official club crests on the coordinate plane. <strong>Click ANY point or club crest</strong> on the chart to instantly open their mathematical Poisson Breakdown!</span>
        </div>
        {risk_html}
      </div>
      <div class="card">{consensus_html}</div>
    </div>

    <!-- Tier 1 Navigation Bar -->
    <div class="tier1-nav">
      <button class="tier1-btn active" onclick="switchTier1('t1-strategy', this)">♟️ Strategic Lineups &amp; Decisions</button>
      <button class="tier1-btn" onclick="switchTier1('t1-blackbox', this)">🤖 ML — Black Box Models</button>
      <button class="tier1-btn" onclick="switchTier1('t1-openbox', this)">🎲 ML — Open Box (Poisson Ranks)</button>
      <button class="tier1-btn" onclick="switchTier1('t1-depth', this)">📊 Positional Depth Charts</button>
    </div>

    <!-- Category 1: Strategic Lineups -->
    <div id="t1-strategy" class="tier1-pane active">
      <div class="card">
        <h2>The CLF Strategy Team — 4 Ready-to-Play Lineup Compositions</h2>
        <p>Optimized via Mixed Integer Linear Programming (MILP) with McCormick Soft Conflict Penalties, Club Exposure Limits, and Correlated Clean Sheet Simulation. Each portfolio includes <strong>12 Starters + 9 Luxury Bench Reserves (21 Players)</strong>.</p>
        
        {quick_chips_html}
        {def_locks_html}
        {''.join(strat_cards)}
      </div>
    </div>

    <!-- Category 2: ML Black Box Models -->
    <div id="t1-blackbox" class="tier1-pane">
      <div class="tier2-nav">
        <button class="tier2-btn active" onclick="switchSubTab('subtab-ens', this)">⭐ CLF Meta Ensemble (Master Stack)</button>
        <button class="tier2-btn" onclick="switchSubTab('subtab-Positional_LightGBM', this)">Positional LightGBM</button>
        <button class="tier2-btn" onclick="switchSubTab('subtab-Residual_XGBoost', this)">Residual XGBoost</button>
        <button class="tier2-btn" onclick="switchSubTab('subtab-Decomposed_Hybrid', this)">Decomposed Hybrid</button>
        <button class="tier2-btn" onclick="switchSubTab('subtab-Neural_MLP', this)">Neural MLP</button>
      </div>
      {''.join(blackbox_cards)}
    </div>

    <!-- Category 3: ML Open Box (Poisson Regressors) -->
    <div id="t1-openbox" class="tier1-pane">
      <div class="tier2-nav">
        <button class="tier2-btn active" onclick="switchSubTab('subtab-rank-a', this)">🛡️ Rank A: Expected Mean (Safe Foundation)</button>
        <button class="tier2-btn" onclick="switchSubTab('subtab-rank-b', this)">🚀 Rank B: Upside Spread (+Δ Maximum Jump)</button>
        <button class="tier2-btn" onclick="switchSubTab('subtab-rank-c', this)">💣 Rank C: Boom-or-Bust Multiplier (Lottery Differentials)</button>
      </div>
      {quick_chips_html}
      {''.join(openbox_cards)}
    </div>

    <!-- Category 4: Positional Depth Charts -->
    <div id="t1-depth" class="tier1-pane">
      <div class="card">
        <h2>Positional Depth Charts (Top 10 Depth across Serie A)</h2>
        <p>Comprehensive player rankings per position featuring both Black-Box point predictions and Open-Box generative Poisson arrival rates.</p>
        {''.join(depth_tables)}
      </div>
    </div>

    <footer>
      CLF Strategic Portfolio Engine • Generated at {report_timestamp} • Crafted for Thiago
    </footer>
  </div>

  <!-- Universal Itemized Receipt Pop-up Modal -->
  <div id="receipt-modal-backdrop" class="modal-backdrop" onclick="closeReceiptModal(event)">
    <div class="modal-card" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div class="modal-player-flex">
          <img id="m-photo" src="" class="m-avatar">
          <div>
            <div class="m-title-row">
              <img id="m-club-badge" src="" class="m-club-badge" onerror="this.style.display='none'">
              <span id="m-name" class="m-name"></span>
              <span id="m-pos" class="pill-pos"></span>
            </div>
            <div id="m-subtext" class="m-subtext"></div>
          </div>
        </div>
        <button class="m-close" onclick="closeReceiptModal()">✕ Close</button>
      </div>
      <div class="modal-body">
        <div class="m-stats-grid">
          <div class="m-stat-box"><span class="m-label">Meta-Ensemble</span><strong id="m-ens"></strong></div>
          <div class="m-stat-box"><span class="m-label">Poisson Mean</span><strong id="m-mean"></strong></div>
          <div class="m-stat-box"><span class="m-label">Floor (P10)</span><strong id="m-floor"></strong></div>
          <div class="m-stat-box"><span class="m-label">Ceil (P90)</span><strong id="m-ceil" style="color:#d97706;"></strong></div>
          <div class="m-stat-box"><span class="m-label">+Δ Spread</span><strong id="m-spread"></strong></div>
          <div class="m-stat-box"><span class="m-label">P(Mito &ge; 8)</span><strong id="m-mito" style="color:#059669;"></strong></div>
        </div>
        <div class="m-section-title">Itemized Poisson Rates &lambda; &times; Official Cartola Weights</div>
        <table class="m-table">
          <thead>
            <tr>
              <th>Scout</th>
              <th>Description</th>
              <th>Rate &lambda;</th>
              <th>Weight</th>
              <th>Expected Pts</th>
            </tr>
          </thead>
          <tbody id="m-scouts-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    // Embedded Universal Player Receipts State & Self-Contained Club Badges
    const RECEIPTS_DATA = {receipts_json};
    const LINEUPS_DATA = {json.dumps(lineups_js_data)};
    const CLUB_BADGES = {club_b64_json};

    // Navigation Switchers
    function switchTier1(paneId, btn) {{
      document.querySelectorAll('.tier1-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tier1-btn').forEach(el => el.classList.remove('active'));
      const target = document.getElementById(paneId);
      if (target) target.classList.add('active');
      btn.classList.add('active');
    }}

    function switchSubTab(subPaneId, btn) {{
      const parentPane = btn.closest('.tier1-pane');
      if (!parentPane) return;
      parentPane.querySelectorAll('.subtab-pane').forEach(el => el.style.display = 'none');
      parentPane.querySelectorAll('.tier2-btn').forEach(el => el.classList.remove('active'));
      const target = document.getElementById(subPaneId);
      if (target) target.style.display = 'block';
      btn.classList.add('active');
    }}

    // Model Academy Collapsible Drawer
    function toggleAcademy(id) {{
      const drawer = document.getElementById('content-' + id);
      const arrow = document.getElementById('arrow-' + id);
      if (!drawer) return;
      if (drawer.style.display === 'none' || drawer.style.display === '') {{
        drawer.style.display = 'block';
        if (arrow) arrow.textContent = '▲';
      }} else {{
        drawer.style.display = 'none';
        if (arrow) arrow.textContent = '▼';
      }}
    }}

    // Lineup View Switcher (Table vs Tactical Pitch)
    function switchView(sKey, viewType, btn) {{
      const card = document.getElementById('card-' + sKey);
      if (!card) return;
      const tblView = document.getElementById('view-table-' + sKey);
      const pitchView = document.getElementById('view-pitch-' + sKey);
      card.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      if (viewType === 'table') {{
        if (tblView) tblView.style.display = 'block';
        if (pitchView) pitchView.style.display = 'none';
      }} else {{
        if (tblView) tblView.style.display = 'none';
        if (pitchView) pitchView.style.display = 'block';
      }}
    }}

    // Universal Itemized Receipt Modal Dispatcher
    function openReceipt(playerId) {{
      const p = RECEIPTS_DATA[playerId];
      if (!p) {{
        console.warn("Receipt not found for player ID:", playerId);
        return;
      }}

      document.getElementById('m-name').textContent = p.name;
      const posEl = document.getElementById('m-pos');
      posEl.textContent = p.pos;
      posEl.className = 'pill-pos pos-' + p.pos.toLowerCase();

      document.getElementById('m-subtext').textContent = p.club + ' vs ' + p.opp + ' (' + (p.is_home ? 'Home' : 'Away') + ') • Pre-match: C$ ' + p.price.toFixed(1);
      
      const photoEl = document.getElementById('m-photo');
      const fallbackSvg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='52' height='52' viewBox='0 0 52 52'%3E%3Ccircle cx='26' cy='26' r='26' fill='%23e2e8f0'/%3E%3Ctext x='26' y='33' font-size='22' text-anchor='middle' fill='%2364748b'%3E👤%3C/text%3E%3C/svg%3E";
      photoEl.src = p.photo || fallbackSvg;
      photoEl.onerror = () => {{ photoEl.src = fallbackSvg; }};

      const clubEl = document.getElementById('m-club-badge');
      clubEl.src = CLUB_BADGES[p.club] || ('assets/clubs/' + p.club + '.png');
      clubEl.style.display = 'inline-block';

      document.getElementById('m-ens').textContent = p.ens_score.toFixed(2);
      document.getElementById('m-mean').textContent = p.gen_mean.toFixed(2);
      document.getElementById('m-floor').textContent = p.floor_p10.toFixed(1);
      document.getElementById('m-ceil').textContent = p.ceil_p90.toFixed(1);
      document.getElementById('m-spread').textContent = '+' + p.upside_spread.toFixed(1);
      document.getElementById('m-mito').textContent = Math.round(p.prob_mito * 100) + '%';

      // Populate scouts table
      const tbody = document.getElementById('m-scouts-body');
      tbody.innerHTML = '';
      (p.scouts || []).forEach(s => {{
        if (Math.abs(s.points) >= 0.05 || ['G', 'A', 'DS'].includes(s.code)) {{
          const tr = document.createElement('tr');
          const color = s.points > 0 ? '#059669' : (s.points < 0 ? '#e11d48' : '#64748b');
          tr.innerHTML = '<td><strong>' + s.code + '</strong></td>' +
                         '<td>' + s.name + '</td>' +
                         '<td class="mono">' + s.rate.toFixed(2) + '</td>' +
                         '<td class="mono">x ' + s.weight.toFixed(1) + '</td>' +
                         '<td class="mono" style="font-weight:700; color:' + color + ';">' + (s.points >= 0 ? '+' : '') + s.points.toFixed(2) + ' pts</td>';
          tbody.appendChild(tr);
        }}
      }});

      if ([1, 2, 3].includes(p.pos_id)) {{
        const tr = document.createElement('tr');
        tr.innerHTML = '<td><strong>SG</strong></td>' +
                       '<td>Saldo de Gols (Clean Sheet)</td>' +
                       '<td class="mono">' + Math.round(p.p_sg * 100) + '%</td>' +
                       '<td class="mono">x 5.0</td>' +
                       '<td class="mono" style="font-weight:700; color:#059669;">+' + p.sg_pts.toFixed(2) + ' pts</td>';
        tbody.appendChild(tr);
      }}

      const backdrop = document.getElementById('receipt-modal-backdrop');
      backdrop.style.display = 'flex';
    }}

    function closeReceiptModal(event) {{
      if (event && event.target && event.target.closest && event.target.closest('.modal-card')) return;
      document.getElementById('receipt-modal-backdrop').style.display = 'none';
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeReceiptModal();
    }});

    // Captaincy Suggestion Engine
    function setCaptainStrategy(sKey, metric, btn) {{
      const card = document.getElementById('card-' + sKey);
      if (!card) return;
      card.querySelectorAll('.cap-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const starterIds = LINEUPS_DATA[sKey] || [];
      if (starterIds.length === 0) return;

      const candidates = starterIds.map(id => RECEIPTS_DATA[id]).filter(Boolean);
      candidates.sort((a, b) => {{
        if (metric === 'mean') return b.ens_score - a.ens_score;
        if (metric === 'ceil') return b.ceil_p90 - a.ceil_p90;
        return b.upside_spread - a.upside_spread;
      }});

      if (candidates.length === 0) return;
      const topCap = candidates[0];

      // Remove existing captain highlights in this lineup
      card.querySelectorAll('.pitch-player-card').forEach(c => c.classList.remove('is-captain'));
      card.querySelectorAll('.pitch-cap-badge').forEach(b => b.style.display = 'none');

      // Highlight new captain
      const topPitchCard = document.getElementById('pcard-' + sKey + '-' + topCap.id);
      if (topPitchCard) topPitchCard.classList.add('is-captain');
      const topPitchBadge = document.getElementById('pcap-' + sKey + '-' + topCap.id);
      if (topPitchBadge) topPitchBadge.style.display = 'block';

      // Update callout box
      const callout = document.getElementById('cap-callout-' + sKey);
      if (callout) {{
        const baseScore = metric === 'mean' ? topCap.ens_score : (metric === 'ceil' ? topCap.ceil_p90 : topCap.upside_spread);
        const boosted = baseScore * 1.5;
        const bonus = baseScore * 0.5;
        const c2 = candidates[1] ? (candidates[1].name + ' (' + ((metric === 'mean' ? candidates[1].ens_score : candidates[1].ceil_p90) * 1.5).toFixed(1) + ' pts)') : '';
        const c3 = candidates[2] ? (candidates[2].name + ' (' + ((metric === 'mean' ? candidates[2].ens_score : candidates[2].ceil_p90) * 1.5).toFixed(1) + ' pts)') : '';
        callout.innerHTML = '<strong>Recommended (C): ' + topCap.name + ' (' + topCap.club + ')</strong> — ' +
                            '1.5x Boost: <strong>' + boosted.toFixed(2) + ' pts</strong> (+' + bonus.toFixed(2) + ' bonus). ' +
                            '<span style="color:#64748b;">Next: ' + c2 + ', ' + c3 + '</span>';
      }}
    }}

    // Wire Plotly Scatter Click Listener
    function wirePlotlyClicks() {{
      const plotDivs = document.querySelectorAll('.plotly-graph-div');
      plotDivs.forEach(div => {{
        if (div && div.on) {{
          div.on('plotly_click', function(data) {{
            if (data && data.points && data.points.length > 0) {{
              const pt = data.points[0];
              if (pt.customdata) {{
                for (let i = 0; i < pt.customdata.length; i++) {{
                  const val = pt.customdata[i];
                  if (RECEIPTS_DATA[val]) {{
                    openReceipt(val);
                    return;
                  }}
                }}
              }}
              if (pt.hovertext) {{
                for (const pid in RECEIPTS_DATA) {{
                  if (RECEIPTS_DATA[pid].name === pt.hovertext) {{
                    openReceipt(pid);
                    return;
                  }}
                }}
              }}
            }}
          }});
        }}
      }});
    }}

    // Initialize captain suggestions and Plotly click events on page load
    document.addEventListener('DOMContentLoaded', () => {{
      Object.keys(LINEUPS_DATA).forEach(sKey => {{
        const card = document.getElementById('card-' + sKey);
        if (card) {{
          const firstBtn = card.querySelector('.cap-btn');
          if (firstBtn) setCaptainStrategy(sKey, 'mean', firstBtn);
        }}
      }});
      setTimeout(wirePlotlyClicks, 600);
    }});
  </script>
</body>
</html>
"""
    report_file.write_text(html_content, encoding="utf-8")
    index_file = REPORTS_DIR / "index.html"
    index_file.write_text(html_content, encoding="utf-8")
    return report_file
