# PRD & Master Plan: Cartola Legends of the Future (`CLF`)

**Project Directory:** `C:\Users\thiag\Documents\CLF\`  
**Upstream Data Source:** `C:\Users\thiag\Documents\CLT\data\` (12 seasons: 2014–2025, 4,560 matches, 359,385 total player-round rows, 131,660 active pitch appearances)  
**Primary Objective:** Predict player performance in round $t$ using strictly pre-match information available at round $t-1$, paired with an interactive HTML educational academy and Plotly visual diagnostic suite.

---

## 1. Problem Statement & Core Philosophy

Predicting single-match fantasy football (Cartola FC) points is notoriously difficult because individual match outcomes are dominated by **low-frequency, high-variance Poisson events** (goals $+8.0$, clean sheets $+5.0$, red cards $-3.0$, penalty saves $+7.0$).

In previous attempts, two bottlenecks prevented success:
1. **Compute Complexity:** Naive row-by-row iteration to compute rolling player windows and opponent-conceded metrics across 350,000+ rows caused massive compute bottlenecks.
2. **Structural & Domain Traps:**
   * **Cross-Season Continuity Trap:** Treating Round 1 of 2020 as immediately following Round 38 of 2019 ignores months of off-season transfers, coaching changes, state championships, and squad overhauls.
   * **The "Arrascaeta Effect" (Regression to the Mean):** Global models (especially decision trees) pool all midfielders together, anchoring elite outliers toward the population mean unless explicit hierarchical player priors are modeled.
   * **Positional & Scoring Heterogeneity:** Goalkeepers (`GOL`), Defenders (`LAT`, `ZAG`), Midfielders (`MEI`), Attackers (`ATA`), and Coaches (`TEC`) follow fundamentally different statistical distributions and scoring rules (e.g., `SG` / Clean Sheets only apply to `GOL`, `LAT`, `ZAG`).
   * **Target Non-Stationarity:** Raw Cartola `points` changed scoring weights across eras (e.g., `RB` vs. `DS` in 2020, `DD` vs. `DE` in 2022). Using `base_norm_score` (from `CLT`) vs. raw `points` is a critical modeling choice.

---

## 2. Active Pushbacks & Architectural Trade-Offs (Learnable Pills)

### 💊 Pill 1: Why Lag Features Were Slow—And How Relational Vectorization Solves It in < 3 Seconds
* **Why it was slow:** Computing *"Average assists conceded by Opponent $B$ in their last 10 matches"* at the **player-row level** ($N = 359,385$) repeats the exact same team-level aggregation ~25 times per match. Worse, using `.apply()` or row loops in Python is $O(N^2)$.
* **The Vectorized Solution (3-Level Relational Aggregation):**
  1. **Level 1 (Team-Match Scout Cube):** Aggregate `normalized_scouts.parquet` (`played == True`) by `(season, round, team_id, position_id)` and `(season, round, team_id)`. There are only $4,560 \times 2 = 9,120$ team-match observations!
  2. **Level 2 (Opponent Conceded Matrix):** Join the Team-Match cube with `matches.parquet` so every team's stats in a match are mirrored as their opponent's **conceded** stats (`G_conceded`, `A_conceded`, `DS_conceded`, `FD_conceded`, `SG_allowed`, even split by position: e.g., *points conceded specifically to opposing `LAT` or `ATA`*).
  3. **Level 3 (Shifted Rolling & EWMA Windows):** Run `.groupby(...).shift(1).rolling(W)` on the 9,120-row Team table and the 131,660-row active Player table using **Polars / DuckDB / Vectorized Pandas**, then left-join back onto the player-round spine. Total runtime: **~1.5 to 3 seconds** on CPU.

### 💊 Pill 2: The Cross-Season Gap (Round 38 $\to$ Round 1)
* **Pushback:** We should **neither** treat Round 1 of Season $S$ as Round 39 of Season $S-1$ (too much roster/tactical drift), **nor** reset all features to `NaN` in Round 1 (which throws away the fact that Arrascaeta or Hulk is still an elite player in Round 1!).
* **The Solution (Dual-Memory State Reset + Bayesian Shrinkage):**
  * **Short-Term Form Features (`rolling_3`, `rolling_5`, `team_conceded_5`):** Strictly partitioned by `(player_id, season)` or `(team_id, season)` with a carryover decay weight $\gamma \in [0, 1]$ during Rounds 1–5.
  * **Long-Term Talent Priors (`career_baseline`, `prev_season_avg`):** Carry across seasons, decayed by an off-season retention factor (e.g., $\lambda = 0.75$) and weighted via **Empirical Bayes (James-Stein) Shrinkage**:
    $$\hat{\mu}_{i, S, r} = \frac{n_{i, S}}{n_{i, S} + k} \bar{y}_{i, S} + \frac{k}{n_{i, S} + k} \mu_{\text{prior}, i, S-1}$$
    In Round 1 ($n_{i,S} = 0$), the feature relies 100% on the shrunk prior ($\mu_{\text{prior}}$). By Round 10 ($n_{i,S} = 10$), current-season form dominates.

### 💊 Pill 3: The "Arrascaeta Problem" & Tree Model Anchoring
* **Pushback:** You hit the nail on the head—standard Gradient Boosted Trees (XGBoost/LightGBM) partition feature space into axis-aligned boxes. Because elite players represent a tiny fraction of rows, tree splits regularize (`min_child_weight`, `L2`) toward the positional average, systematically under-predicting geniuses like Arrascaeta, Hulk, or Suarez.
* **The Solution:**
  1. **Target Encoding with Out-of-Fold / Temporal Shrinkage:** Explicitly pass the player's shrunk historical baseline ($\hat{\mu}_{\text{player}}$) and **player-over-position delta** ($\hat{\mu}_{\text{player}} - \bar{\mu}_{\text{pos}}$) as features.
  2. **Residual Learning (De-meaned Target):** Test predicting the **residual over baseline** ($\Delta y_{i,t} = y_{i,t} - \hat{\mu}_{i,t-1}$) vs. raw points. When a tree predicts $\Delta y_{i,t}$ based on opponent weakness and home advantage, adding it back to Arrascaeta's baseline ($8.5 + \Delta y$) preserves his elite ceiling!
  3. **Model Family Showdown:** Compare Linear/Ridge (great at extrapolating baselines), GBDTs (LightGBM/XGBoost/CatBoost—great at non-linear interactions), and Neural Networks (MLP / Tabular ResNet with **Learned Player & Team Embeddings**).

### 💊 Pill 4: Positional Segmentation & Why Coach (`TEC`) is Not a Horoscope
* **Pushback on Coaches (`TEC`):** A Coach's score in Cartola is **not** a horoscope—it is literally a **deterministic formula**! In Cartola FC, the Coach (`position_id == 6`) receives the **exact arithmetic mean of the 11 fielded players of their team** in that match (plus small era bonuses). Therefore, predicting a Coach's score is identical to predicting **Team Expected Fantasy Points (`xFP_team`)**! Instead of a noisy individual model, `TEC` should be modeled at the **team-fixture level**—and its prediction actually serves as a powerful macro-feature for all outfield players on that team!
* **Pushback on Outfield Positions (`GOL`, `LAT`, `ZAG`, `MEI`, `ATA`):**
  * Pooling `GOL` with `ATA` in a single naive model is toxic because features like `opponent_shots_on_target` increase `GOL` points (via saves `DE`) while `SG` (Clean Sheet, $+5.0$ binary jump) creates a bimodal distribution for `GOL`, `LAT`, and `ZAG` that doesn't exist for `MEI` and `ATA`.
  * **Even deeper:** Rather than just predicting total `base_norm_score` as a single opaque number, we should compare:
    * **Direct Position-Specific Models:** 5 specialized models (`GOL`, `LAT`, `ZAG`, `MEI`, `ATA`) + 1 Team-Level `TEC` model.
    * **Decomposed Component / Two-Stage Architecture:**
      * *Sub-Model A:* Predict $P(\text{SG} = 1)$ (Clean Sheet probability via Poisson/Logistic team goal model) $\times 5.0$ pts for `GOL/LAT/ZAG`.
      * *Sub-Model B:* Predict **Base Action Score** (continuous floor: `DS`, `FS`, `FF`, `FD`, `DE`, `FC`, `CA`).
      * *Sub-Model C:* Predict **Goal/Assist Involvement** (`G`, `A`).

### 💊 Pill 5: Why `PR-AUC` Alone is Insufficient (Evaluation Metrics)
* **Pushback:** You mentioned `PR-AUC`—which is great if we frame the problem as binary classification (e.g., *"Will this player 'mitar' ($\ge 8.0$ points)?"*, where positives are rare ~10–15%). However, in Cartola, **ranking** who to draft in your starting XI matters just as much as regression error or binary classification!
* **Our Multi-Lens Evaluation Suite:**
  1. **Regression Metrics:** `RMSE`, `MAE`, and `R²` on continuous points.
  2. **Classification ("Mito" Detection, $y \ge 8.0$ pts):** `PR-AUC` (Precision-Recall AUC) and `ROC-AUC` + Calibration curves.
  3. **Ranking & Portfolio Metrics (The Real Cartola Test):**
     * **Per-Round Spearman Rank Correlation ($\rho$) & NDCG@10** per position.
     * **Simulated "Top-XI Oracle vs. Model Lineup" Backtest:** In every round of the test set (e.g., 2024–2025), pick a valid 4-3-3 lineup using model predictions vs. price/average baselines and measure total points scored across the 38 rounds!

---

## 3. Scope & Constraints

### In-Scope
1. **Target Definition & Population:**
   * Primary population: Active pitch appearances (`played == True`, $N = 131,660$) so the model learns *conditional performance given the player plays* (modeling "will the player be a starter?" is a separate pre-round lineup problem, as Cartola already provides a `provável` status flag before market close).
   * Target variable: `base_norm_score` from `CLT` (era-harmonized scoring weights so 2014–2025 are on an identical scale), with comparison against raw `points`.
2. **High-Speed Vectorized Feature Store (`features.parquet`):**
   * Player rolling/EWMA form windows ($W \in \{3, 5, 10\}$ rounds) for total score, base floor score (excluding `G`/`A`/`SG`), attacking scouts (`G`, `A`, `FD`, `FF`, `FS`), and defensive scouts (`DS`, `DE`, `SG`, `FC`).
   * Bayesian shrunk player talent priors & consistency metrics (standard deviation, floor 25th percentile, ceiling 90th percentile, home vs. away player splits).
   * Match context & Opponent Conceded features: Home/Away indicator, Elo/Rolling Team Strength, Opponent rolling goals conceded, Opponent rolling fantasy points conceded **specifically to the player's position** (e.g., how many points Opponent $B$ concedes to opposing `LAT`s or `ATA`s in their last 5/10 games).
   * Market Intelligence features: Cartola `price` and `price_variation` (at $t-1$), plus player price relative to team and positional mean (capturing crowd/market wisdom).
3. **Strict Temporal Validation Framework:**
   * Expanding-window walk-forward validation across seasons (e.g., Train: `2014–2022`, Validation: `2023`, Out-of-Sample Test: `2024–2025`)—zero future data leakage.
4. **Model Zoo Comparison:**
   * **Baselines:** Rolling 5-match average, Cartola Market Price rank, Empirical Bayes Player Prior.
   * **Linear Models:** Ridge / ElasticNet / Tweedie GLM per position.
   * **Gradient Boosted Trees:** LightGBM / XGBoost (Global vs. Position-Specific vs. Residual Target vs. Decomposed Component Models).
   * **Neural Network:** PyTorch Tabular MLP / ResNet with learned `player_id`, `team_id`, and `position_id` embeddings.
5. **Interactive Educational Suite & Standalone HTML Reports (Plotly):**
   * `reports/01_eda_and_distributions.html`: Interactive Plotly explorer of positional distributions, home/away effects, cross-season autocorrelation decay, and feature correlations.
   * `reports/02_feature_engineering_lab.html`: Visual deep-dive into lag windows, EWMA vs. Simple Moving Average, Bayesian Shrinkage curves, and SHAP / Permutation Feature Importance.
   * `reports/03_model_arena_and_backtest.html`: Model comparison dashboard (RMSE, MAE, Spearman $\rho$, PR-AUC for "Mitos", Arrascaeta/outlier residual analysis, and simulated 38-round season lineup backtest).
   * `academy/index.html`: A complete interactive **HTML Data Science Tutorial & Quiz Academy** covering every mathematical and engineering concept we build, with interactive self-check quizzes, formula breakdowns, and code walkthroughs.

### Out-of-Scope (Unless Requested)
* Scraping external live injury/news feeds or betting odds outside the `CLT` repository.
* Deploying a live production web server (all deliverables are reproducible Python scripts, Parquet datasets, trained models, and rich standalone HTML reports/tutorials).

---

## 4. Proposed Directory Architecture (`C:\Users\thiag\Documents\CLF\`)

```text
C:\Users\thiag\Documents\CLF\
├── plan.md                          # This PRD & architectural blueprint
├── checkpoint.md                    # Continuous progress & learning log
├── requirements.txt                 # Reproducible dependencies
├── src/
│   ├── 01_build_feature_store.py    # Vectorized 3-level relational feature engine (<3s)
│   ├── 02_baseline_and_linear.py    # Heuristics, Empirical Bayes, Ridge/GLM models
│   ├── 03_tree_models.py            # LightGBM/XGBoost (Global, Positional, Residual, Decomposed)
│   ├── 04_neural_embeddings.py      # Tabular Neural Net with Player/Team embeddings
│   ├── 05_evaluate_and_backtest.py  # PR-AUC, Spearman, NDCG, & 38-round Lineup Simulator
│   └── 06_generate_html_suite.py    # Builds standalone Plotly reports + Interactive Academy
├── data/
│   ├── team_match_cube.parquet      # Level 1 & 2 aggregated team & opponent conceded metrics
│   ├── clf_feature_store.parquet    # Full 131,660-row feature matrix ready for ML
│   └── predictions_oof.parquet      # Out-of-sample predictions across all model families
├── reports/
│   ├── 01_eda_and_distributions.html
│   ├── 02_feature_engineering_lab.html
│   └── 03_model_arena_and_backtest.html
└── academy/
    └── index.html                   # Interactive HTML Course, Learnable Pills & Quizzes
```

---

## 5. Step-by-Step Implementation Plan

1. **Phase 1 — Alignment & Sign-Off (Current Step):**
   * Review this PRD and resolve the Open Questions below.
2. **Phase 2 — Vectorized Feature Store Engineering (`01_build_feature_store.py`):**
   * Build the Team-Match cube, Opponent-Conceded positional matrix, Player rolling/EWMA lags, Bayesian talent priors, and cross-season decay logic. Verify zero temporal leakage with automated assertions.
3. **Phase 3 — Exploratory Data Analysis & Feature Evaluation Report (`01_eda_and_distributions.html` & `02_feature_engineering_lab.html`):**
   * Generate standalone interactive Plotly HTML reports to visually inspect feature distributions, cross-season resets, correlation matrices, and Mutual Information / univariate predictive power before training complex models.
4. **Phase 4 — Modeling Arena (`02_baseline_and_linear.py`, `03_tree_models.py`, `04_neural_embeddings.py`):**
   * Train and compare Baselines, Ridge Regression, LightGBM/XGBoost (Global vs. Positional vs. Residual vs. Decomposed), and Neural Embeddings on the temporal split (`2014–2022` Train, `2023` Val, `2024–2025` Test).
5. **Phase 5 — Evaluation, Backtesting & The Interactive HTML Academy (`05_evaluate_and_backtest.py`, `06_generate_html_suite.py`):**
   * Compute Regression (`MAE`, `RMSE`), Classification (`PR-AUC` for $\ge 8.0$ pts), Ranking (`Spearman`, `NDCG@10`), and run the 38-round Cartola lineup simulator.
   * Build `reports/03_model_arena_and_backtest.html` and `academy/index.html` (interactive tutorial + quizzes).

---

## 6. Open Questions for Thiago

Before we write any code, let's align on these 4 design decisions:

1. **Target Score Choice (`base_norm_score` vs. raw `points`):**
   * In `CLT`, we created `base_norm_score`, which applies modern consistent Cartola weights across all 12 seasons (`2014–2025`) and harmonizes `RB -> DS` and `DD -> DE`. Should we use `base_norm_score` as our primary prediction target so a model trained on 2014–2023 speaks the exact same scoring language as 2024–2025?
2. **Active Players (`played == True`) vs. All Rostered Rows:**
   * Out of 359,385 rows, **131,660** are active appearances (`played == True`), while the remaining ~227k are bench/unfielded rows with `0.0` points. I strongly recommend filtering our training and evaluation spine to `played == True` (conditional performance given the player takes the pitch), while computing lags based on *actual matches played by the player* (e.g., player's last 5 appearances) plus a `rounds_since_last_played` feature to capture rust/injury return. Do you agree with this framing?
3. **Definition of a "Mito" for `PR-AUC`:**
   * Since we will track both continuous ranking/regression metrics AND binary classification (`PR-AUC`), what threshold defines a positive "Mito" class for you?
     * **Option A:** Fixed threshold (e.g., `score >= 8.0` or `score >= 10.0` points).
     * **Option B:** Relative positional quantile per round (e.g., **Top 15% of scorers in that position in that round**, which automatically adjusts for tough defensive rounds vs. high-scoring rounds).
     * **Option C:** Track both in the evaluation report!
4. **Era Window (`2014–2025` full 12 seasons vs. `2018–2025` modern era):**
   * Should we feed all 12 seasons (`2014–2025`) into training, or evaluate whether restricting training to recent seasons (or weighting recent seasons higher) improves test performance on `2024–2025`?
