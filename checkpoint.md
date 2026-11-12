# CLF — Cartola Legends of the Future: Checkpoint & Learning Log

**Date:** November 12, 2026  
**Location:** `C:\Users\thiag\Documents\CLF\`  
**Status:** Phases 1–9 Complete (Feature Store, 10-Model Arena, 38-Round Backtest, HTML Academy Suite, 2026 Live Ingestion, Click-and-Go CLI Engine, Generative Scout Engine, MILP Strategic Portfolio Optimizer, and 76-Round Solids vs. Wildcards Empirical Study)

---

## 1. Completed Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **PRD & Master Plan** | [plan.md](file:///C:/Users/thiag/Documents/CLF/plan.md) | Architectural specification, trade-off analysis, and 5-pill design rationale. |
| **Vectorized Feature Engine** | [01_build_feature_store.py](file:///C:/Users/thiag/Documents/CLF/src/01_build_feature_store.py) | 3-level relational vectorization (`team_match_cube` $\to$ `pos_def` $\to$ `clf_feature_store.parquet` with 131,660 rows $\times$ 152 columns in **~11 seconds**). |
| **Model Zoo & Backtest Engine** | [02_train_and_evaluate_models.py](file:///C:/Users/thiag/Documents/CLF/src/02_train_and_evaluate_models.py) | Trains & evaluates 10 model families across strict temporal splits (`2014–2022` Train, `2023` Val, `2024–2025` Test) + 38-round 4-3-3 Cartola lineup backtest. |
| **HTML & Academy Generator** | [03_generate_html_suite.py](file:///C:/Users/thiag/Documents/CLF/src/03_generate_html_suite.py) | Generates the 3 standalone Plotly dashboards and the Interactive HTML Academy. |
| **2026 Raw Ingestion Engine** | [04_ingest_2026.py](file:///C:/Users/thiag/Documents/CLF/src/04_ingest_2026.py) | Acquires live snapshots (`rodada-1..34.csv`), de-accumulates cumulative scouts, maps team IDs (Remo `364`), and aligns fixtures. |
| **Round 35 Specialist Inference** | [05_infer_2026_round35.py](file:///C:/Users/thiag/Documents/CLF/src/05_infer_2026_round35.py) | Out-of-sample check on 2026 R1–34 + 15-player squad suggestions per specialist for Round 35. |
| **Model Registry & Persistence** | [model_registry.py](file:///C:/Users/thiag/Documents/CLF/src/model_registry.py) | Serializes trained specialists into `data/models/` via `joblib`, enabling **0.088s instantaneous loading**. |
| **Automated Round Fetcher** | [round_fetcher.py](file:///C:/Users/thiag/Documents/CLF/src/round_fetcher.py) | Idempotent snapshot caching, fixture integrity check (10 matches / 20 clubs), and training grounds audit detection. |
| **Click-and-Go CLI Tool** | [run_inference.py](file:///C:/Users/thiag/Documents/CLF/run_inference.py) | Automated CLI: auto-detects round, predicts in **~10 seconds**, prints 4 Strategic Rosters + Rank A/B/C tables, and emits Plotly HTML reports. |
| **Report 01: EDA & Physics** | [01_eda_and_distributions.html](file:///C:/Users/thiag/Documents/CLF/reports/01_eda_and_distributions.html) | Positional distributions, Clean Sheet (`SG`) bimodality, Home/Away splits, Cross-Season autocorrelation decay, and 2020 R38 sanitization. |
| **Report 02: Feature Lab** | [02_feature_engineering_lab.html](file:///C:/Users/thiag/Documents/CLF/reports/02_feature_engineering_lab.html) | Arrascaeta 60-game lag comparison (`Rolling-3` vs `EWMA` vs `Dual-Memory EB`), LightGBM Gain Importance per position, and Club-by-Position Vulnerability Heatmap. |
| **Report 03: Model Arena** | [03_model_arena_and_backtest.html](file:///C:/Users/thiag/Documents/CLF/reports/03_model_arena_and_backtest.html) | Full 2024–2025 Out-of-Sample Leaderboard, 38-round cumulative points curves, and Elite Star Anchoring comparison. |
| **Report 04: R35 Inference** | [04_round35_inference.html](file:///C:/Users/thiag/Documents/CLF/reports/04_round35_inference.html) | Live 2026 R35 specialist suggestions, out-of-sample metrics for R1-34, and positional depth tables. |
| **Interactive Round 35 Dashboard** | [inference_2026_r35.html](file:///C:/Users/thiag/Documents/CLF/reports/inference_2026_r35.html) | Interactive Plotly Risk vs. Reward spectrum, 4 Strategic Lineup Packages tab, Rank A/B/C tabs, and Itemized Receipts. |
| **Generative Scout Engine** | [generative_specialist.py](file:///C:/Users/thiag/Documents/CLF/src/generative_specialist.py) | 9 Poisson-deviance models + SG classifier + 2D-vectorized Monte Carlo simulator (Floor P10, Ceiling P90, Mito Prob). |
| **Generative 16:9 Presentation** | [poisson_lesson.html](file:///C:/Users/thiag/Documents/CLF/academy/poisson_lesson.html) | Zero-dependency, fixed 16:9 slide masterclass (Syne + Plus Jakarta Sans) on Poisson football physics. |
| **Strategic Portfolio Optimizer** | [portfolio_optimizer.py](file:///C:/Users/thiag/Documents/CLF/src/portfolio_optimizer.py) | MILP (`scipy.optimize.milp`) solver producing 4 distinct rosters (`Raw_Model_Blank`, `Conservative_Shield`, `Correlated_Fortress`, `Asymmetric_Chaser`) with McCormick soft conflict penalties and correlated team simulation. |
| **76-Round Portfolio Study** | [06_solids_vs_wildcards_study.py](file:///C:/Users/thiag/Documents/CLF/src/06_solids_vs_wildcards_study.py) | Out-of-sample simulation across 76 rounds (2024 & 2025) evaluating Solids vs. Wildcards ratios, Positional Risk Allocation, 5-Man Defensive Locks, and Matchup Cannibalization. |
| **Report 05: Portfolio Study** | [05_strategic_portfolio_study.html](file:///C:/Users/thiag/Documents/CLF/reports/05_strategic_portfolio_study.html) | Standalone interactive Plotly publication of the Cartola Efficient Frontier, Positional Paradox, and SG Lock calibration. |
| **Editorial Report Generator** | [report_generator.py](file:///C:/Users/thiag/Documents/CLF/src/report_generator.py) | Standalone cream editorial dashboard generator featuring 4-3-3 tactical pitch view, universal receipt modal, two-tier model navigation, and captaincy advisor. |
| **Interactive Academy** | [index.html](file:///C:/Users/thiag/Documents/CLF/academy/index.html) | 6-Module interactive data science course, live Empirical Bayes calculator, 98-feature encyclopedia, and 6 self-grading quizzes. |

---

## 2. Key Technical Findings & Learnable Pills

### 💊 Pill 1: 3-Level Relational Vectorization (Solving Compute Bottlenecks)
* Aggregating player scouts to the `(season, round, team_id)` and `(season, round, opp_id, position_id)` grain first collapses 359,385 player rows into **8,629 team-match rows**.
* Computing all rolling opponent-conceded windows (`W=3, 5, 10`) at the collapsed level runs in **1.15 seconds**, eliminating the $O(N \times W)$ row-by-row iteration that historically stalled feature engineering.

### 💊 Pill 2: The 2020 Missing Round 37 Cumulative Bleed Trap
* In `henriquepgomide/caRtola`, `2020/rodada-37.csv` was missing, and `rodada-38.csv` dumped cumulative season-to-date scouts for 493 players (causing `base_norm_score` to average `53.38` pts while true single-round `points` averaged `1.44` pts).
* Adding `sanitize_cumulative_bleed_anomalies()` (`|base_norm_score - points| > 6.5`) repaired 1,115 corrupted rows (0.8% of data) and improved Global LightGBM's 2024 lineup simulation by **+867 points**.

### 💊 Pill 3: Solving the "Arrascaeta Problem" via Residual Target Learning
* Standard `Positional_LightGBM` under-predicts elite stars (top 8% prior) by `-0.624` pts/game due to leaf L2 regularization shrinking toward the positional mean.
* Training `Residual_XGBoost` on the de-meaned residual $\Delta y_{i,t} = y_{i,t} - \hat{\mu}_{\text{dual\_EB}, i, t-1}$ and adding the player's Empirical Bayes baseline back reduced Elite Star Bias by **86% (to `-0.085` pts/game)**.

### 💊 Pill 4: Why Coach (`TEC`) is the Most Predictable Position ($\rho = 0.43$)
* Because a Coach's score is the mean of 11 fielded players, individual Poisson noise cancels out by $\sigma / \sqrt{11} \approx 3.3\times$, yielding the highest Spearman correlation and highest Top-15% Quantile `PR-AUC` (`0.33`) of all positions.

### 💊 Pill 5: The Model Registry Pattern in Production
* **Problem:** Retraining 10 model families across 12 seasons of history (131,660 rows) takes ~45–60 seconds, which ruins the user experience of an interactive CLI or automated weekly cron job.
* **Solution:** Decouple model training from inference. By serializing all trained specialists (`Ridge`, `Positional_LightGBM`, `Residual_XGBoost`, `Decomposed_Hybrid`, and `Neural_MLP`) to `data/models/*.joblib`, the models load into memory in **0.088 seconds**. Retraining is only triggered on-demand via `--retrain`.

### 💊 Pill 6: Defensive Idempotent Ingestion & Upstream Contract Validation
* **Pre-flight Check:** The `round_fetcher` checks local cached CSVs (`data/raw/{season}/rodada-{r}.csv`) before issuing any HTTP request. If already present, network calls are completely bypassed.
* **Contract Enforcement:** If a future round is requested whose prior scouts are not yet released on GitHub (e.g. asking for Round 36 before Round 35 concludes), the engine raises `MissingScoutsError` and exits gracefully with an actionable explanation instead of failing on NaN inputs.
* **Audit Detection:** If a historical round ($S \le 2024$ or $S=2026, R \le 34$) is requested, a visible `[AUDIT WARNING]` is logged to alert the user that ground truth exists and the model is running in backtest evaluation mode.

### 💊 Pill 7: Consensus Ensemble Voting as an Epistemic Hedge
* In addition to computing the continuous scalar prediction of the `CLF_Meta_Ensemble` stack:
  $$\hat{y}_{\text{Meta}} = 0.30 \cdot \hat{y}_{\text{PosLGBM}} + 0.30 \cdot \hat{y}_{\text{ResXGB}} + 0.25 \cdot \hat{y}_{\text{Decomp}} + 0.15 \cdot \hat{y}_{\text{Neural}}$$
  the CLI tallies **independent votes** across all specialists.
* Players selected unanimously ($5/5$ votes, e.g. Neymar, Pedro, Samuel Lino, Matheus Bidu, Wallace) represent picks where tree gradient boosting, residual de-meaning, decomposed Poisson Clean Sheet math, deep neural latent representations, and generative scout arrival rates all agree, creating an organic hedge against single-model algorithmic idiosyncrasies.

### 💊 Pill 8: The Generative Scout Rate Engine & Asymmetric Risk (Monte Carlo Floor vs. Ceiling)
* **The Black-Box Blind Spot:** Discriminative tree regressors predict a single point estimate $\hat{y}$ (e.g. Pedro = 7.0 pts vs. Marlon = 5.4 pts). They cannot tell you *how* those points will be earned, nor can they tell you whether a player is a high-floor tackle grinder or a high-ceiling volatile striker.
* **The Generative Arrival Formulation:** Cartola points are strictly linear:
  $$\text{Points} = \sum_{s \in \text{Scouts}} w_s \cdot N_s \implies \mathbb{E}[\text{Points}] = \sum_{s} w_s \cdot \lambda_s$$
  where $\lambda_{i,s} = \mu_{\text{pos}, s} \times \theta_{i,s} \times \delta_{j,\text{pos},s} \times \gamma_{s,\text{home}}$. This yields a **100% itemized grocery receipt** (e.g. Pedro: $0.45 \text{ G} \times 8.0 + 0.18 \text{ A} \times 5.0 + 1.25 \text{ FD} \times 1.2 + \dots = 7.57\text{ pts}$).
* **Dispersion Physics:** Our analysis of 131,660 matches revealed that attacking scouts follow textbook Poisson equidispersion ($\text{Var}/\text{Mean} \approx 1.05\text{--}1.15$), but Goalkeeper Saves (`DE`) have a $\text{Var}/\text{Mean}$ ratio of **4.26**, requiring a Negative Binomial dispersion parameter ($\alpha \approx 0.32$).
* **Dual Ranking Strategy:**
  1. **Rank by Expected Mean** ($\mathbb{E}[Y]$): Best for protecting a 1st place league lead with low variance.
  2. **Rank by Ceiling** ($P_{90}$): Best for league chasers needing 15+ point differential explosions (e.g. Pedro $P_{90} = 15.6$ pts, $P(\text{Mito}) = 39\%$).

### 💊 Pill 9: McCormick Linearization for Soft Anti-Cannibalization Constraints
* **Problem:** When an unconstrained optimizer selects an Attacker $i$ (Team $A$) and a Defender/Goalkeeper $j$ (Team $B$) in the same match, their ceilings are mutually destructive ($\operatorname{Cov}(Y_i, Y_j) < 0$): across 2018–2025, a Goalkeeper's mean score collapses from **6.38 pts** (when opposing attackers score 0 goals) to **1.89 pts** (1 goal) and **0.51 pts** (2+ goals).
* **Mathematical Formulation:** Instead of a brittle hard exclusion constraint ($x_i + x_j \le 1$) that can break solver feasibility, we introduce a continuous McCormick auxiliary variable $c_{ij} \ge 0$ in `scipy.optimize.milp`:
  $$c_{ij} \ge x_i + x_j - 1, \quad \text{Objective Penalty: } -\gamma_{ij} c_{ij}$$
  where $\gamma_{ij} = 5.0 \cdot P(SG_j) \cdot P(G_i \ge 1) + 1.2 \lambda_{G, i}$ is the **exact expected points destroyed** when attacker $i$ scores against defender $j$. Across 76 test rounds, this eliminated all **23 matchup conflicts** that the unconstrained baseline made!

### 💊 Pill 10: The Two-Regime Positional Paradox (Marathon Budget Arbitrage vs. Tiro Curto Tail Explosion)
* Holding the total portfolio mix fixed at **8 Solids + 3 Wildcards** across 76 out-of-sample rounds (2024–2025) revealed two distinct strategic regimes:
  1. **38-Round Season Marathon (`Liga Clássica`):**
     * **10 Solids / 1 Surgical Wildcard (`10S_1W`)** achieved the highest frontier season total (**2,640.9 pts/season**) and highest Mito round rate (**28.0%** of rounds $\ge 85$ pts), beating `11S_0W` (2,637.1 pts, 21.3% Mito rate).
     * Among positional allocations, **`PosB (Wildcard DEF + Solid ATT)`** won cumulative season total (**2,618.1 pts/season** vs. 2,536.0 for `PosA`). Why? Because the expected mean gap between #1 Solid Defenders and #6 Wildcard Defenders is $<0.4$ pts, and cheap Wildcard Defenders ($C\$ 5\text{--}7$) free up budget to afford every expensive Solid Superstar in Attack (`MEI/ATA` stars like Pedro, Hulk, Arrascaeta).
  2. **1-Round Knockout Sprint (`Tiro Curto / Mata-Mata`):**
     * Spending all 3 Wildcards in Midfield or Attack (**`PosC: 3 Wildcards in MEI`** and **`PosD: 3 Wildcards in ATA`**) produced the **highest single-round peaks in the entire 76-round study: 151.31 pts and 141.71 pts** (vs. 112.99 pts for `PosB`), because only attacking positions possess unbounded multi-goal Poisson right tails!

### 💊 Pill 11: "Fechamento de Defesa" (5-Man Correlated Defensive Lock) & Conditional P(SG) Thresholding
* Locking all 5 defenders (`1 GOL + 2 LAT + 2 ZAG`) from the same club binds 5 roster slots to a single Bernoulli trial $SG \sim \operatorname{Bernoulli}(P(SG))$, creating a variance boost $\approx 25 \cdot P(SG)(1 - P(SG))$.
* Across 76 out-of-sample rounds:
  * When the Clean Sheet **holds**, the 5-man defensive unit averages **36.8 pts** on its own (+25.0 pt SG bonus + tackles/saves), winning **26.3% of all rounds outright**!
  * When a single goal **breaks** the lock, the 5-man unit collapses to **11.4 pts** (a **25.4-point bimodal swing**).
  * **Conditional Trigger Rule:** Activating the 5-man lock **only** when a dominant favorite clears $P(SG) \ge 38\%$ (e.g., Flamengo in 2026 R35 at $P(SG) = 46.6\%$) and pairing that locked backline with **Solid Superstar Attackers** prevents negative-EV lock gambles in low-SG rounds.

### 💊 Pill 12: Dual-Role Bench — Decoupling Game Rule Artifacts from Managerial Advisory Gradients
* **The Artificial Restriction:** Cartola's game rule ($\text{Price}_{\text{reserve}} \le \min \text{Price}_{\text{starters}}$) forces squads with cheap budget starters to populate reserve benches with artificial $C\$ 1.00$ scrubs (e.g. Yago Melo, Athos), completely hiding high-EV alternatives.
* **The Advisory Gradient Formulation:** Decoupling the 9 reserves from starter price floors transforms the bench into an unconstrained **Best-in-Slot Marginal Spending Frontier** ranked strictly by projected points ($\mathbb{E}[Y]$).
* **The Shadow Price of Wealth:** A manager with $C\$ 167$ (or any bankroll $> C\$ 135$) can immediately identify high-gradient upgrades ($\Delta y / \Delta c$). For example, upgrading starter `Josué (C$ 6.4, 4.40 pts)` to bench `Arrascaeta (C$ 12.9, 5.60 pts)` yields $+1.20\text{ pts}$ for $+C\$ 6.50$ ($0.185\text{ pts/C\$}$, more than double the average draft efficiency).

### 💊 Pill 13: Tactical Pitch UI & Modal Dispatch Architecture (Zero-Lag Client State)
* **Problem:** Switching between tabular views, tactical soccer formations, and drilling down into Poisson receipts often introduces UX friction when bound to server re-renders, obscure inline icons, or font distortions.
* **Solution:** Embedding a normalized, client-side relational state (`RECEIPTS_DATA = { player_id: { ...scouts, rates, odds } }`) directly in the HTML payload enables:
  1. **Strict Typographic Unification:** Clean, high-legibility `Plus Jakarta Sans` throughout all titles, tables, cards, and badges, eliminating wide-font geometric stretching.
  2. **Sub-millisecond Universal Receipts:** Prominent dedicated `[🧾 Breakdown]` pill buttons on every table row, pitch card, and top Quick-Access strip, plus clickable player names and Plotly scatter dots.
  3. **Visual Scatter Plot (Club Badges on Plotly):** Base64 club crests rendered directly at `(x, y)` coordinates for the top 35 candidates, combining immediate club spotting with click-to-receipt interactivity.
  4. **Tactical 4-3-3 Pitch View:** SVG grass pitch layout with dotted departmental boundaries (Attack, Midfield, Defense) and 5 vertical luxury bench columns beneath the pitch.
  5. **Multi-Criterion Captaincy Advisor:** Dynamic 3-way toggle (Expected Mean, P90 Ceiling, or Upside Spread) that instantly re-ranks outfield starters, updates $1.5\times$ boosted projections, and pins the captain badge with zero network latency.

---

## 3. CLI Quickstart & Usage Instructions

The CLI entrypoint [run_inference.py](file:///C:/Users/thiag/Documents/CLF/run_inference.py) is ready to run directly from PowerShell or bash:

### 1. Click-and-Go (Auto-Detection)
Automatically detects the latest cached round and predicts the next unplayed round:
```powershell
python C:\Users\thiag\Documents\CLF\run_inference.py
```
*(Runs in ~11 seconds from cold start, outputs terminal table + parquet + HTML report).*

### 2. Custom Bankroll Budget (e.g. C$ 167.0)
Managers with surplus wealth can directly optimize the starting 12 at their exact bankroll:
```powershell
python C:\Users\thiag\Documents\CLF\run_inference.py --season 2026 --round 35 --budget 167
```

### 3. Specific Season & Round
```powershell
python C:\Users\thiag\Documents\CLF\run_inference.py --season 2026 --round 35
```

### 4. Graceful Error on Missing Future Data
Attempting to predict an unreleased round cleanly notifies the user:
```powershell
python C:\Users\thiag\Documents\CLF\run_inference.py --season 2026 --round 36
```
*Output: `❌ [MissingScoutsError] Round 35 scouts for Season 2026 are not available on GitHub (404 Not Found). To predict round 36, scouts through round 35 must be published first.`*

### 5. Historical Backtest Audit
Running on an already-played round emits a visible audit warning:
```powershell
python C:\Users\thiag\Documents\CLF\run_inference.py --season 2026 --round 30
```
*Output: `⚠️ [AUDIT WARNING] Season 2026 Round 30 was ALREADY PLAYED in live 2026. Running backtest audit.`*
*Output: `⚠️ [AUDIT WARNING] Season 2026 Round 30 was ALREADY PLAYED in live 2026. Running backtest audit.`*

### 5. Force Model Retraining
To refit and re-serialize the entire model zoo on latest data:
```powershell
python C:\Users\thiag\Documents\CLF\run_inference.py --retrain
```
