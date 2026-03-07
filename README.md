# 📈 Stock_It_Up

A modular, deterministic, explainable CLI-based financial decision assistant for equity evaluation using historical stock data (NSE & BSE).

> **Dataset size: ~5GB (auto-download supported)**

---

## 1️⃣ Problem Understanding

The objective was to build a conversational system that:

* Allows users to explore stock data from NSE and BSE
* Computes performance metrics over a chosen time horizon
* Applies weighted scoring based on user preference
* Adapts to different risk profiles
* Produces ranked recommendations and portfolio allocations
* Clearly explains *why* a stock is recommended
* Simulates portfolio behaviour via Monte Carlo analysis
* Screens the **entire exchange** by any metric — including positional queries (`worst`, `2nd best`, `last`)
* Allows re-sorting of analysis results by different dimensions in-session

Key constraints:

* Must not rely entirely on AI models
* Must use real dataset (2000–2023 historical CSVs)
* Must be deterministic and explainable
* Should follow modular, maintainable architecture

The system acts as a **Decision Companion**, not just a ranking calculator.

---

## 2️⃣ Assumptions Made

1. Historical daily data (2000–2023) is sufficient to compute CAGR, Volatility, Sharpe, Sortino, MDD, and Avg Volume.
2. Users compare stocks relative to each other, not against the entire market.
3. Scoring is **relative** (min–max normalisation across selected stocks).
4. Risk profile modifies weight emphasis rather than altering raw metrics.
5. Risk-free rate is fixed at **6% p.a.** (`config.py`).
6. Stage 1 portfolio volatility (independent-stock approximation) is an **upper bound** on true Sharpe when stocks are positively correlated.
7. `lower_is_better` metrics (Volatility, Max Drawdown) are inverted during positional and sort operations.
8. Directory structure:

   ```
   comp_stock_data/
       stock_data_NSE/<TICKER>/<TICKER>_<YEAR>.csv
       stock_data_BSE/<TICKER>/<TICKER>_<YEAR>.csv
       .cache/                  ← auto-built metric cache
   ```

---

## 📂 Dataset

The system uses **historical daily stock data (2000–2023) for NSE and BSE**.

The full dataset is approximately **5 GB**, which exceeds GitHub's repository size limits.
Therefore, the dataset is hosted externally and can be downloaded automatically.

### Download Instructions

After cloning the repository, run:

```bash
python download_dataset.py
```

The script will:

1. Download the dataset archive (~5 GB)
2. Extract it into the correct directory structure
3. Remove the temporary zip file

### Expected Directory Structure

After downloading, the project should contain:

```
comp_stock_data/
    stock_data_NSE/
        <TICKER>/
            <TICKER>_<YEAR>.csv
    stock_data_BSE/
        <TICKER>/
            <TICKER>_<YEAR>.csv
    .cache/
```

### Notes

* The `.cache` directory is **generated automatically** during screener queries.
* The dataset is downloaded **only once**; subsequent runs will skip the download.

---

## 3️⃣ Architecture

```
ConversationManager           → FSM orchestration, global intent routing
IntentParser                  → NLP / command routing + param extraction
SessionContext                → per-session user state
DataLoader                    → filesystem I/O + lru_cache
MetricCache                   → persistent JSON metric cache (SHA-256 fingerprinted)
MetricsEngine                 → financial calculations (CAGR, Vol, Sharpe, MDD, Sortino)
ScoringEngine                 → weighted min-max normalisation + risk-profile adjustment
ScreenerEngine                → market-wide screener (heapq, cache fast-path, fetch_position)
PortfolioEngine               → allocation + risk decomposition + covariance + Monte Carlo
ExplanationEngine             → deterministic per-stock interpretation
AllocationExplanationEngine   → portfolio-level explanation + Monte Carlo section
ResponseGenerator             → formatting only — no computation
```

Each layer has a **single responsibility**. No cross-layer logic leakage.

---

## 4️⃣ Metrics Implemented

| Metric | Engine | Formula |
|---|---|---|
| **CAGR** | MetricsEngine | `(P_end / P_start)^(1/years) − 1` |
| **Volatility** | MetricsEngine | Annualised std of log returns |
| **Avg Volume** | MetricsEngine | Mean daily traded volume |
| **Latest Price** | MetricsEngine | Most recent closing price |
| **Sharpe Ratio** | MetricsEngine | `(CAGR − Rf) / Volatility` |
| **Max Drawdown** | MetricsEngine | `max((Peak − Trough) / Peak)` |
| **Sortino Ratio** | MetricsEngine | `(CAGR − Rf) / Downside deviation` |
| **Portfolio CAGR** | PortfolioEngine | `Σ wᵢ · CAGRᵢ` |
| **Portfolio Vol (Stage 1)** | PortfolioEngine | `√(Σ wᵢ² · σᵢ²)` — independent approx |
| **Portfolio Vol (Stage 2)** | PortfolioEngine | `√(wᵀ Σ w)` — covariance-aware |
| **Portfolio Sharpe** | PortfolioEngine | `(Rp − Rf) / σp` |
| **VaR 95%** | PortfolioEngine | 5th percentile of simulated return distribution |
| **CVaR 95%** | PortfolioEngine | Mean of returns below VaR |

### Metric direction metadata (`constants.METRIC_REGISTRY`)

Every metric carries a `higher_is_better` flag used by **both** the screener sort direction logic and the sort/filter command:

| Metric | Direction |
|---|---|
| CAGR, Score, Sharpe, Sortino, Avg Volume | `higher_is_better = True` |
| Volatility, Max Drawdown | `higher_is_better = False` (lower = safer = better) |

---

## 5️⃣ Portfolio Engine — 4 Stages

### Stage 1 — Independent Approximation
```
σp² = Σ wᵢ² · σᵢ²
```
Fast. Assumes zero correlation. Documents upper-bound Sharpe assumption.

### Stage 2 — Covariance-Aware
```
σp² = wᵀ Σ w
```
Full correlation matrix. Requires user-supplied `covariance_matrix`.

### Stage 3 — Factor Models
_Planned — CAPM / multi-factor._

### Stage 4 — Monte Carlo Simulation ✅
```
r ~ N(μ, Σ)  via Cholesky decomposition
```
Pure-Python Cholesky (`LLᵀ = Σ`). 10,000 simulations by default. Returns:
- Mean return, Std dev
- VaR 95%, CVaR 95%
- Probability of loss
- Reproducible via `seed=`

---

## 6️⃣ Screener Mode — List Queries

Scan the **entire exchange** by any metric and return a top-N ranked list.

### Commands
```
top 10 NSE                     ← top 10 by CAGR (default)
top 10 NSE by cagr             ← explicit
top 10 NSE by risk             ← highest volatility (use 'lowest' for safest)
top 10 NSE by volume
top 10 NSE by sharpe
top 10 NSE by sortino
top 10 NSE by drawdown
top 10 NSE by score            ← weighted multi-metric composite
lowest 10 NSE by volatility    ← least volatile (safest)
top 5 BSE by risk-adjusted
```

### Ordinal Rank Labels
Every result row now carries an ordinal rank label (`1st`, `2nd`, `3rd` …). The table header shows `RANK` and the top result is highlighted with a **🏆 Best pick** callout.

### Metric Alias Table

| You type | Resolved metric |
|---|---|
| `by cagr` / `by growth` / `by return` | CAGR |
| `by risk` / `by volatility` / `by safe` / `by safest` | Volatility |
| `by volume` / `by avg volume` | Avg Volume |
| `by price` / `by latest price` | Latest Price |
| `by sharpe` / `by risk-adjusted` | Sharpe |
| `by sortino` / `by downside` | Sortino |
| `by drawdown` / `by mdd` / `by max drawdown` | Max Drawdown |
| `by score` / `by rating` / `by ranked` | Composite Score |

### Metric Cache (auto-built)

The first screener query per exchange builds a persistent JSON cache at `comp_stock_data/.cache/<EXCHANGE>_<N>y.json`. All subsequent queries are served in **< 1 second** instead of ~4 minutes.

- Cache is SHA-256 fingerprinted by all CSV file mtimes
- Auto-invalidated when CSVs change on disk
- Written atomically (temp → rename) — crash-safe
- Refresh manually: `rebuild cache NSE` / `refresh cache BSE`

---

## 7️⃣ Screener Mode — Positional Queries *(new)*

Ask for a **single stock** at any specific rank position.

### Supported Queries

| Query | Meaning |
|---|---|
| `best NSE by cagr` | Single highest-CAGR stock |
| `worst BSE by cagr` | Single lowest-CAGR stock |
| `2nd best NSE by score` | 2nd highest composite score |
| `3rd worst BSE by sharpe` | 3rd lowest Sharpe ratio |
| `second last NSE by cagr` | 2nd worst CAGR |
| `last BSE by volatility` | Most volatile stock (highest vol) |
| `fifth best NSE by volume` | 5th highest volume |

Supports **digit ordinals** (`2nd`, `3rd`, `10th`) and **word ordinals** (`second`, `third` … `tenth`).

### Metric-Aware Direction

`worst by volatility` correctly returns the **highest** volatility stock (not the lowest), because the system reads `METRIC_REGISTRY[metric]["higher_is_better"]` to resolve scan direction automatically:

```
worst by cagr         → asc  (lowest CAGR first)     ✅
worst by volatility   → desc (highest vol first)      ✅
best by volatility    → asc  (lowest/safest first)    ✅
```

### Deterministic Tie-Breaking

When two stocks share an identical metric value, the secondary sort key is **ticker name (alphabetical, ascending)** — ensuring positional queries always return the same answer regardless of data loading order.

### Output Card
```
========================================================
  3rd Worst  NSE  by CAGR  (3-year horizon)
--------------------------------------------------------
  Ticker    : XYZSTOCK
  CAGR      : -12.34%
  Rank      : 3rd from the bottom
========================================================
Tip: 'explain XYZSTOCK' for full breakdown.
     'top 10 NSE by cagr' to see the full list.
```

---

## 8️⃣ Analysis Session — Sort / Filter *(new)*

After completing an analysis session (choosing specific stocks), re-sort the ranked table by any dimension **without re-running the analysis**.

### Sort Commands

| Command | Effect |
|---|---|
| `sort by risk` | Volatility ↑ — safest first (lower_is_better metric) |
| `sort by returns` | CAGR ↓ — highest growth first |
| `sort by volume` | Avg Volume ↓ — most liquid first |
| `sort by score` | Restore original composite score ranking |
| `sort by price` | Latest Price ↓ — most expensive first |
| `sort by sharpe` | Sharpe ↓ — best risk-adjusted return first |
| `worst first` | Ascending direction for current sort field |
| `best first` | Descending direction |
| `sort by cagr asc` | Explicit direction override |

Key behaviour:
- `context.results` is **never mutated** — `sort by score` always restores original ranking
- `sort by risk` defaults to **ascending** (safest first), consistent with `lower_is_better`
- When sorted by a non-score field, output shows both `1st by <field>` and `🏆 Best pick (by score)` simultaneously

---

## 9️⃣ Allocation Methods

| Method | Logic |
|---|---|
| `proportional` | `wᵢ = score_i / Σ score_j` |
| `softmax` | `wᵢ = e^(score_i) / Σ e^(score_j)` |
| `risk_adjusted` | `wᵢ ∝ score_i / volatility_i` |

Constraints: `max_cap` and `min_floor` supported. Last element absorbs rounding remainder to guarantee `Σ wᵢ = 1` exactly.

---

## 🔟 Mathematical Guarantees (Tested)

| Invariant | Tolerance |
|---|---|
| `Σ allocation = 1.0` | < 1e-9 |
| `Σ risk_share = 1.0` | < 1e-9 |
| `Σ capital ≈ budget` | < 1e-2 (rounding) |
| Stage 2 Vol ≤ Stage 1 Vol (positive corr.) | Proven analytically |
| CVaR ≤ VaR | Monte Carlo invariant |
| Cholesky `L@Lᵀ = Σ` | < 1e-10 |
| Tied metric values → alphabetical ticker order | Deterministic by design |

---

## 1️⃣1️⃣ Discoverability Commands

Work in **any** conversation state:

```
list                     → list all tickers (paginated, 50/page)
list NSE                 → NSE tickers
list NSE page 3
search TCS               → prefix + substring search
help / keywords          → show all supported commands (updated with new modes)
exchanges / markets      → show available exchanges
rebuild cache NSE        → force-rebuild metric cache
refresh cache BSE
```

---

## 1️⃣2️⃣ Edge Cases Handled

✔ Allocation sum = 1.0 (float drift guarded)
✔ Risk shares sum = 1.0
✔ Capital = budget (rounding absorbed in last element)
✔ Zero-score division guarded
✔ Non-positive-definite covariance matrix raises ValueError
✔ Single-stock portfolio
✔ All stocks identical metrics (normalisation division-by-zero guard)
✔ Searching partial ticker names
✔ Pagination beyond total pages
✔ No exchange selected before screener/positional query (with helpful prompt)
✔ Missing/corrupt CSV silently skipped
✔ Cache corruption (invalid JSON) treated as cache miss → rebuild
✔ Positional query out of range (`position > available tickers`) → graceful message
✔ `worst by volatility` correctly returns highest-vol stock (not lowest)
✔ Tied metric values produce deterministic alphabetical ordering
✔ Sort commands before any analysis → helpful guidance message
✔ Test isolation — MetricCache mocked in all ConversationManager tests

---

## 1️⃣3️⃣ How to Run

### Clone & setup
```bash
git clone <repo-url>
cd Stock_It_Up
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r Stock_It_Up/requirements.txt
```

### Download the dataset
```bash
python download_dataset.py
```

### Run the CLI
```bash
cd Stock_It_Up
python main.py
```

### Run Tests
```bash
pytest                       # 443 tests, ~3 seconds
pytest tests/test_screener.py -v           # screener + positional + sort tests
pytest tests/test_metric_cache.py -v       # cache tests only
```

---

## 1️⃣4️⃣ Example Session

```
> top 10 NSE
  [builds cache first time ~4 min, instant thereafter]
  RANK   TICKER     CAGR
  1st    ...

> top 10 NSE by sharpe      ← list mode
  [instant from cache]

> worst NSE by cagr          ← positional: single worst CAGR stock
  3-year horizon card shown

> 2nd best NSE by score      ← positional: single 2nd-best score
  card shown

> NSE
> 500000
> medium
> 3 years
> TCS INFY WIPRO
  [ranked analysis table shown]

> sort by risk               ← re-sort by volatility, safest first
> sort by returns            ← re-sort by CAGR, highest first
> sort by score              ← restore original score ranking

> explain TCS
> rebuild cache NSE
```

---

## 1️⃣5️⃣ Known Limitations

| Limitation | Notes |
|---|---|
| Stage 1 Vol ignores correlation | Upper bound. Use Stage 2 with explicit covariance matrix for precision. |
| No transaction costs | Assumed frictionless rebalancing |
| No survivorship bias correction | All tickers in dataset included |
| Risk-free rate static at 6% | Set in `config.py` |
| Covariance matrix must be user-supplied | Stage 2 / Monte Carlo |
| No intraday or options data | Daily OHLC only |
| No sector / market-cap filtering | Future work |
| Word ordinals supported up to `tenth` (10) | Digit ordinals (`11th`, `50th`, …) work without limit |

---

## 1️⃣6️⃣ What Remains (Future Work)

- [ ] Stage 3 — Factor Models (CAPM / Fama-French)
- [ ] Sector & market-cap awareness (`top IT stocks`)
- [ ] Dynamic risk-free rate (RBI repo rate API)
- [ ] Web / Streamlit interface
- [ ] Benchmark comparison (vs Nifty 50 / Sensex)
- [ ] Percentile-based thresholds for explanation engine
- [ ] Context-aware sort — re-sort after screener list (not just analysis session)
- [ ] Natural-language flexibility (`runner up`, `bottom 3rd`, `second from bottom`)

---

## 1️⃣7️⃣ Final Summary

Stock_It_Up is:

* **Deterministic** — no randomness except opt-in Monte Carlo seed; tie-breaking is alphabetical by ticker
* **Explainable** — every recommendation comes with a 5-section breakdown
* **Modular** — 12 single-responsibility engines
* **Production-grade tested** — **443 tests** covering mathematical invariants, parser stress, behavioral profiles, cache lifecycle, positional queries, metric-direction inversion, and sort/filter logic
* **Fast** — screener queries served in < 1 second from fingerprinted cache
* **Risk-aware** — 4 stages of portfolio risk modeling from simple to full simulation
* **Flexible querying** — list mode (`top N`), positional mode (`worst`, `2nd best`), and in-session sort/filter

It transforms historical stock data into a structured decision-support system rather than a simple ranking script.

---
