# 📈 Stock_It_Up

A modular, deterministic, explainable CLI-based financial decision assistant for equity evaluation using historical stock data (NSE & BSE).

---

# 1️⃣ Problem Understanding

The objective was to build a conversational system that:

* Allows users to explore stock data from NSE and BSE
* Computes performance metrics over a chosen time horizon
* Applies weighted scoring based on user preference
* Adapts to different risk profiles
* Produces ranked recommendations and portfolio allocations
* Clearly explains *why* a stock is recommended
* Simulates portfolio behaviour via Monte Carlo analysis

Key constraints:

* Must not rely entirely on AI models
* Must use real dataset (2000–2023 historical CSVs)
* Must be deterministic and explainable
* Should follow modular, maintainable architecture

The system acts as a **Decision Companion**, not just a ranking calculator.

---

# 2️⃣ Assumptions Made

1. Historical daily data (2000–2023) is sufficient to compute CAGR, Volatility, Sharpe, Sortino, MDD, and Avg Volume.
2. Users compare stocks relative to each other, not against the entire market.
3. Scoring is **relative** (min–max normalisation across selected stocks).
4. Risk profile modifies weight emphasis rather than altering raw metrics.
5. Risk-free rate is fixed at **6% p.a.** (`config.py`).
6. Stage 1 portfolio volatility (independent-stock approximation) is an **upper bound** on true Sharpe when stocks are positively correlated.
7. Directory structure:

   ```
   comp_stock_data/
       stock_data_NSE/<TICKER>/<TICKER>_<YEAR>.csv
       stock_data_BSE/<TICKER>/<TICKER>_<YEAR>.csv
       .cache/                  ← auto-built metric cache
   ```

---

# 3️⃣ Architecture

```
ConversationManager      → FSM orchestration
IntentParser             → NLP / command routing
SessionContext           → per-session user state
DataLoader               → filesystem I/O + lru_cache
MetricCache              → persistent JSON metric cache (fingerprinted)
MetricsEngine            → financial calculations (CAGR, Vol, Sharpe, MDD, Sortino)
ScoringEngine            → weighted min-max normalisation
ScreenerEngine           → market-wide screener (heapq, cache fast-path)
PortfolioEngine          → allocation + risk decomposition + covariance + Monte Carlo
ExplanationEngine        → deterministic per-stock interpretation
AllocationExplanationEngine → portfolio-level explanation + Monte Carlo section
ResponseGenerator        → formatting only
```

Each layer has a single responsibility. No cross-layer logic leakage.

---

# 4️⃣ Metrics Implemented

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

---

# 5️⃣ Portfolio Engine — 4 Stages

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

# 6️⃣ Screener Mode

Scan the entire exchange by any metric and return top-N stocks.

### Commands
```
top 10 NSE                     ← top 10 by CAGR (default)
top 10 NSE by cagr             ← explicit
top 10 NSE by risk             ← volatility (lowest risk last)
top 10 NSE by volume
top 10 NSE by sharpe
top 10 NSE by sortino
top 10 NSE by drawdown
top 10 NSE by score            ← weighted multi-metric composite
lowest 10 NSE by volatility    ← least volatile (safest)
top 5 BSE by risk-adjusted
```

### Full Metric Alias Table

| You type | Metric |
|---|---|
| `by cagr` / `by growth` / `by return` | CAGR |
| `by risk` / `by risky` / `by volatility` / `by volatile` / `by safe` / `by safest` | Volatility |
| `by volume` / `by avg volume` | Avg Volume |
| `by price` / `by latest price` | Latest Price |
| `by sharpe` / `by risk-adjusted` / `by risk adjusted` | Sharpe |
| `by sortino` / `by downside` | Sortino |
| `by drawdown` / `by mdd` / `by max drawdown` | Max Drawdown |
| `by score` / `by rating` / `by ranked` | Composite Score |

### Metric Cache (auto-built)

The first screener query per exchange builds a persistent JSON cache at `comp_stock_data/.cache/<EXCHANGE>_<N>y.json`. All subsequent queries are served from the cache in **< 1 second** instead of ~4 minutes.

- Cache is fingerprinted by SHA-256 of all CSV file mtimes
- Auto-invalidated when CSVs change on disk
- Written atomically (temp → rename) — crash-safe
- Refresh manually: `rebuild cache NSE` / `refresh cache BSE`

---

# 7️⃣ Allocation Methods

| Method | Logic |
|---|---|
| `proportional` | `wᵢ = score_i / Σ score_j` |
| `softmax` | `wᵢ = e^(score_i) / Σ e^(score_j)` |
| `risk_adjusted` | `wᵢ ∝ score_i / volatility_i` |

Constraints: `max_cap` and `min_floor` supported. Last element absorbs rounding remainder to guarantee `Σ wᵢ = 1` exactly.

---

# 8️⃣ Mathematical Guarantees (Tested)

| Invariant | Tolerance |
|---|---|
| `Σ allocation = 1.0` | < 1e-9 |
| `Σ risk_share = 1.0` | < 1e-9 |
| `Σ capital ≈ budget` | < 1e-2 (rounding) |
| Stage 2 Vol ≤ Stage 1 Vol (positive corr.) | Proven analytically |
| CVaR ≤ VaR | Monte Carlo invariant |
| Cholesky `L@Lᵀ = Σ` | < 1e-10 |

---

# 9️⃣ Discoverability Commands

Work in **any** conversation state:

```
list                     → list all tickers (paginated, 50/page)
list NSE                 → NSE tickers
list NSE page 3
search TCS               → prefix + substring search
help / keywords          → show all commands
exchanges / markets      → show available exchanges
rebuild cache NSE        → force-rebuild metric cache
refresh cache BSE
```

---

# 🔟 Edge Cases Handled

✔ Allocation sum = 1.0 (float drift guarded)  
✔ Risk shares sum = 1.0  
✔ Capital = budget (rounding absorbed in last element)  
✔ Zero-score division guarded  
✔ Non-positive-definite covariance matrix raises ValueError  
✔ Single-stock portfolio  
✔ All stocks identical metrics (normalisation division-by-zero guard)  
✔ Searching partial ticker names  
✔ Pagination beyond total pages  
✔ No exchange selected before screener  
✔ Missing/corrupt CSV silently skipped  
✔ Cache corruption (invalid JSON) treated as cache miss → rebuild  
✔ Test isolation — MetricCache mocked in all ConversationManager tests  

---

# 1️⃣1️⃣ How to Run

### Clone & setup
```bash
git clone <repo-url>
cd Stock_It_Up/Stock_It_Up
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run the CLI
```bash
python main.py
```

### Run Tests
```bash
pytest                       # 389 tests, ~3 seconds
pytest tests/test_metric_cache.py -v   # cache tests only
```

---

# 1️⃣2️⃣ Example Session

```
> top 10 NSE
  [builds cache first time ~4 min, instant thereafter]

> top 10 NSE by sharpe
  [instant from cache]

> top 10 BSE by risk
  [lowest volatility stocks]

> NSE
> 500000
> medium
> 3 years
> TCS INFY WIPRO
> explain TCS
> rebuild cache NSE
```

---

# 1️⃣3️⃣ Known Limitations

| Limitation | Notes |
|---|---|
| Stage 1 Vol ignores correlation | Upper bound. Use Stage 2 with explicit covariance matrix for precision. |
| No transaction costs | Assumed frictionless rebalancing |
| No survivorship bias correction | All tickers in dataset included |
| Risk-free rate static at 6% | Set in `config.py` |
| Covariance matrix must be user-supplied | Stage 2 / Monte Carlo |
| No intraday or options data | Daily OHLC only |
| No sector / market-cap filtering | Future work |

---

# 1️⃣4️⃣ What Remains (Future Work)

- [ ] Stage 3 — Factor Models (CAPM / Fama-French)
- [ ] Sector & market-cap awareness (`top IT stocks`)
- [ ] Dynamic risk-free rate (RBI repo rate API)
- [ ] Web / Streamlit interface
- [ ] Benchmark comparison (vs Nifty 50 / Sensex)
- [ ] Percentile-based thresholds for explanation engine

---

# 1️⃣5️⃣ Final Summary

Stock_It_Up is:

* **Deterministic** — no randomness except opt-in Monte Carlo seed
* **Explainable** — every recommendation comes with a 5-section breakdown
* **Modular** — 11 single-responsibility engines
* **Production-grade tested** — 389 tests covering mathematical invariants, parser stress, behavioral profiles, and cache lifecycle
* **Fast** — screener queries served in < 1 second from fingerprinted cache
* **Risk-aware** — 4 stages of portfolio risk modeling from simple to full simulation

It transforms historical stock data into a structured decision-support system rather than a simple ranking script.

---
