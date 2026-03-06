# Build Process — Stock_It_Up

A developer diary documenting how the system evolved from a blank file to a production-grade financial decision assistant.

---

## How It Started

The project brief was deceptively simple: **help a user pick stocks**.

The first instinct was to reach for a pre-trained LLM or an external financial API. Both were ruled out immediately — the constraint was a fixed CSV dataset and deterministic, explainable output. No black boxes allowed.

The real starting point was asking: *"What does a financial analyst actually do?"*

1. Gathers historical price data
2. Calculates performance metrics
3. Normalises and weights them based on investor profile
4. Explains the recommendation

That four-step workflow became the backbone of the architecture. Each step became one engine.

---

## How My Thinking Evolved

### Phase 1 — Metrics First

The first module written was `MetricsEngine`. CAGR was trivial. Volatility required annualisation (`√252` trading days). Sharpe needed a risk-free rate — set at 6% in `config.py` rather than hardcoded so it can be changed later.

Max Drawdown was the trickiest. Initial implementation used a nested loop — O(n²). Rewrote it using a rolling peak with `cummax()` — O(n) and vastly faster on 23 years of daily data.

### Phase 2 — Scoring Engine

Normalisation was the first major design decision. Options considered:

| Method | Verdict |
|---|---|
| Z-score | Sensitive to outliers; a single extreme stock skews all scores |
| Percentile rank | Loses magnitude information |
| **Min-max (chosen)** | Bounded [0,1], preserves relative distances, predictable |

Risk profile integration was added next. Rather than separate scoring models for LOW/MEDIUM/HIGH risk, the weights themselves are adjusted — same formula, different emphasis. This kept the engine single-responsibility.

### Phase 3 — Conversation Layer

The FSM (`ConversationManager` + `SessionContext`) was written to replace an early spaghetti approach where one giant `handle_message` function held all logic. The key insight was separating *"what did the user mean"* (`IntentParser`) from *"what state are we in"* (`ConversationManager`) from *"how do we display it"* (`ResponseGenerator`).

### Phase 4 — Screener Mode

Scanning 1,800 NSE stocks every query was the first real performance challenge. The solution was a two-level approach:

1. **`heapq.nlargest` / `nsmallest`** — O(n log k) instead of full sort O(n log n)
2. **`MetricCache`** — persistent JSON cache so subsequent queries are O(k) dict lookups

Cache invalidation was the hardest part. SHA-256 fingerprinting of all CSV file modification timestamps made it both correct and efficient — no expensive file reads until the fingerprint changes.

### Phase 5 — Discoverability

Mid-project user testing revealed a key UX gap: users didn't know what tickers or exchanges were available. Added `list`, `search`, `exchanges`, and `help` as global commands that work in any conversation state.

### Phase 6 — Portfolio Engine

The PortfolioEngine required the most mathematical care. Three allocation methods were implemented (proportional, softmax, risk-adjusted). The hardest requirement was `Σ wᵢ = 1.0` exactly — floating-point arithmetic makes this non-trivial. Solution: last element absorbs the accumulated rounding remainder.

Monte Carlo simulation via Cholesky decomposition was implemented in pure Python (no NumPy for the decomposition itself) so there is no external dependency beyond standard library + `pandas`.

### Phase 7 — Rank Labels

After the screener produced results, users had no clear sense of which was 1st vs 3rd. Added `_ordinal()` helper to `screener_engine.py` and injected `rank` (int) and `rank_label` (ordinal string) into all four result-building paths. The top result got a **🏆 Best pick** callout in the formatter.

The 11/12/13 edge case (`"11th"` not `"11st"`) was a subtle gotcha — covered with a specific test.

### Phase 8 — Positional Queries

Users quickly asked: *"can I see the worst stock?"* and *"what's the 2nd best?"*. This required more than just reversing a list.

The key design choices:

**1. Intent disambiguation.**
`best` / `worst` / `last` were deliberately separated from `top N` / `lowest N`. `best` = one specific card (SCREEN_POSITION), `top N` = a ranked list (SCREEN_TOP). This required `_POSITION_RE` to be checked before `_SCREENER_RE` in `parse_intent`.

**2. Metric-aware direction.**
The most subtle bug risk was `"worst by volatility"`. Naively, `from_end=True` → `direction="asc"` — but ascending volatility gives the *safest* stocks, not the worst. The fix was reading `METRIC_REGISTRY[metric]["higher_is_better"]` at runtime:

```
worst by cagr       → asc  (lowest CAGR first)   ← correct
worst by volatility → desc (highest vol first)    ← correct
```

Without this, the entire positional feature would be silently wrong for risk metrics.

**3. Deterministic tie-breaking.**
Financial systems must be reproducible. Added secondary sort by ticker name (alphabetical) across all four screener result paths so equal metric values always resolve the same way.

### Phase 9 — Sort / Filter

The final UX gap: after analysing a chosen set of stocks, users wanted to re-examine the table from different angles without restarting. Added `SORT_RESULTS` as a global intent.

Critical constraint: `context.results` must **never be mutated**. Sort always produces a *copy*, so `sort by score` is always a valid restore command.

The `format_sorted_table` output deliberately shows *both* the alternative-sort leader and the original score winner simultaneously — so the user never loses track of the overall best pick while exploring a different dimension.

### Phase 10 — Web Application & AI Interpretations

The command-line interface was rigorous, but financial data needs visual hierarchy. We built a Django-based Web Application bridging the backend Python engines to a NextJS-style frontend.

**1. Connecting the Explanation Engine to APIs**
The `AllocationExplanationEngine` generates rich, multi-paragraph analysis on capital allocation and Monte Carlo risk boundaries. We created `ExplanationSchema` in Pydantic to strictly serialize this output structure (Summary, Rationale, Risk Distribution, Final Verdict) through the Django JSON endpoints.

**2. Fail-Safe Service Integration**
What if the AI explanation parsing fails on an edge case? The analysis data shouldn't be lost.
In `PortfolioService` and `RiskService`, the engine callback is isolated inside a `try/except` block. If `AllocationExplanationEngine.explain()` throws an exception, the error is swallowed and logged, and `explanation=None` is returned natively in the API. The UI then elegantly skips rendering the Explanation Card while seamlessly displaying the underlying Chart.js visualizations.

**3. Frontend User Experience (Ticker Picker)**
The `ticker_picker.js` was enhanced to provide immediate, context-aware feedback. Now, when a user clicks into the Ticker search box, it triggers a `fetchSuggestions` event *without* requiring them to type any characters. It automatically requests the top available tickers strictly filtered by the currently chosen Exchange (NSE vs BSE), dramatically improving discoverability of supported stocks.

---

## Alternative Approaches Considered

| Decision point | Option considered | Why rejected |
|---|---|---|
| Query parsing | Rasa / NLTK | Overkill for structured financial commands; adds training data burden |
| Metric normalisation | Z-score | Outlier-sensitive; unpredictable bounds |
| Cache storage | SQLite | JSON is sufficient and human-inspectable for debugging |
| Monte Carlo | NumPy `linalg.cholesky` | Pure-Python Cholesky kept for educational clarity + zero hidden dependency |
| Allocation | Equal weight | Ignores score differences — defeats the purpose of ranking |
| Sort direction for `worst by volatility` | Blindly `asc` | Silently wrong — fixed by reading `higher_is_better` from registry |

---

## Refactoring Decisions

### 1. Splitting `handle_message` into a global + state-dispatch pattern
Original: one large if-else chain. Problem: adding a new global command required touching the entire function. Fix: global intents handled at the top, then `_dispatch()` delegates to per-state handlers.

### 2. `_METRIC_ALIASES` extracted to local dict inside parser
Initially the alias table was in the conversation manager. Moved into `IntentParser.extract_screener_params()` (and also `extract_position_params()`) so parsing is fully self-contained and testable in isolation.

### 3. `METRIC_REGISTRY` as the single source of truth
Before Phase 8, direction logic was hardcoded at various call sites. Consolidating to `constants.METRIC_REGISTRY["higher_is_better"]` eliminated four separate places that independently decided asc/desc — and fixed the volatility direction bug system-wide.

### 4. Immutable sort results
First implementation of `SORT_RESULTS` mutated `context.results` in-place. Discovered during testing that `sort by score` then failed to restore original order. Switched to `sorted(self.context.results, ...)` (returns new list) — original never touched.

---

## Mistakes and Corrections

| Mistake | Impact | Correction |
|---|---|---|
| Max Drawdown O(n²) nested loop | 23 years × 252 days = 5,796 comparisons per stock | Rewrote with `cummax()` |
| `Σ wᵢ` floating-point drift | Allocations summed to 0.9999...8 | Last element absorbs remainder |
| `worst by volatility` → asc by default | Returns safest stock instead of riskiest | Read `higher_is_better` from `METRIC_REGISTRY` |
| `context.results` mutated on sort | `sort by score` couldn't restore original order | Sort produces copy, original untouched |
| Test contamination: MetricCache writing to production cache dir | Screener tests corrupted the real cache | Patched `MetricCache.__init__` in all tests |
| Missing metric aliases for `"risk"` in screener | `"top 10 BSE by risk"` returned default CAGR instead | Added `"risk": "volatility"` to alias table |

---

## What Changed During Development and Why

| Change | Reason |
|---|---|
| Added `METRIC_REGISTRY` to `constants.py` | Single source for display name, unit, scale, and `higher_is_better` — prevents divergence |
| `_ordinal()` helper in `screener_engine.py` | Rank labels needed in multiple output paths; centralised avoids duplication |
| `_POSITION_RE` checked before `_SCREENER_RE` | `"best"` must not accidentally fire `SCREEN_TOP`; explicit ordering prevents silent wrong routing |
| `SORT_RESULTS` added to `COMMAND_MAP` before positional regex | Multi-word phrases (`"sort by"`) must win before single-word `"sort"` is matched elsewhere |
| `follow_up()` message updated | Users were unaware of `sort by <field>` commands; hint was added to analysis results footer |
| `help / keywords` card expanded | New screener and sort commands were not appearing in help |

---

## Test Architecture

Tests are layered to mirror the module hierarchy:

```
tests/
    test_metrics.py            → MetricsEngine unit tests
    test_scoring_engine.py     → ScoringEngine + normalisation invariants
    test_portfolio.py          → PortfolioEngine allocation + Monte Carlo
    test_metric_cache.py       → Cache lifecycle, fingerprinting, corruption recovery
    test_screener.py           → ScreenerEngine (8 test classes):
                                   TestScreenerIntent          — intent routing
                                   TestScreenerParams          — param extraction
                                   TestMetricsEngine           — compute_metric dispatch
                                   TestScreenerEngine          — rank fields, metric/score modes
                                   TestScreenerEngineRankFields — rank/rank_label
                                   TestFormatScreenerResults   — output formatting
                                   TestPositionalQueries       — fetch_position, SCREEN_POSITION
                                   TestSortResults             — SORT_RESULTS, format_sorted_table
    test_parser_stress.py      → Parser robustness (mixed case, noisy input, edge boundary)
    test_conversation_manager.py → Full end-to-end turn integration
    test_explanation_engine.py → ExplanationEngine output contracts
    test_discoverability.py    → list / search / exchanges commands
```

**Total: 443 tests.  All pass in < 3 seconds.**

Key testing philosophy:
- All disk I/O is mocked — no CSV reads in any unit test
- `MetricCache` is always patched in ConversationManager tests
- Mathematical invariants (`Σ wᵢ = 1.0`, `CVaR ≤ VaR`) are explicitly asserted with tight tolerances
- Parser stress tests verify resilience to uppercase, filler words, and edge spacing
- API tests verify correct weight parsing from query strings and POST bodies

| test file | coverage |
|---|---|
| `test_scoring_engine.py` | Weighted scoring, Sharpe weight impact, zero-weight guard, normalisation invariants |
| `test_django_api.py` | `api_ranking` weight parsing (`weight_*` query params → dict) |

---

### Phase 11 — Weighted Metrics, Ranking Explanations & Portfolio Weights

#### 11.1 — Custom Weighted Ranking

Users asked to combine multiple metrics rather than choosing just one. The ScoringEngine already existed for the analysis session but was not exposed through the Ranking API.

The solution was threefold:

1. **Schema extension**: Added `weights: dict[str, float] | None` to `RankingSelection` so the Pydantic contract holds the user's intent.
2. **API parsing**: Weights arrive as URL query params prefixed with `weight_` (e.g. `weight_return=0.7`). `views.py` loops over `request.GET` items and builds the dict. Invalid float values are silently skipped.
3. **ScoringEngine reuse**: `ScreenerEngine` already called `ScoringEngine.compute_weighted_scores()` for the `"score"` metric. Passing the user-provided weights dict there was a one-line change — normalisation, risk-profile adjustment, and min-max logic all worked without modification.

The "Score" metric was renamed to **"Custom"** in `constants.METRIC_REGISTRY` to communicate that weights are user-defined. No backend logic changed — only the display string.

Input validation edge case: HTML `<input type="number" step="0.1">` rejects `0.34` because it is not a multiple of 0.1. Changing to `step="0.01"` permits two decimal places without any backend changes.

#### 11.2 — RankingExplanationEngine

After ranking results were returned, there was no interpretive text explaining *why* those stocks appear at the top. The `AllocationExplanationEngine` pattern was replicated exactly:

- Fully stateless (`@staticmethod` only)
- Does not compute metrics — only interprets existing results
- Produces six sections: `summary`, `methodology`, `top_stocks`, `metric_insight`, `weights_used`, `final_statement`
- `weights_used` section is only populated when metric is `"score"` (Custom)

The engine is called inside `RankingService.build_payload()` wrapped in a `try/except`. If it fails, `explanation=None` is returned gracefully — same fail-safe pattern as portfolio.

The UI panel uses a cyan color theme to visually distinguish it from the Portfolio (indigo) and Risk (rose) panels.

#### 11.3 — Custom Scoring Weights for Portfolio

The portfolio page used `DEFAULT_SCREENER_WEIGHTS` hardcoded in `PortfolioService`. The request was to expose the same six weight sliders on the portfolio page.

Key design choice: **weights only affect step 1 (scoring order)**, not the allocation method. The allocation (proportional/softmax/risk-adjusted) still operates on the *output* of the scoring step. Users control which stocks score highly; the method then distributes capital across those scores.

The frontend only sends weights when the Custom Weights panel is **open**. When closed, no `weight_*` keys appear in the POST body and the service falls back to defaults — no boolean flag needed.

#### 11.4 — UX Polish

Three smaller improvements shipped together:

| Change | Detail |
|---|---|
| N parameter hard caps | NSE ≤ 1,970 · BSE ≤ 4,465. The `max` attribute on the N input updates dynamically when exchange changes. A validation error blocks submission if exceeded. |
| Loading animation | Replaced plain text with an SVG ring animation labeled "Crunching market data..." |
| Toggle button styling | Custom Weights toggle upgraded from a text link to a bordered secondary button with `+` / `×` icon swap on open/close. |

---
