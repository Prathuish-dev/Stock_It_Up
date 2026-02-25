# 📈 Stock_It_Up

A modular, deterministic, explainable CLI-based financial decision assistant for equity evaluation using historical stock data (NSE & BSE).

---

# 1️⃣ Problem Understanding

The objective was to build a conversational system that:

* Allows users to explore stock data from NSE and BSE
* Computes performance metrics over a chosen time horizon
* Applies weighted scoring based on user preference
* Adapts to different risk profiles
* Produces ranked recommendations
* Clearly explains *why* a stock is recommended

Key constraints:

* Must not rely entirely on AI models
* Must use real dataset (2000–2023 historical CSVs)
* Must be deterministic and explainable
* Should follow modular, maintainable architecture

The system should act as a **Decision Companion**, not just a ranking calculator.

---

# 2️⃣ Assumptions Made

1. Historical daily data (2000–2023) is sufficient to compute:

   * CAGR
   * Volatility
   * Liquidity proxy (average volume)

2. Users compare stocks relative to each other, not against the entire market.

3. Scoring is **relative** (min–max normalization across selected stocks).

4. Risk profile modifies weight emphasis rather than altering raw metrics.

5. Directory structure reflects:

   ```
   comp_stock_data/
       stock_data_NSE/
       stock_data_BSE/
   ```

6. All CSVs are clean and consistent in schema.

---

# 3️⃣ Why This Structure Was Chosen

The system follows strict separation of concerns:

```
ConversationManager  → orchestration
IntentParser         → NLP/command routing
SessionContext       → user state
DataLoader           → filesystem access
MetricsEngine        → financial calculations
ScoringEngine        → weighted normalization logic
ExplanationEngine    → deterministic interpretation
ResponseGenerator    → formatting only
```

### Why?

* Prevent logic duplication
* Improve testability
* Enable deterministic behavior
* Allow future feature extension
* Keep explanation independent of scoring

This ensures each layer has a single responsibility.

---

# 4️⃣ Design Decisions & Trade-Offs

## ✅ Min–Max Normalization

Used for score scaling:

* Simple
* Interpretable
* Works well for ranking
* Easy to explain

Trade-off:

* Scores are relative to selected stocks.
* Adding/removing stocks changes normalized values.

---

## ✅ Risk Profile Adjusts Weights (Not Metrics)

LOW risk → more weight on stability
HIGH risk → more weight on return

Trade-off:

* Simpler and transparent
* Less mathematically complex than modifying return formula

---

## ✅ Deterministic Rule-Based Explanation

No LLM used for reasoning.

ExplanationEngine:

* Uses thresholds
* Uses ranking context
* Uses score gaps
* Uses qualitative labels

Trade-off:

* Less flexible than generative AI
* But fully auditable and testable

---

## ✅ Directory-Based Search

Search and listing use directory listing only.
No CSV read for browsing.

Trade-off:

* Faster performance
* Slightly dependent on naming consistency

---

## ✅ Global Command Overrides

Commands like:

* `list`
* `search`
* `help`
* `exchanges`

Work in any conversation state.

Trade-off:

* Slightly more routing logic
* Much better UX

---

# 5️⃣ Edge Cases Considered

✔ All stocks have identical metric values (max == min normalization)
✔ User provides weights that don’t sum to 1
✔ Single-stock comparison
✔ Searching partial ticker names
✔ Pagination beyond total pages
✔ No exchange selected before listing
✔ Rank 2 explanation when only 1 stock exists
✔ Risk profile switching mid-session
✔ Mid-analysis global commands

---

# 6️⃣ Features

## 📊 Core Analysis

* CAGR computation
* Volatility computation
* Liquidity proxy (average volume)
* Weighted scoring
* Risk-profile-aware weighting
* Ranked output
* Deterministic explanation (5 sections)

## 🧠 Explanation Engine

Outputs:

* Summary
* Numeric breakdown (raw → normalized → weight → contribution)
* Qualitative classification
* Comparative analysis
* Final recommendation statement

## 🔎 Market Exploration

Commands:

```
list
list NSE
list NSE page 3
search TCS
help
keywords
exchanges
markets
compare TCS INFY
```

## 📄 Pagination

* 50 companies per page
* Ceiling-division page calculation

---

# 7️⃣ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone <repo-url>
cd Stock_It_Up
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the CLI

```bash
python main.py
```

---

### 5️⃣ Run Tests

```bash
pytest
```

---

# 8️⃣ Example Flow

```
> list NSE
> search TCS
> compare TCS INFY
> analyze TCS INFY RELIANCE
> explain TCS
```

---

# 9️⃣ What I Would Improve With More Time

## 🔹 1. Add Additional Metrics

* Sharpe Ratio
* Maximum Drawdown
* Benchmark comparison

## 🔹 2. Portfolio Suggestion Mode

Instead of 1 winner:

* Suggest 2–3 allocations
* Score-weighted capital distribution

## 🔹 3. Percentile-Based Thresholds

Instead of fixed 0.75 / 0.40:

* Dynamic percentile classification
* Better scaling for large universes

## 🔹 4. Sector & Market Cap Awareness

Enable:

```
top IT stocks
low risk banking stocks
```

## 🔹 5. Web Interface

Convert CLI into:

* Streamlit dashboard
* REST API
* Or web-based UI

## 🔹 6. Performance Optimization

* Cache computed metrics
* Lazy compute only when needed
* Parallel metric computation

---

# 🔟 Final Summary

Stock_It_Up is:

* Deterministic
* Explainable
* Modular
* Fully test-covered
* CLI-driven
* Risk-aware
* Scalable

It transforms historical stock data into a structured decision-support system rather than a simple ranking script.

---
