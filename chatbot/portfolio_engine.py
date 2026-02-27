"""
chatbot/portfolio_engine.py
---------------------------
Pure transformation engine: scored stock results → portfolio allocations.

Design contract:
  - No metric computation
  - No score computation
  - No DataLoader dependency
  - No FSM awareness
  - Fully deterministic and stateless (all methods are @staticmethod)
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Dict, Optional

from chatbot.config import RISK_FREE_RATE


class PortfolioEngine:
    """
    Convert a ranked list of scored stocks into weighted portfolio allocations.

    Supports three allocation strategies:
        ``"proportional"``  – weight ∝ total_score  (default, most transparent)
        ``"softmax"``       – softmax over scores   (smoother preference boost)
        ``"risk_adjusted"`` – score adjusted by volatility given risk profile

    Optional constraints (passed via *config*):
        ``max_cap``   – maximum allocation for any single stock (e.g. 0.60)
        ``min_floor`` – minimum allocation for any single stock (e.g. 0.05)

    All allocations are in [0, 1] and sum to exactly 1.
    """

    DEFAULT_METHOD: str = "proportional"
    MAX_CAP: float = 0.60     # class-level default cap (unused unless in config)
    MIN_FLOOR: float = 0.05   # class-level default floor (unused unless in config)

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #

    @staticmethod
    def allocate(
        scored_results: List[Dict],
        method: str = "proportional",
        risk_profile: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Allocate portfolio weights across *scored_results*.

        Parameters
        ----------
        scored_results:
            List of dicts, each **must** contain ``"ticker"`` and
            ``"total_score"``.  Optional keys: ``"volatility"``, ``"cagr"``.
        method:
            One of ``"proportional"``, ``"softmax"``, ``"risk_adjusted"``.
        risk_profile:
            ``"LOW"`` / ``"MEDIUM"`` / ``"HIGH"`` — used only by
            ``"risk_adjusted"`` method.
        config:
            Optional dict with constraint keys ``"max_cap"`` and/or
            ``"min_floor"``.

        Returns
        -------
        List of dicts with keys ``"ticker"``, ``"allocation"``,
        ``"total_score"``.
        """
        if not scored_results:
            return []

        if len(scored_results) == 1:
            return PortfolioEngine._single_stock(scored_results[0])

        # Shallow-copy each dict to avoid mutating the caller's list
        results = [dict(stock) for stock in scored_results]

        scores = PortfolioEngine._extract_scores(results)

        if method == "proportional":
            weights = PortfolioEngine._proportional(scores)
        elif method == "softmax":
            weights = PortfolioEngine._softmax(scores)
        elif method == "risk_adjusted":
            weights = PortfolioEngine._risk_adjusted(results, risk_profile)
        else:
            raise ValueError(
                f"Unknown allocation method: {method!r}. "
                "Choose from 'proportional', 'softmax', 'risk_adjusted'."
            )

        # Optional caps / floors
        weights = PortfolioEngine._apply_constraints(weights, config)

        # Re-normalise after constraints shift values
        weights = PortfolioEngine._normalize(weights)

        return PortfolioEngine._attach_allocations(results, weights)

    @staticmethod
    def compute_risk_decomposition(allocations: List[Dict]) -> List[Dict]:
        """
        Computes each stock's contribution to portfolio risk.
        Uses simplified independent-volatility model.

        Adds ``"risk_contribution"`` and ``"risk_share"`` to each dict.
        """
        total_risk = sum(
            a["allocation"] * a.get("volatility", 0.0)
            for a in allocations
        )

        if total_risk == 0:
            return allocations

        for a in allocations:
            contribution = a["allocation"] * a.get("volatility", 0.0)
            a["risk_contribution"] = contribution
            a["risk_share"] = contribution / total_risk

        return allocations

    @staticmethod
    def allocate_capital(
        allocations: List[Dict],
        total_capital: float
    ) -> List[Dict]:
        """
        Calculates the capital amount for each stock based on allocation weight.

        Adds ``"capital_amount"`` to each dict.

        Uses **remainder absorption** — all stocks except the last one are
        rounded to 2 dp; the final stock receives ``total_capital - sum_of_rest``
        so that the capital amounts always sum exactly to *total_capital*,
        regardless of floating-point rounding.
        """
        if not allocations:
            return allocations

        # Safety guard: allocations must sum to 1 before we distribute capital
        alloc_total = sum(a["allocation"] for a in allocations)
        assert abs(alloc_total - 1.0) < 1e-9, (
            f"Allocations must sum to 1.0 before distributing capital "
            f"(got {alloc_total})."
        )

        distributed = 0.0
        for a in allocations[:-1]:
            amt = round(total_capital * a["allocation"], 2)
            a["capital_amount"] = amt
            distributed += amt

        # Last stock absorbs the rounding residual — guarantees exact budget match
        allocations[-1]["capital_amount"] = round(total_capital - distributed, 2)

        return allocations

    @staticmethod
    def compute_portfolio_sharpe(allocations: List[Dict]) -> float:
        """
        Compute the portfolio-level Sharpe Ratio using a weighted return
        and the **independent-stock** (uncorrelated) volatility assumption.

        Formula
        -------
        ::

            E[Rp] = sum(wi * CAGRi)          # weighted portfolio return
            σp    = sqrt(sum(wi**2 * σi**2)) # uncorrelated portfolio vol
            Sharpe = (E[Rp] - Rf) / σp

        Parameters
        ----------
        allocations : List[Dict]
            Output from :meth:`PortfolioEngine.allocate`.  Each dict must
            contain:

            * ``"allocation"``  — weight wi (floats, must sum to 1.0)
            * ``"cagr"``        — annualised CAGR as a decimal
            * ``"volatility"``  — annualised volatility as a decimal

        Returns
        -------
        float
            Portfolio Sharpe Ratio. Returns ``0.0`` when portfolio
            volatility is effectively zero to avoid division-by-zero.

        Raises
        ------
        AssertionError
            If weights do not sum to 1.0 within 1e-9 tolerance.

        Notes
        -----
        Stage 1 assumes zero correlation between stocks, which slightly
        **underestimates** true portfolio volatility (since positive
        co-movements are ignored). The result is therefore an **upper bound**
        on true portfolio Sharpe when stocks are positively correlated.
        Stage 2 (covariance-aware) is a future upgrade.
        """
        if not allocations:
            return 0.0

        alloc_sum = sum(a["allocation"] for a in allocations)
        assert abs(alloc_sum - 1.0) < 1e-9, (
            f"Allocations must sum to 1.0 "
            f"(got {alloc_sum}). Run PortfolioEngine.allocate() first."
        )

        portfolio_return = sum(
            a["allocation"] * a.get("cagr", 0.0)
            for a in allocations
        )

        # Independent-stock variance: σp² = Σ(wi² · σi²)
        variance = sum(
            (a["allocation"] ** 2) * (a.get("volatility", 0.0) ** 2)
            for a in allocations
        )
        portfolio_vol = math.sqrt(variance)

        if portfolio_vol == 0.0:
            return 0.0

        return (portfolio_return - RISK_FREE_RATE) / portfolio_vol

    @staticmethod
    def portfolio_summary(allocations: List[Dict]) -> dict:
        """
        Compute and return a structured portfolio-level summary.

        Returns
        -------
        dict
            ``portfolio_return``    — weighted CAGR (decimal)
            ``portfolio_volatility`` — independent-assumption vol (decimal)
            ``portfolio_sharpe``    — portfolio Sharpe Ratio
            ``allocations``         — original allocation list (pass-through)
        """
        if not allocations:
            return {
                "portfolio_return":     0.0,
                "portfolio_volatility": 0.0,
                "portfolio_sharpe":     0.0,
                "allocations":          [],
            }

        port_return = sum(
            a["allocation"] * a.get("cagr", 0.0) for a in allocations
        )
        variance = sum(
            (a["allocation"] ** 2) * (a.get("volatility", 0.0) ** 2)
            for a in allocations
        )
        port_vol   = math.sqrt(variance)
        port_sharpe = PortfolioEngine.compute_portfolio_sharpe(allocations)

        return {
            "portfolio_return":     port_return,
            "portfolio_volatility": port_vol,
            "portfolio_sharpe":     port_sharpe,
            "portfolio_mdd":        PortfolioEngine.compute_portfolio_mdd(allocations),
            "portfolio_sortino":    PortfolioEngine.compute_portfolio_sortino(allocations),
            "allocations":          allocations,
        }

    @staticmethod
    def compute_portfolio_mdd(allocations: List[Dict]) -> float:
        """
        Compute portfolio Maximum Drawdown using a weighted daily-return
        reconstruction of the portfolio price index.

        Requires each allocation dict to carry a ``"daily_returns"``
        key (a ``pd.Series`` of daily fractional returns for the horizon
        window).  When that key is absent, falls back to a first-order
        approximation using the per-stock MDD weighted by allocation.

        Parameters
        ----------
        allocations : List[Dict]
            Each dict must contain ``"allocation"`` (weight) and either

            * ``"daily_returns"`` (pd.Series) — preferred, or
            * ``"max_drawdown"``  (float)     — fallback approximation.

        Returns
        -------
        float
            MDD as a positive decimal in [0, 1]. Lower is better.
            Returns ``0.0`` when neither source is available.
        """
        if not allocations:
            return 0.0

        # Preferred path: reconstruct portfolio return series
        if "daily_returns" in allocations[0]:
            try:
                portfolio_returns = sum(
                    a["allocation"] * a["daily_returns"] for a in allocations
                )
                portfolio_index = (1 + portfolio_returns).cumprod()
                running_peak    = portfolio_index.cummax()
                drawdowns       = (running_peak - portfolio_index) / running_peak
                return float(drawdowns.max())
            except Exception:
                pass  # fall through to approximation

        # Fallback: weighted average of per-stock MDD values
        if "max_drawdown" in allocations[0]:
            return float(sum(
                a["allocation"] * a.get("max_drawdown", 0.0)
                for a in allocations
            ))

        return 0.0

    @staticmethod
    def compute_portfolio_sortino(allocations: List[Dict]) -> float:
        """
        Compute portfolio-level Sortino Ratio.

        Uses a weighted portfolio return series to measure only downside
        deviation (below the daily risk-free hurdle), then applies the
        standard Sortino formula on the annualised values.

        Parameters
        ----------
        allocations : List[Dict]
            Each dict must contain ``"allocation"`` (weight) and either

            * ``"daily_returns"`` (pd.Series) — preferred, or
            * ``"cagr"`` + ``"volatility"`` — falls back to Stage-1 approximation.

        Returns
        -------
        float
            Sortino Ratio. Returns ``0.0`` when no data is available.
        """
        if not allocations:
            return 0.0

        daily_rf = RISK_FREE_RATE / 252

        # Preferred path: simulate portfolio daily returns
        if "daily_returns" in allocations[0]:
            try:
                portfolio_returns = sum(
                    a["allocation"] * a["daily_returns"] for a in allocations
                )
                cagr_approx = float((1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1)
                downside    = portfolio_returns[portfolio_returns < daily_rf]
                if len(downside) == 0:
                    return 0.0
                downside_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252))
                if downside_std == 0.0:
                    return 0.0
                return (cagr_approx - RISK_FREE_RATE) / downside_std
            except Exception:
                pass

        # Fallback: weighted-CAGR / independent-downside approximation
        # (assumes Sharpe ≈ Sortino for symmetric returns; coarse but valid)
        return PortfolioEngine.compute_portfolio_sharpe(allocations)

    # ------------------------------------------------------------------ #
    #  Stage 2 — Covariance-aware volatility  (σp² = wᵀΣw)
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_portfolio_volatility_covariance(
        allocations: List[Dict],
        covariance_matrix: List[List[float]],
    ) -> float:
        """
        Compute portfolio volatility using the full covariance matrix.

        Formula::

            σp² = wᵀ Σ w = Σᵢ Σⱼ wᵢ wⱼ σᵢⱼ
            σp  = √(σp²)

        Unlike Stage 1 (independent-stock assumption), this correctly captures
        the correlation structure between assets.  Portfolio volatility will be
        **lower** than Stage 1 when stocks are positively correlated because the
        covariance terms are already baked into the off-diagonal elements of Σ.

        .. note::
            Result quality depends on the stability of the input covariance
            matrix.  Estimates from short sample windows (< 3 years) may be
            numerically unstable; shrinkage estimators are recommended for
            production use.

        Parameters
        ----------
        allocations : List[Dict]
            Each dict must contain ``"allocation"`` (float).
            Weights must sum to 1.0 within 1e-9.
        covariance_matrix : List[List[float]]
            N×N annualised covariance matrix in allocation order.
            ``cov[i][j]`` = annualised covariance between stock i and stock j.
            Diagonal elements are the per-stock annualised variances (σᵢ²).

        Returns
        -------
        float
            Portfolio volatility (standard deviation) as a positive decimal.
            Returns ``0.0`` for an empty allocation list.

        Raises
        ------
        AssertionError
            * If the covariance matrix row count ≠ number of allocations.
            * If any row of the matrix has the wrong length (non-square).
            * If weights do not sum to 1.0 within 1e-9.
        """
        if not allocations:
            return 0.0

        n = len(allocations)

        if len(covariance_matrix) != n:
            raise AssertionError(
                f"Covariance matrix has {len(covariance_matrix)} rows "
                f"but {n} allocations were provided."
            )
        for row_idx, row in enumerate(covariance_matrix):
            if len(row) != n:
                raise AssertionError(
                    f"Covariance matrix must be square N×N. "
                    f"Row {row_idx} has {len(row)} columns, expected {n}."
                )

        weights     = [a["allocation"] for a in allocations]
        weight_sum  = sum(weights)
        if abs(weight_sum - 1.0) > 1e-9:
            raise AssertionError(
                f"Allocations must sum to 1.0 (got {weight_sum}). "
                "Run PortfolioEngine.allocate() first."
            )

        # σp² = wᵀ Σ w  (pure Python — no numpy dependency)
        portfolio_variance = 0.0
        for i in range(n):
            for j in range(n):
                portfolio_variance += weights[i] * weights[j] * covariance_matrix[i][j]

        # Numerical guard: floating-point errors can produce tiny negatives
        if portfolio_variance < 0.0:
            portfolio_variance = 0.0

        return math.sqrt(portfolio_variance)

    @staticmethod
    def compute_portfolio_sharpe_covariance(
        allocations: List[Dict],
        covariance_matrix: List[List[float]],
    ) -> float:
        """
        Compute the portfolio Sharpe Ratio using covariance-aware volatility.

        Formula::

            Rp     = Σ wᵢ · expected_returnᵢ
            σp     = compute_portfolio_volatility_covariance(w, Σ)
            Sharpe = (Rp − Rf) / σp

        Parameters
        ----------
        allocations : List[Dict]
            Each dict must contain:

            * ``"allocation"``      — weight wᵢ (must sum to 1.0)
            * ``"expected_return"`` — annualised expected return (decimal)

        covariance_matrix : List[List[float]]
            N×N annualised covariance matrix.

        Returns
        -------
        float
            Covariance-aware Sharpe Ratio.
            Returns ``0.0`` when portfolio volatility is zero.
        """
        if not allocations:
            return 0.0

        portfolio_return = sum(
            a["allocation"] * a.get("expected_return", 0.0)
            for a in allocations
        )

        volatility = PortfolioEngine.compute_portfolio_volatility_covariance(
            allocations, covariance_matrix
        )

        if volatility == 0.0:
            return 0.0

        return (portfolio_return - RISK_FREE_RATE) / volatility

    @staticmethod
    def portfolio_summary_covariance(
        allocations: List[Dict],
        covariance_matrix: List[List[float]],
    ) -> dict:
        """
        Structured portfolio summary using the full covariance matrix (Stage 2).

        Returns
        -------
        dict
            ``portfolio_return``    — weighted expected return (decimal)
            ``portfolio_volatility`` — covariance-aware portfolio volatility
            ``portfolio_sharpe``    — covariance-aware Sharpe Ratio
            ``model``               — ``"covariance-aware"`` (Stage 2 label)
            ``allocations``         — input allocation list (pass-through)
        """
        if not allocations:
            return {
                "portfolio_return":     0.0,
                "portfolio_volatility": 0.0,
                "portfolio_sharpe":     0.0,
                "model":                "covariance-aware",
                "allocations":          [],
            }

        portfolio_return = sum(
            a["allocation"] * a.get("expected_return", 0.0)
            for a in allocations
        )
        volatility = PortfolioEngine.compute_portfolio_volatility_covariance(
            allocations, covariance_matrix
        )
        sharpe = (portfolio_return - RISK_FREE_RATE) / volatility if volatility > 0.0 else 0.0

        return {
            "portfolio_return":     portfolio_return,
            "portfolio_volatility": volatility,
            "portfolio_sharpe":     sharpe,
            "model":                "covariance-aware",
            "allocations":          allocations,
        }

    # ------------------------------------------------------------------ #
    #  Stage 4 — Monte Carlo simulation  (multivariate normal via Cholesky)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cholesky_decomposition(matrix: List[List[float]]) -> List[List[float]]:
        """
        Compute the lower-triangular Cholesky factor L of a symmetric
        positive-definite matrix such that  A = L Lᵀ.

        Pure-Python implementation — no numpy dependency.

        Parameters
        ----------
        matrix : List[List[float]]
            N×N symmetric positive-definite matrix (e.g. a covariance matrix).

        Returns
        -------
        List[List[float]]
            Lower-triangular Cholesky factor L (N×N).

        Raises
        ------
        ValueError
            If the matrix is not positive definite (encountered a non-positive
            value on the diagonal during decomposition).
        """
        n = len(matrix)
        L = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                sum_val = sum(L[i][k] * L[j][k] for k in range(j))

                if i == j:
                    diag = matrix[i][i] - sum_val
                    if diag <= 0.0:
                        raise ValueError(
                            f"Covariance matrix is not positive definite "
                            f"(non-positive diagonal at element [{i},{i}])."
                        )
                    L[i][j] = math.sqrt(diag)
                else:
                    L[i][j] = (matrix[i][j] - sum_val) / L[j][j]

        return L

    @staticmethod
    def simulate_portfolio_monte_carlo(
        allocations: List[Dict],
        covariance_matrix: List[List[float]],
        num_simulations: int = 10_000,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Monte Carlo simulation of portfolio returns using a multivariate
        normal distribution parameterised by the asset expected returns
        and the full covariance matrix.

        Algorithm
        ---------
        1. Decompose Σ via Cholesky: Σ = LLᵀ
        2. For each trial:
           a. Draw z ~ N(0, I_n)
           b. Compute correlated shocks: c = L · z
           c. Asset returns: r_i = μ_i + c_i
           d. Portfolio return: Rp = wᵀ r
        3. Aggregate: mean, std, VaR₉₅, CVaR₉₅, P(loss)

        .. note::
            Returns are interpreted as *annual* if μ and Σ are annualised.
            For an *M*-month horizon, scale μ by M/12 and Σ by M/12 before
            calling this method.

        Parameters
        ----------
        allocations : List[Dict]
            Each dict must contain:
            * ``"allocation"``      — weight wᵢ (must sum to 1.0)
            * ``"expected_return"`` — annualised expected return (decimal)
        covariance_matrix : List[List[float]]
            N×N annualised covariance matrix.
        num_simulations : int
            Number of Monte Carlo draws.  Default 10,000.
        seed : int, optional
            Random seed for reproducibility.  ``None`` = non-deterministic.

        Returns
        -------
        dict
            ``mean_return``        — mean simulated portfolio return
            ``std_dev``            — std of simulated portfolio returns
            ``var_95``             — 95% Value at Risk (5th percentile)
            ``cvar_95``            — 95% CVaR / Expected Shortfall (mean of worst 5%)
            ``probability_of_loss`` — fraction of trials yielding negative return
            ``simulated_returns``  — full list of simulated return values (length = num_simulations)

        Raises
        ------
        AssertionError
            If weights do not sum to 1.0 within 1e-9.
        ValueError
            If the covariance matrix is not positive definite.
        """
        if not allocations:
            return {}

        import random as _random
        if seed is not None:
            _random.seed(seed)

        n       = len(allocations)
        weights = [a["allocation"]      for a in allocations]
        means   = [a.get("expected_return", 0.0) for a in allocations]

        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-9:
            raise AssertionError(
                f"Allocations must sum to 1.0 (got {weight_sum}). "
                "Run PortfolioEngine.allocate() first."
            )

        L = PortfolioEngine._cholesky_decomposition(covariance_matrix)

        simulated_returns: List[float] = []

        for _ in range(num_simulations):
            # Independent standard normals
            z = [_random.gauss(0.0, 1.0) for _ in range(n)]

            # Correlate via Cholesky: c = L · z
            correlated = [
                sum(L[i][k] * z[k] for k in range(i + 1))
                for i in range(n)
            ]

            # Shift by expected returns
            asset_returns = [means[i] + correlated[i] for i in range(n)]

            # Portfolio return for this trial
            simulated_returns.append(
                sum(weights[i] * asset_returns[i] for i in range(n))
            )

        # ---- Statistics ------------------------------------------------ #
        mean_return = sum(simulated_returns) / num_simulations

        variance = sum(
            (r - mean_return) ** 2 for r in simulated_returns
        ) / num_simulations          # population variance (matches CLT convention)

        std_dev = math.sqrt(variance)

        sorted_returns = sorted(simulated_returns)

        # 95% VaR: 5th-percentile (worst 5% cutoff)
        var_idx = max(int(0.05 * num_simulations), 1)
        var_95  = sorted_returns[var_idx - 1]   # 0-indexed, last element of worst 5%

        # 95% CVaR (Expected Shortfall): mean of the worst 5%
        cvar_95 = sum(sorted_returns[:var_idx]) / var_idx

        probability_of_loss = sum(1 for r in simulated_returns if r < 0.0) / num_simulations

        return {
            "mean_return":          mean_return,
            "std_dev":              std_dev,
            "var_95":               var_95,
            "cvar_95":              cvar_95,
            "probability_of_loss":  probability_of_loss,
            "simulated_returns":    simulated_returns,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_scores(results: List[Dict]) -> List[float]:
        """Pull ``total_score`` from each result dict; raise if missing."""
        scores = []
        for stock in results:
            score = stock.get("total_score")
            if score is None:
                raise ValueError(
                    f"Missing 'total_score' for ticker {stock.get('ticker', '?')!r}."
                )
            scores.append(float(score))
        return scores

    @staticmethod
    def _proportional(scores: List[float]) -> List[float]:
        """
        Weight each stock proportionally to its score.

        Falls back to equal weighting when all scores are zero.
        """
        total = sum(scores)
        if total == 0:
            return [1 / len(scores)] * len(scores)
        return [s / total for s in scores]

    @staticmethod
    def _softmax(scores: List[float]) -> List[float]:
        """
        Softmax weighting — boosts preference for higher-scored stocks
        while keeping all weights strictly positive.

        Uses the numerically stable max-shift variant.
        """
        max_s = max(scores)
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    @staticmethod
    def _risk_adjusted(
        results: List[Dict],
        risk_profile: Optional[str],
    ) -> List[float]:
        """
        Adjust scores by volatility according to the investor's risk profile.

        - ``"LOW"`` / ``"CONSERVATIVE"``  → penalise high-volatility stocks.
        - ``"HIGH"`` / ``"AGGRESSIVE"``   → reward high-volatility stocks.
        - Anything else (``None``, ``"MEDIUM"``)  → use raw score.
        """
        adjusted_scores: List[float] = []
        rp = (risk_profile or "").upper()

        for stock in results:
            score = float(stock["total_score"])
            volatility = float(stock.get("volatility", 0.0))

            if rp in ("LOW", "CONSERVATIVE"):
                # Divide by volatility+ε to penalise risky stocks
                adjusted = score / (volatility + 1e-6)
            elif rp in ("HIGH", "AGGRESSIVE"):
                # Multiply by (1+vol) to reward growth potential
                adjusted = score * (1 + volatility)
            else:
                adjusted = score

            adjusted_scores.append(adjusted)

        return PortfolioEngine._normalize(adjusted_scores)

    @staticmethod
    def _apply_constraints(
        weights: List[float],
        config: Optional[Dict],
    ) -> List[float]:
        """
        Apply per-stock cap and/or floor constraints via iterative redistribution.

        After each clipping pass any excess (cap) or deficit (floor) is
        re-distributed proportionally to the unconstrained stocks, and the
        loop repeats until the weights stabilise.  This guarantees the
        invariants hold exactly (no drift from the subsequent _normalize step).
        """
        if not config:
            return weights

        max_cap   = config.get("max_cap")
        min_floor = config.get("min_floor")

        # Work in-place on a copy; make sure they already sum to 1
        w = PortfolioEngine._normalize(list(weights))

        for _ in range(len(w) * 2):          # at most O(n) passes needed
            changed = False

            # --- Cap pass ------------------------------------------------
            if max_cap is not None:
                excess = 0.0
                free_total = 0.0
                for i, v in enumerate(w):
                    if v > max_cap + 1e-10:
                        excess += v - max_cap
                        w[i]   = max_cap
                        changed = True
                    elif v < max_cap - 1e-10:
                        free_total += v
                if excess > 0 and free_total > 0:
                    scale = 1 + excess / free_total
                    for i in range(len(w)):
                        if w[i] < max_cap - 1e-10:
                            w[i] *= scale

            # --- Floor pass ----------------------------------------------
            if min_floor is not None:
                deficit = 0.0
                free_total = 0.0
                for i, v in enumerate(w):
                    if v < min_floor - 1e-10:
                        deficit += min_floor - v
                        w[i]    = min_floor
                        changed = True
                    elif v > min_floor + 1e-10:
                        free_total += v
                if deficit > 0 and free_total > 0:
                    scale = 1 - deficit / free_total
                    for i in range(len(w)):
                        if w[i] > min_floor + 1e-10:
                            w[i] *= scale

            if not changed:
                break

        return w

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        """Scale *values* so they sum to 1.0. Falls back to equal weight."""
        total = sum(values)
        if total == 0:
            return [1 / len(values)] * len(values)
        return [v / total for v in values]

    @staticmethod
    def _attach_allocations(
        results: List[Dict],
        weights: List[float],
    ) -> List[Dict]:
        """Zip allocation weights back onto the result dicts."""
        output = []
        for stock, weight in zip(results, weights):
            output.append({
                "ticker":      stock["ticker"],
                "allocation":  round(weight, 4),
                "total_score": stock["total_score"],
            })
        return output

    @staticmethod
    def _single_stock(stock: Dict) -> List[Dict]:
        """Trivial case: one stock gets 100 % of the allocation."""
        return [{
            "ticker":      stock["ticker"],
            "allocation":  1.0,
            "total_score": stock["total_score"],
        }]
