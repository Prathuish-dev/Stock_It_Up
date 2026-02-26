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

import math
from typing import List, Dict, Optional


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
        """
        for a in allocations:
            a["capital_amount"] = round(total_capital * a["allocation"], 2)

        return allocations

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
