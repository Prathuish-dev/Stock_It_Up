(() => {
    const form = document.getElementById("portfolio-form");
    if (!form) return;

    const runBtn = document.getElementById("portfolio-run-btn");
    const btnLabel = runBtn ? runBtn.querySelector(".btn-label") : null;
    const spinner = runBtn ? runBtn.querySelector(".spinner") : null;
    const errorBox = document.getElementById("portfolio-error");
    const warningBox = document.getElementById("portfolio-warning");
    const summarySection = document.getElementById("portfolio-summary");
    const resultsSection = document.getElementById("portfolio-results");
    const chartsSection = document.getElementById("portfolio-charts");
    const explanationSection = document.getElementById("portfolio-explanation");
    const body = document.getElementById("portfolio-body");

    const sumReturn = document.getElementById("sum-return");
    const sumVol = document.getElementById("sum-vol");
    const sumSharpe = document.getElementById("sum-sharpe");
    const sumLoss = document.getElementById("sum-loss");

    const charts = { pie: null, capital: null, scatter: null };
    const tickerPicker =
        typeof window.createTickerPicker === "function"
            ? window.createTickerPicker({
                exchangeSelectId: "portfolio-exchange",
                hiddenInputId: "portfolio-tickers-hidden",
                searchInputId: "portfolio-ticker-search",
                selectedContainerId: "portfolio-ticker-selected",
                suggestionsContainerId: "portfolio-ticker-suggestions",
                maxItems: 20,
            })
            : null;

    function setLoading(isLoading) {
        if (!runBtn) return;
        runBtn.disabled = isLoading;
        if (btnLabel) btnLabel.textContent = isLoading ? "Analyzing..." : "Analyze Portfolio";
        if (spinner) spinner.classList.toggle("hidden", !isLoading);
    }

    function showBox(node, text) {
        if (!node) return;
        if (text) {
            node.textContent = text;
            node.classList.remove("hidden");
        } else {
            node.textContent = "";
            node.classList.add("hidden");
        }
    }

    function toPct(value) {
        return `${(Number(value || 0) * 100).toFixed(2)}%`;
    }

    function destroyCharts() {
        if (charts.pie) charts.pie.destroy();
        if (charts.capital) charts.capital.destroy();
        if (charts.scatter) charts.scatter.destroy();
        charts.pie = null;
        charts.capital = null;
        charts.scatter = null;
    }

    function renderCharts(chartData) {
        destroyCharts();
        const pieCanvas = document.getElementById("portfolioAllocationChart");
        const capitalCanvas = document.getElementById("portfolioCapitalChart");
        const scatterCanvas = document.getElementById("portfolioScatterChart");
        if (!pieCanvas || !capitalCanvas || !scatterCanvas) return;

        charts.pie = new Chart(pieCanvas, {
            type: "pie",
            data: {
                labels: chartData.allocation_labels || [],
                datasets: [{ data: chartData.allocation_values || [] }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });

        charts.capital = new Chart(capitalCanvas, {
            type: "bar",
            data: {
                labels: chartData.capital_labels || [],
                datasets: [{ label: "Capital (INR)", data: chartData.capital_values || [], backgroundColor: "#0ea5e9" }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });

        charts.scatter = new Chart(scatterCanvas, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Stocks",
                        data: chartData.scatter_points || [],
                        backgroundColor: "#22d3ee",
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: "Volatility (%)" } },
                    y: { title: { display: true, text: "CAGR (%)" } },
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const p = ctx.raw || {};
                                return `${p.ticker}: ${p.y}% return, ${p.x}% risk`;
                            },
                        },
                    },
                },
            },
        });
    }

    function render(payload) {
        if (!payload || !payload.ok) {
            showBox(errorBox, payload && payload.error ? payload.error : "Unable to analyze portfolio.");
            return;
        }

        showBox(errorBox, payload.error);
        showBox(warningBox, payload.warning);

        sumReturn.textContent = toPct(payload.summary.portfolio_return);
        sumVol.textContent = toPct(payload.summary.portfolio_volatility);
        sumSharpe.textContent = Number(payload.summary.portfolio_sharpe || 0).toFixed(2);
        sumLoss.textContent = toPct(payload.summary.probability_of_loss);

        body.innerHTML = (payload.allocations || []).map((a) => `
            <tr class="border-t border-slate-700/60">
                <td class="px-3 py-3 font-semibold text-white">${a.ticker}</td>
                <td class="px-3 py-3">${(a.allocation * 100).toFixed(2)}%</td>
                <td class="px-3 py-3">₹${Number(a.capital_amount).toLocaleString()}</td>
                <td class="px-3 py-3">${Number(a.total_score).toFixed(4)}</td>
                <td class="px-3 py-3">${(a.cagr * 100).toFixed(2)}%</td>
                <td class="px-3 py-3">${(a.volatility * 100).toFixed(2)}%</td>
                <td class="px-3 py-3">${Number(a.sharpe).toFixed(2)}</td>
                <td class="px-3 py-3">${(a.risk_share * 100).toFixed(2)}%</td>
            </tr>
        `).join("");

        if (payload.explanation && explanationSection) {
            document.getElementById("expl-summary").textContent = payload.explanation.summary;
            document.getElementById("expl-rationale").textContent = payload.explanation.strategy_rationale;
            document.getElementById("expl-risk-dist").textContent = payload.explanation.risk_distribution;
            document.getElementById("expl-portfolio-risk").textContent = payload.explanation.portfolio_risk;
            document.getElementById("expl-final").textContent = payload.explanation.final_statement;
            explanationSection.classList.remove("hidden");
        } else if (explanationSection) {
            explanationSection.classList.add("hidden");
        }

        summarySection.classList.remove("hidden");
        resultsSection.classList.remove("hidden");
        chartsSection.classList.remove("hidden");
        renderCharts(payload.chart_data || {});
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        setLoading(true);
        showBox(errorBox, null);

        try {
            const data = new FormData(form);
            const payload = {
                exchange: String(data.get("exchange") || "NSE"),
                tickers: tickerPicker
                    ? tickerPicker.getTickers()
                    : String(data.get("tickers") || "").replace(/,/g, " ").split(/\s+/).filter(Boolean),
                budget: Number(data.get("budget") || 100000),
                method: String(data.get("method") || "proportional"),
                risk_profile: String(data.get("risk_profile") || "MEDIUM"),
                horizon_years: Number(data.get("horizon_years") || 3),
            };

            const res = await fetch("/api/portfolio", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Accept": "application/json" },
                body: JSON.stringify(payload),
            });
            const body = await res.json();
            render(body);
        } catch (error) {
            showBox(errorBox, "Unable to analyze portfolio right now.");
        } finally {
            setLoading(false);
        }
    });
})();
