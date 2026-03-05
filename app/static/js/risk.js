(() => {
    const form = document.getElementById("risk-form");
    if (!form) return;

    const runBtn = document.getElementById("risk-run-btn");
    const btnLabel = runBtn ? runBtn.querySelector(".btn-label") : null;
    const spinner = runBtn ? runBtn.querySelector(".spinner") : null;
    const errorBox = document.getElementById("risk-error");
    const warningBox = document.getElementById("risk-warning");
    const summarySection = document.getElementById("risk-summary");
    const tableSection = document.getElementById("risk-table-section");
    const chartsSection = document.getElementById("risk-charts");
    const explanationSection = document.getElementById("risk-explanation");
    const body = document.getElementById("risk-body");

    const riskVar = document.getElementById("risk-var");
    const riskCvar = document.getElementById("risk-cvar");
    const riskStd = document.getElementById("risk-std");
    const riskMean = document.getElementById("risk-mean");
    const riskLoss = document.getElementById("risk-loss");

    const charts = { histogram: null, contribution: null, scatter: null };
    const tickerPicker =
        typeof window.createTickerPicker === "function"
            ? window.createTickerPicker({
                exchangeSelectId: "risk-exchange",
                hiddenInputId: "risk-tickers-hidden",
                searchInputId: "risk-ticker-search",
                selectedContainerId: "risk-ticker-selected",
                suggestionsContainerId: "risk-ticker-suggestions",
                maxItems: 25,
            })
            : null;

    function setLoading(isLoading) {
        if (!runBtn) return;
        runBtn.disabled = isLoading;
        if (btnLabel) btnLabel.textContent = isLoading ? "Running..." : "Run Risk Analytics";
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
        if (charts.histogram) charts.histogram.destroy();
        if (charts.contribution) charts.contribution.destroy();
        if (charts.scatter) charts.scatter.destroy();
        charts.histogram = null;
        charts.contribution = null;
        charts.scatter = null;
    }

    function renderCharts(chartData) {
        destroyCharts();
        const histCanvas = document.getElementById("riskHistogramChart");
        const contribCanvas = document.getElementById("riskContributionChart");
        const scatterCanvas = document.getElementById("riskScatterChart");
        if (!histCanvas || !contribCanvas || !scatterCanvas) return;

        charts.histogram = new Chart(histCanvas, {
            type: "bar",
            data: {
                labels: chartData.histogram_labels || [],
                datasets: [{
                    label: "Probability (%)",
                    data: chartData.histogram_values || [],
                    backgroundColor: "#0ea5e9",
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { maxRotation: 70, minRotation: 70 } },
                    y: { title: { display: true, text: "Probability (%)" } },
                },
            },
        });

        charts.contribution = new Chart(contribCanvas, {
            type: "doughnut",
            data: {
                labels: chartData.risk_labels || [],
                datasets: [{ data: chartData.risk_values || [] }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });

        const assets = chartData.scatter_assets || [];
        const portfolio = chartData.scatter_portfolio || null;

        charts.scatter = new Chart(scatterCanvas, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Assets",
                        data: assets,
                        backgroundColor: "#22d3ee",
                        pointRadius: 4,
                    },
                    {
                        label: "Portfolio",
                        data: portfolio ? [portfolio] : [],
                        backgroundColor: "#f59e0b",
                        pointRadius: 7,
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
            showBox(errorBox, payload && payload.error ? payload.error : "Unable to run risk analytics.");
            return;
        }

        showBox(errorBox, payload.error);
        showBox(warningBox, payload.warning);

        riskVar.textContent = toPct(payload.summary.var_95);
        riskCvar.textContent = toPct(payload.summary.cvar_95);
        riskStd.textContent = toPct(payload.summary.std_dev);
        riskMean.textContent = toPct(payload.summary.mean_return);
        riskLoss.textContent = toPct(payload.summary.probability_of_loss);

        body.innerHTML = (payload.allocations || []).map((a) => `
            <tr class="border-t border-slate-700/60">
                <td class="px-3 py-3 font-semibold text-white">${a.ticker}</td>
                <td class="px-3 py-3">${(a.allocation * 100).toFixed(2)}%</td>
                <td class="px-3 py-3">${(a.risk_share * 100).toFixed(2)}%</td>
                <td class="px-3 py-3">${(a.cagr * 100).toFixed(2)}%</td>
                <td class="px-3 py-3">${(a.volatility * 100).toFixed(2)}%</td>
                <td class="px-3 py-3">${Number(a.sharpe).toFixed(2)}</td>
            </tr>
        `).join("");

        if (payload.explanation && explanationSection) {
            document.getElementById("expl-summary").textContent = payload.explanation.summary;
            document.getElementById("expl-risk-dist").textContent = payload.explanation.risk_distribution;
            document.getElementById("expl-monte-carlo").textContent = payload.explanation.monte_carlo;
            document.getElementById("expl-risk-decomp").textContent = payload.explanation.risk_decomposition;
            document.getElementById("expl-final").textContent = payload.explanation.final_statement;
            explanationSection.classList.remove("hidden");
        } else if (explanationSection) {
            explanationSection.classList.add("hidden");
        }

        summarySection.classList.remove("hidden");
        tableSection.classList.remove("hidden");
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
                method: String(data.get("method") || "proportional"),
                risk_profile: String(data.get("risk_profile") || "MEDIUM"),
                horizon_years: Number(data.get("horizon_years") || 3),
                num_simulations: Number(data.get("num_simulations") || 3000),
            };
            const res = await fetch("/api/risk", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Accept": "application/json" },
                body: JSON.stringify(payload),
            });
            const body = await res.json();
            render(body);
        } catch (error) {
            showBox(errorBox, "Unable to run risk analytics right now.");
        } finally {
            setLoading(false);
        }
    });
})();
