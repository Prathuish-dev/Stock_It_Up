(() => {
    const form = document.getElementById("ranking-form");
    if (!form) {
        return;
    }

    const button = document.getElementById("run-ranking-btn");
    const label = button ? button.querySelector(".btn-label") : null;
    const spinner = button ? button.querySelector(".spinner") : null;

    const errorBox = document.getElementById("ranking-error");
    const warningBox = document.getElementById("ranking-warning");
    const resultsSection = document.getElementById("ranking-results-section");
    const chartsSection = document.getElementById("ranking-charts-section");
    const resultsSummary = document.getElementById("results-summary");
    const metricHeaderLabel = document.getElementById("metric-column-label");
    const resultsBody = document.getElementById("ranking-results-body");
    const pagination = document.getElementById("ranking-pagination");
    const sortButtons = Array.from(document.querySelectorAll(".table-sort-btn"));

    let currentRows = [];
    let currentOrder = "best";
    let sortState = { key: "rank", dir: "asc" };

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

    function setLoading(isLoading) {
        if (!button) return;
        button.disabled = isLoading;
        if (label) label.textContent = isLoading ? "Processing..." : "Run Ranking";
        if (spinner) spinner.classList.toggle("hidden", !isLoading);
    }

    function formatPercent(value) {
        return `${(Number(value || 0) * 100).toFixed(2)}%`;
    }

    function formatNumber(value) {
        return Number(value || 0).toFixed(2);
    }

    function updateSortIndicators() {
        sortButtons.forEach((btn) => {
            const indicator = btn.querySelector(".sort-indicator");
            if (!indicator) return;
            if (btn.dataset.sortKey === sortState.key) {
                indicator.textContent = sortState.dir === "asc" ? "▲" : "▼";
            } else {
                indicator.textContent = "";
            }
        });
    }

    function renderTableRows(rows) {
        if (!resultsBody) return;
        resultsBody.innerHTML = rows.map((row, idx) => {
            let extraClass = "";
            if (idx === 0 && currentOrder === "best") extraClass = "row-best";
            if (idx === 0 && currentOrder === "worst") extraClass = "row-worst";
            return `
                <tr
                    class="border-t border-slate-700/60 ${extraClass}"
                    data-rank="${row.rank}"
                    data-rank-label="${row.rank_label}"
                    data-ticker="${row.ticker}"
                    data-metric-value="${row.metric_value}"
                    data-display-value="${row.display_value}"
                    data-cagr="${row.cagr}"
                    data-volatility="${row.volatility}"
                    data-sharpe="${row.sharpe}"
                >
                    <td class="px-3 py-3">${row.rank_label || row.rank || ""}</td>
                    <td class="px-3 py-3 font-semibold text-white">${row.ticker || ""}</td>
                    <td class="px-3 py-3">${row.display_value || ""}</td>
                    <td class="px-3 py-3">${formatPercent(row.cagr)}</td>
                    <td class="px-3 py-3">${formatPercent(row.volatility)}</td>
                    <td class="px-3 py-3">${formatNumber(row.sharpe)}</td>
                </tr>
            `;
        }).join("");
    }

    function normalizeRows(rawRows) {
        return (rawRows || []).map((row) => {
            return {
                rank: Number(row.rank || 0),
                rank_label: row.rank_label || "",
                ticker: row.ticker || "",
                metric_value: Number(row.metric_value || row.value || 0),
                display_value: row.display_value || "",
                cagr: Number(row.cagr || (row.metrics && row.metrics.cagr) || 0),
                volatility: Number(
                    row.volatility || (row.metrics && row.metrics.volatility) || 0
                ),
                sharpe: Number(row.sharpe || (row.metrics && row.metrics.sharpe) || 0),
            };
        });
    }

    function getInitialRowsFromDom() {
        if (!resultsBody) return [];
        return Array.from(resultsBody.querySelectorAll("tr")).map((tr) => ({
            rank: Number(tr.dataset.rank || 0),
            rank_label: tr.dataset.rankLabel || "",
            ticker: tr.dataset.ticker || "",
            metric_value: Number(tr.dataset.metricValue || 0),
            display_value: tr.dataset.displayValue || "",
            cagr: Number(tr.dataset.cagr || 0),
            volatility: Number(tr.dataset.volatility || 0),
            sharpe: Number(tr.dataset.sharpe || 0),
        }));
    }

    function sortRowsInPlace() {
        const dirFactor = sortState.dir === "asc" ? 1 : -1;
        currentRows.sort((a, b) => {
            let cmp = 0;
            switch (sortState.key) {
                case "rank":
                    cmp = a.rank - b.rank;
                    break;
                case "ticker":
                    cmp = a.ticker.localeCompare(b.ticker);
                    break;
                case "metric":
                    cmp = a.metric_value - b.metric_value;
                    break;
                case "cagr":
                    cmp = a.cagr - b.cagr;
                    break;
                case "volatility":
                    cmp = a.volatility - b.volatility;
                    break;
                case "sharpe":
                    cmp = a.sharpe - b.sharpe;
                    break;
                default:
                    cmp = 0;
            }
            if (cmp === 0) {
                cmp = a.ticker.localeCompare(b.ticker);
            }
            return cmp * dirFactor;
        });
    }

    function applySort(key) {
        if (!currentRows.length) return;
        if (sortState.key === key) {
            sortState.dir = sortState.dir === "asc" ? "desc" : "asc";
        } else {
            sortState.key = key;
            if (key === "rank" || key === "ticker") {
                sortState.dir = "asc";
            } else {
                sortState.dir = "desc";
            }
        }
        sortRowsInPlace();
        renderTableRows(currentRows);
        updateSortIndicators();
    }

    function renderResults(payload) {
        if (!payload || !payload.ok) {
            showBox(errorBox, payload && payload.error ? payload.error : "Unable to load ranking.");
            return;
        }

        showBox(errorBox, payload.error);
        showBox(warningBox, payload.warning);

        const rows = payload.results || [];
        if (rows.length === 0) {
            if (resultsSection) resultsSection.classList.add("hidden");
            if (chartsSection) chartsSection.classList.add("hidden");
            return;
        }

        currentOrder = payload.selected && payload.selected.order ? payload.selected.order : "best";
        currentRows = normalizeRows(rows);
        sortState = { key: "rank", dir: "asc" };
        renderTableRows(currentRows);
        updateSortIndicators();

        if (metricHeaderLabel) {
            metricHeaderLabel.textContent = payload.metric_display || "Metric";
        }

        if (resultsSummary) {
            const pg = payload.pagination || {};
            resultsSummary.textContent = `Showing page ${pg.page} of ${pg.total_pages} (${pg.total_results} results)`;
        }

        if (pagination) {
            const pg = payload.pagination || {};
            const page = Number(pg.page || 1);
            const totalPages = Number(pg.total_pages || 1);
            const selected = payload.selected || {};
            if (totalPages <= 1) {
                pagination.innerHTML = "";
            } else {
                const links = [];
                for (let p = 1; p <= totalPages; p += 1) {
                    const activeClass = p === page ? "page-link-active" : "";
                    links.push(
                        `<a class="page-link ${activeClass}" href="#" data-page="${p}" data-ajax-page="1">${p}</a>`
                    );
                }
                pagination.innerHTML = links.join("");
                pagination.dataset.exchange = selected.exchange || "NSE";
                pagination.dataset.metric = selected.metric || "cagr";
                pagination.dataset.order = selected.order || "best";
                pagination.dataset.limit = String(selected.limit || 10);
                pagination.dataset.horizonYears = String(selected.horizon_years || 3);
            }
        }

        if (resultsSection) resultsSection.classList.remove("hidden");
        if (chartsSection) chartsSection.classList.remove("hidden");

        if (window.StockCharts && typeof window.StockCharts.renderRankingCharts === "function") {
            window.StockCharts.renderRankingCharts(payload.chart_data || {});
        }
    }

    async function fetchRanking(params, pushHistory = true) {
        setLoading(true);
        showBox(errorBox, null);

        try {
            const url = `/api/ranking?${params.toString()}`;
            const response = await fetch(url, {
                method: "GET",
                headers: { "Accept": "application/json" },
            });

            const payload = await response.json();
            renderResults(payload);

            if (pushHistory && payload && payload.ok) {
                const selected = payload.selected || {};
                const query = new URLSearchParams({
                    exchange: selected.exchange || "NSE",
                    metric: selected.metric || "cagr",
                    order: selected.order || "best",
                    limit: String(selected.limit || 10),
                    horizon_years: String(selected.horizon_years || 3),
                    page: String((payload.pagination && payload.pagination.page) || 1),
                });
                window.history.replaceState({}, "", `/ranking?${query.toString()}`);
            }
        } catch (error) {
            form.submit();
        } finally {
            setLoading(false);
        }
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const params = new URLSearchParams(new FormData(form));
        params.set("page", "1");
        fetchRanking(params, true);
    });

    if (pagination) {
        pagination.addEventListener("click", (event) => {
            const target = event.target;
            if (!(target instanceof HTMLAnchorElement)) return;
            if (target.dataset.ajaxPage !== "1") return;
            event.preventDefault();

            const page = target.dataset.page || "1";
            const params = new URLSearchParams({
                exchange: pagination.dataset.exchange || "NSE",
                metric: pagination.dataset.metric || "cagr",
                order: pagination.dataset.order || "best",
                limit: pagination.dataset.limit || "10",
                horizon_years: pagination.dataset.horizonYears || "3",
                page,
            });
            fetchRanking(params, true);
        });
    }

    sortButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const key = btn.dataset.sortKey;
            if (!key) return;
            applySort(key);
        });
    });

    if (pagination && pagination.dataset.order) {
        currentOrder = pagination.dataset.order;
    }
    currentRows = getInitialRowsFromDom();
    updateSortIndicators();
})();
