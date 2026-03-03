(() => {
    if (typeof Chart === "undefined") {
        return;
    }

    const chartState = {
        scatter: null,
        bar: null,
        pie: null,
    };

    function destroyCharts() {
        if (chartState.scatter) chartState.scatter.destroy();
        if (chartState.bar) chartState.bar.destroy();
        if (chartState.pie) chartState.pie.destroy();
        chartState.scatter = null;
        chartState.bar = null;
        chartState.pie = null;
    }

    function renderRankingCharts(payload) {
        destroyCharts();

        const scatterData = payload.scatter || [];
        const barData = payload.cagrBar || { labels: [], values: [] };
        const pieData = payload.pie || { labels: [], values: [] };

        const scatterCanvas = document.getElementById("riskReturnChart");
        if (scatterCanvas && scatterData.length > 0) {
            chartState.scatter = new Chart(scatterCanvas, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Risk vs Return",
                        data: scatterData,
                        backgroundColor: "#22d3ee",
                        borderColor: "#0891b2",
                        pointRadius: 5,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const point = ctx.raw || {};
                                return `${point.ticker}: ${point.y}% return, ${point.x}% risk`;
                            },
                        },
                    },
                },
                scales: {
                    x: { title: { display: true, text: "Volatility (%)" } },
                    y: { title: { display: true, text: "CAGR (%)" } },
                },
            },
            });
        }

        const barCanvas = document.getElementById("cagrBarChart");
        if (barCanvas && barData.labels.length > 0) {
            chartState.bar = new Chart(barCanvas, {
            type: "bar",
            data: {
                labels: barData.labels,
                datasets: [
                    {
                        label: "CAGR %",
                        data: barData.values,
                        backgroundColor: "#0ea5e9",
                        borderRadius: 8,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
            },
            });
        }

        const pieCanvas = document.getElementById("allocationPieChart");
        if (pieCanvas && pieData.labels.length > 0) {
            chartState.pie = new Chart(pieCanvas, {
            type: "pie",
            data: {
                labels: pieData.labels,
                datasets: [
                    {
                        data: pieData.values,
                        backgroundColor: [
                            "#22d3ee",
                            "#0ea5e9",
                            "#0284c7",
                            "#0369a1",
                            "#0891b2",
                            "#06b6d4",
                            "#38bdf8",
                        ],
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
            },
            });
        }
    }

    window.StockCharts = {
        renderRankingCharts,
    };

    const payloadNode = document.getElementById("ranking-chart-data");
    if (!payloadNode) {
        return;
    }

    let payload = {};
    try {
        payload = JSON.parse(payloadNode.textContent || "{}");
    } catch (error) {
        payload = {};
    }
    renderRankingCharts(payload);
})();
