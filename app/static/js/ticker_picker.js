(() => {
    function tokenize(raw) {
        return String(raw || "")
            .replace(/,/g, " ")
            .split(/\s+/)
            .map((x) => x.trim().toUpperCase())
            .filter(Boolean);
    }

    function unique(items) {
        const out = [];
        const seen = new Set();
        for (const item of items) {
            if (!seen.has(item)) {
                seen.add(item);
                out.push(item);
            }
        }
        return out;
    }

    window.createTickerPicker = function createTickerPicker(options) {
        const exchangeSelect = document.getElementById(options.exchangeSelectId);
        const hiddenInput = document.getElementById(options.hiddenInputId);
        const searchInput = document.getElementById(options.searchInputId);
        const selectedWrap = document.getElementById(options.selectedContainerId);
        const suggestionsWrap = document.getElementById(options.suggestionsContainerId);
        const maxItems = Number(options.maxItems || 20);

        if (!exchangeSelect || !hiddenInput || !searchInput || !selectedWrap || !suggestionsWrap) {
            return { getTickers: () => [] };
        }

        let tickers = unique(tokenize(hiddenInput.value)).slice(0, maxItems);
        let suggestions = [];

        function syncHidden() {
            hiddenInput.value = tickers.join(" ");
        }

        function renderSelected() {
            selectedWrap.innerHTML = "";
            tickers.forEach((ticker) => {
                const chip = document.createElement("span");
                chip.className = "ticker-chip";
                chip.textContent = ticker;

                const removeBtn = document.createElement("button");
                removeBtn.type = "button";
                removeBtn.className = "ticker-chip-remove";
                removeBtn.textContent = "x";
                removeBtn.addEventListener("click", () => {
                    tickers = tickers.filter((t) => t !== ticker);
                    syncHidden();
                    renderSelected();
                });

                chip.appendChild(removeBtn);
                selectedWrap.appendChild(chip);
            });
        }

        function addTicker(raw) {
            const symbol = String(raw || "").trim().toUpperCase();
            if (!symbol) return;
            if (tickers.includes(symbol)) {
                searchInput.value = "";
                return;
            }
            if (tickers.length >= maxItems) return;
            tickers.push(symbol);
            syncHidden();
            renderSelected();
            searchInput.value = "";
            hideSuggestions();
        }

        function hideSuggestions() {
            suggestions = [];
            suggestionsWrap.innerHTML = "";
            suggestionsWrap.classList.add("hidden");
        }

        function renderSuggestions() {
            suggestionsWrap.innerHTML = "";
            if (!suggestions.length) {
                hideSuggestions();
                return;
            }
            suggestions.forEach((item) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "ticker-suggestion-item";
                btn.textContent = item;
                btn.addEventListener("click", () => addTicker(item));
                suggestionsWrap.appendChild(btn);
            });
            suggestionsWrap.classList.remove("hidden");
        }

        async function fetchSuggestions(query) {
            const q = String(query || "").trim();
            if (!q) {
                hideSuggestions();
                return;
            }
            const exchange = exchangeSelect.value || "NSE";
            try {
                const res = await fetch(
                    `/api/tickers/search?exchange=${encodeURIComponent(exchange)}&q=${encodeURIComponent(q)}&limit=12`,
                    { headers: { Accept: "application/json" } }
                );
                const body = await res.json();
                suggestions = (body.items || []).filter((x) => !tickers.includes(x));
                renderSuggestions();
            } catch (_err) {
                hideSuggestions();
            }
        }

        searchInput.addEventListener("input", () => {
            fetchSuggestions(searchInput.value);
        });

        searchInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                if (suggestions.length > 0) {
                    addTicker(suggestions[0]);
                } else {
                    addTicker(searchInput.value);
                }
            }
            if (event.key === "Escape") {
                hideSuggestions();
            }
        });

        searchInput.addEventListener("blur", () => {
            setTimeout(() => hideSuggestions(), 120);
        });

        exchangeSelect.addEventListener("change", () => {
            hideSuggestions();
        });

        syncHidden();
        renderSelected();

        return {
            getTickers: () => [...tickers],
            setTickers: (next) => {
                tickers = unique(tokenize(Array.isArray(next) ? next.join(" ") : next)).slice(0, maxItems);
                syncHidden();
                renderSelected();
            },
        };
    };
})();

