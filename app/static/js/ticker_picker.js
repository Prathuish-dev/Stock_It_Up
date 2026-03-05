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
        const helperText = options.helperTextId ? document.getElementById(options.helperTextId) : null;
        const maxItems = Number(options.maxItems || 20);

        if (!exchangeSelect || !hiddenInput || !searchInput || !selectedWrap || !suggestionsWrap) {
            return { getTickers: () => [] };
        }

        let tickers = unique(tokenize(hiddenInput.value)).slice(0, maxItems);
        let suggestions = [];
        let isPointerSelecting = false;
        let debounceTimer = null;
        let activeController = null;

        function syncHidden() {
            hiddenInput.value = tickers.join(" ");
        }

        function setHelperMessage(message, isError = false) {
            if (!helperText) return;
            helperText.textContent = message || "Only valid tickers from suggestions can be added.";
            helperText.classList.toggle("ticker-helper-error", Boolean(isError));
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

        function addTicker(raw, source = "typed") {
            const symbol = String(raw || "").trim().toUpperCase();
            if (!symbol) return;
            if (tickers.includes(symbol)) {
                searchInput.value = "";
                setHelperMessage("");
                return;
            }
            if (tickers.length >= maxItems) return;
            if (source === "typed") {
                setHelperMessage("Select a valid ticker from the dropdown suggestions.", true);
                return;
            }
            tickers.push(symbol);
            syncHidden();
            renderSelected();
            searchInput.value = "";
            setHelperMessage("");
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
                const pick = (event) => {
                    if (event) event.preventDefault();
                    isPointerSelecting = true;
                    addTicker(item, "suggestion");
                    setTimeout(() => {
                        isPointerSelecting = false;
                    }, 0);
                };
                btn.addEventListener("pointerdown", pick);
                btn.addEventListener("mousedown", (event) => {
                    pick(event);
                });
                btn.addEventListener("click", (event) => {
                    pick(event);
                });
                suggestionsWrap.appendChild(btn);
            });
            suggestionsWrap.classList.remove("hidden");
        }

        async function fetchSuggestions(query, forceShow = false) {
            const q = String(query || "").trim();
            if (!q && !forceShow) {
                hideSuggestions();
                return;
            }
            const exchange = exchangeSelect.value || "NSE";
            try {
                if (activeController) {
                    activeController.abort();
                }
                activeController = new AbortController();
                const res = await fetch(
                    `/api/tickers/search?exchange=${encodeURIComponent(exchange)}&q=${encodeURIComponent(q)}&limit=12`,
                    {
                        headers: { Accept: "application/json" },
                        signal: activeController.signal,
                    }
                );
                const body = await res.json();
                suggestions = (body.items || []).filter((x) => !tickers.includes(x));
                renderSuggestions();
            } catch (_err) {
                if (_err && _err.name === "AbortError") {
                    return;
                }
                hideSuggestions();
            } finally {
                activeController = null;
            }
        }

        async function fetchCandidatesOnce(query) {
            const q = String(query || "").trim();
            if (!q) return [];
            const exchange = exchangeSelect.value || "NSE";
            try {
                const res = await fetch(
                    `/api/tickers/search?exchange=${encodeURIComponent(exchange)}&q=${encodeURIComponent(q)}&limit=20`,
                    { headers: { Accept: "application/json" } }
                );
                const body = await res.json();
                return Array.isArray(body.items) ? body.items : [];
            } catch (_err) {
                return [];
            }
        }

        searchInput.addEventListener("input", () => {
            setHelperMessage("");
            if (debounceTimer) {
                clearTimeout(debounceTimer);
            }
            debounceTimer = setTimeout(() => {
                fetchSuggestions(searchInput.value, true);
            }, 250);
        });

        searchInput.addEventListener("focus", () => {
            fetchSuggestions(searchInput.value, true);
        });

        searchInput.addEventListener("click", () => {
            fetchSuggestions(searchInput.value, true);
        });

        searchInput.addEventListener("keydown", async (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                if (suggestions.length > 0) {
                    addTicker(suggestions[0], "suggestion");
                } else {
                    const typed = String(searchInput.value || "").trim().toUpperCase();
                    const candidates = await fetchCandidatesOnce(typed);
                    if (candidates.length > 0) {
                        const exact = candidates.find((x) => x.toUpperCase() === typed);
                        addTicker(exact || candidates[0], "suggestion");
                    } else {
                        addTicker(searchInput.value, "typed");
                    }
                }
            }
            if (event.key === "Escape") {
                hideSuggestions();
            }
        });

        searchInput.addEventListener("blur", () => {
            setTimeout(() => {
                if (!isPointerSelecting) {
                    hideSuggestions();
                }
            }, 120);
        });

        exchangeSelect.addEventListener("change", () => {
            hideSuggestions();
            tickers = [];
            syncHidden();
            renderSelected();
            setHelperMessage("Exchange changed. Please select tickers for the new exchange.");
        });

        syncHidden();
        renderSelected();
        setHelperMessage("");

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
