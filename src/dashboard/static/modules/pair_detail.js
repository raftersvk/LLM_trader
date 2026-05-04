/** Pair Detail module - Combined position + prompt + response + chart for one pair. */

let cachedPrompt = null;
let cachedResponse = null;

export async function updatePairDetail() {
    const symbol = window.state?.currentPair;
    if (!symbol || symbol === '__global__') {
        const container = document.getElementById('pair-detail-header');
        if (container) container.innerHTML = '<div class="empty-state"><p>Select a pair to view details</p></div>';
        return;
    }

    updatePairHeader(symbol);
    await updatePairPosition(symbol);
    await updatePairPrompt(symbol);
    await updatePairResponse(symbol);
    await updatePairChart(symbol);
}

async function updatePairHeader(symbol) {
    const container = document.getElementById('pair-detail-header');
    const title = document.getElementById('pair-detail-title');
    if (title) title.textContent = symbol;

    try {
        const response = await fetch(`/api/brain/status?symbol=${encodeURIComponent(symbol)}`);
        const data = await response.json();
        const trend = data.trend || '--';
        const trendClass = trend === 'BULLISH' ? 'trend-bullish' : trend === 'BEARISH' ? 'trend-bearish' : 'trend-neutral';
        const action = data.action || '--';
        const actionClass = action.includes('BUY') ? 'action-buy' : action.includes('SELL') ? 'action-sell' : 'action-hold';
        const confidence = data.confidence !== '--' && data.confidence !== undefined ? `${data.confidence}%` : '--';

        if (container) {
            container.innerHTML = `
                <span class="pair-badge ${trendClass}">${escapeHtml(trend)}</span>
                <span class="pair-badge ${actionClass}">${escapeHtml(action)}</span>
                <span class="pair-badge badge-confidence">${escapeHtml(confidence)}</span>
            `;
        }
    } catch (e) {
        console.error('Failed to load pair header:', e);
    }
}

async function updatePairPosition(symbol) {
    const container = document.getElementById('pair-position-content');
    if (!container) return;

    try {
        const response = await fetch(`/api/brain/position?symbol=${encodeURIComponent(symbol)}`);
        const data = await response.json();
        if (!data.has_position) {
            container.innerHTML = '<div class="no-position"><span class="no-position-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg></span><span>No active position</span></div>';
            return;
        }
        const priceToUse = data.current_price;
        const pnl = priceToUse ? calculatePnL(data.entry_price, priceToUse, data.direction) : null;
        const isProfit = pnl !== null && pnl >= 0;
        const directionClass = data.direction === 'LONG' ? 'long' : 'short';

        container.innerHTML = `
            <div class="position-grid">
                <div class="position-badge ${directionClass}"><span>${escapeHtml(data.direction)}</span></div>
                <div class="position-symbol">${escapeHtml(data.symbol)}</div>
                <div class="position-stat"><span class="label">Entry</span><span class="value">${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.entry_price)}</span></div>
                <div class="position-stat"><span class="label">Current</span><span class="value">${priceToUse ? new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(priceToUse) : '--'}</span></div>
                <div class="position-pnl ${isProfit ? 'profit' : 'loss'}">${pnl !== null ? (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%' : '--'}</div>
                <div class="position-sl-tp">
                    <div class="sl">SL: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.stop_loss)} <span class="pct">(-${(data.sl_distance_pct * 100).toFixed(1)}%)</span></div>
                    <div class="tp">TP: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.take_profit)} <span class="pct">(+${(data.tp_distance_pct * 100).toFixed(1)}%)</span></div>
                </div>
                <div class="position-meta"><span>R:R ${data.rr_ratio.toFixed(1)}</span><span>${escapeHtml(String(data.confidence))}</span></div>
            </div>
        `;
    } catch (e) {
        container.innerHTML = '<div class="error">Error loading position data</div>';
    }
}

async function updatePairPrompt(symbol) {
    const viewer = document.getElementById('pair-prompt-viewer');
    const meta = document.getElementById('pair-prompt-meta');
    if (!viewer) return;
    try {
        const response = await fetch(`/api/monitor/last_prompt?symbol=${encodeURIComponent(symbol)}`);
        const data = await response.json();
        const content = data.prompt || 'No prompt available';
        const timestamp = data.timestamp ? new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(data.timestamp)) : 'N/A';
        const source = data.source === 'disk' ? '💾 From disk' : '🧠 From memory';
        if (meta) meta.textContent = `${source} | ${timestamp}`;
        if (content && window.marked && window.DOMPurify) {
            viewer.innerHTML = DOMPurify.sanitize(marked.parse(content));
            viewer.classList.add('discord-content');
        } else {
            viewer.textContent = content;
        }
        cachedPrompt = content;
    } catch (e) {
        viewer.textContent = "Error fetching prompt: " + e.message;
    }
}

async function updatePairResponse(symbol) {
    const viewer = document.getElementById('pair-response-viewer');
    const meta = document.getElementById('pair-response-meta');
    if (!viewer) return;
    try {
        const response = await fetch(`/api/monitor/last_response?symbol=${encodeURIComponent(symbol)}`);
        const data = await response.json();
        const content = data.response || 'No response available';
        const timestamp = data.timestamp ? new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(data.timestamp)) : 'N/A';
        const source = data.source === 'disk' ? '💾 From disk' : '🧠 From memory';
        if (meta) meta.textContent = `${source} | ${timestamp}`;
        if (content && window.marked) {
            let processed = content;
            processed = processed.replace(/```json[\s\S]*?```/g, '');
            processed = processed.replace(/\n*\{\s*"analysis"[\s\S]*?\n\}\s*$/g, '');
            processed = processed.replace(/\n{3,}/g, '\n\n');
            if (window.DOMPurify) {
                viewer.innerHTML = DOMPurify.sanitize(marked.parse(processed.trim()));
            } else {
                viewer.textContent = processed.trim();
            }
        } else {
            viewer.innerHTML = `<pre style="white-space: pre-wrap; margin: 0;">${escapeHtml(content)}</pre>`;
        }
        cachedResponse = content;
    } catch (e) {
        viewer.textContent = "Error fetching response: " + e.message;
    }
}

async function updatePairChart(symbol) {
    const img = document.getElementById('pair-analysis-chart');
    const noMsg = document.getElementById('pair-no-chart-msg');
    const container = document.getElementById('pair-chart-container');
    if (!img || !noMsg) return;
    try {
        const response = await fetch(`/api/visuals/charts/latest?symbol=${encodeURIComponent(symbol)}`);
        if (response.ok) {
            const data = await response.json();
            if (data.chart_base64) {
                img.src = `data:image/png;base64,${data.chart_base64}`;
                img.style.display = 'block';
                noMsg.style.display = 'none';
                if (container) container.style.position = 'relative';
            } else {
                img.style.display = 'none';
                noMsg.textContent = 'No chart in memory - waiting for next analysis';
                noMsg.style.display = 'block';
            }
        } else {
            img.style.display = 'none';
            noMsg.textContent = 'Chart service unavailable';
            noMsg.style.display = 'block';
        }
    } catch (e) {
        img.style.display = 'none';
        noMsg.textContent = 'Error loading chart';
        noMsg.style.display = 'block';
    }
}

function calculatePnL(entryPrice, currentPrice, direction) {
    if (!entryPrice || !currentPrice) return 0;
    const diff = direction === 'LONG' ? (currentPrice - entryPrice) / entryPrice : (entryPrice - currentPrice) / entryPrice;
    return diff * 100;
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

export function initPairDetail() {
    const btnRefresh = document.getElementById('btn-refresh-price');
    if (btnRefresh) {
        btnRefresh.addEventListener('click', async () => {
            const symbol = window.state?.currentPair;
            if (!symbol || symbol === '__global__') return;
            try {
                const resp = await fetch(`/api/brain/refresh-price?symbol=${encodeURIComponent(symbol)}`);
                const data = await resp.json();
                if (data.success) await updatePairPosition(symbol);
            } catch (e) { console.error('Price refresh failed:', e); }
        });
    }

    const btnCopyPrompt = document.getElementById('btn-copy-pair-prompt');
    if (btnCopyPrompt) {
        btnCopyPrompt.addEventListener('click', () => {
            if (!cachedPrompt) return;
            navigator.clipboard.writeText(cachedPrompt).catch(err => console.error('Copy failed:', err));
        });
    }

    const btnCopyResponse = document.getElementById('btn-copy-pair-response');
    if (btnCopyResponse) {
        btnCopyResponse.addEventListener('click', () => {
            if (!cachedResponse) return;
            navigator.clipboard.writeText(cachedResponse).catch(err => console.error('Copy failed:', err));
        });
    }

    document.querySelectorAll('[id^="accordion-"][id$="-header"]').forEach(header => {
        header.addEventListener('click', () => {
            const body = header.nextElementSibling;
            if (body) body.classList.toggle('collapsed');
        });
    });
}
