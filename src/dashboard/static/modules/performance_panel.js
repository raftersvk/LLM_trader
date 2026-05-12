/** Performance Panel module - Detailed per-pair and cross-pair analytics. */

let performanceChart = null;
let tradeHistory = [];
let tradePage = 0;
const TRADES_PER_PAGE = 10;

export async function initPerformancePanel() {
    await updatePerformancePanel();
}

export async function updatePerformancePanel() {
    const container = document.getElementById('performance-content');
    if (!container) return;

    try {
        const symbol = window.state?.currentPair;
        const symbolParam = symbol && symbol !== '__global__' ? `?symbol=${encodeURIComponent(symbol)}` : '';

        const [statsResp, historyResp, tradesResp] = await Promise.all([
            fetch(`/api/performance/stats${symbolParam}`),
            fetch(`/api/performance/history${symbolParam}`),
            fetch(`/api/performance/trades${symbolParam}`)
        ]);

        const stats = await statsResp.json();
        const historyData = await historyResp.json();
        const tradesData = await tradesResp.json();

        tradeHistory = tradesData.trades || [];
        tradePage = 0;

        renderPerformancePanel(container, stats, historyData, tradeHistory);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><p>Error loading performance data</p><p style="font-size: 0.85em; color: var(--accent-danger);">${escapeHtml(e.message)}</p></div>`;
    }
}

function renderPerformancePanel(container, stats, historyData, _trades) {
    container.innerHTML = `
        <div class="stats-grid">
            ${renderStatsCards(stats)}
        </div>
        <section class="panel" style="margin-top: 24px;">
            <div class="panel-header"><h3>Performance Chart</h3></div>
            <div id="perf-detail-chart" class="chart-container"></div>
        </section>
        <section class="panel" style="margin-top: 24px;">
            <div class="panel-header">
                <h3>Trade History</h3>
                <div id="trade-pagination-info" style="font-size: 12px; color: var(--text-muted);"></div>
            </div>
            <div id="trade-history-table" style="overflow-x: auto;"></div>
            <div id="trade-pagination-controls" style="display: flex; justify-content: center; gap: 8px; margin-top: 16px;"></div>
        </section>
    `;

    initPerfChart(historyData);
    renderTradeHistory();
}

function renderTradeHistory() {
    const tableContainer = document.getElementById('trade-history-table');
    const paginationInfo = document.getElementById('trade-pagination-info');
    const paginationControls = document.getElementById('trade-pagination-controls');

    if (!tableContainer || !tradeHistory.length) {
        if (tableContainer) tableContainer.innerHTML = '<div class="empty-state"><p>No trades recorded yet</p></div>';
        return;
    }

    const totalPages = Math.ceil(tradeHistory.length / TRADES_PER_PAGE);
    const start = tradePage * TRADES_PER_PAGE;
    const end = start + TRADES_PER_PAGE;
    const pageTrades = tradeHistory.slice(start, end);

    paginationInfo.textContent = `Showing ${start + 1}-${Math.min(end, tradeHistory.length)} of ${tradeHistory.length} trades`;

    tableContainer.innerHTML = `
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead>
                <tr style="border-bottom: 1px solid var(--border-color);">
                    <th style="text-align: left; padding: 8px; color: var(--text-muted);">Time</th>
                    <th style="text-align: left; padding: 8px; color: var(--text-muted);">Pair</th>
                    <th style="text-align: left; padding: 8px; color: var(--text-muted);">Action</th>
                    <th style="text-align: right; padding: 8px; color: var(--text-muted);">Price</th>
                    <th style="text-align: right; padding: 8px; color: var(--text-muted);">Quantity</th>
                    <th style="text-align: right; padding: 8px; color: var(--text-muted);">P&L</th>
                </tr>
            </thead>
            <tbody>
                ${pageTrades.map(trade => {
                    const actionClass = trade.action?.toUpperCase().includes('BUY') ? 'stat-positive' :
                                       trade.action?.toUpperCase().includes('CLOSE') ? 'stat-negative' : '';
                    const pnl = trade.pnl_quote !== undefined ? trade.pnl_quote : (trade.pnl !== undefined ? trade.pnl : null);
                    const pnlDisplay = pnl !== null ? `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}` : '-';
                    const pnlClass = pnl !== null ? (pnl >= 0 ? 'stat-positive' : 'stat-negative') : '';
                    return `
                        <tr style="border-bottom: 1px solid var(--border-subtle);">
                            <td style="padding: 8px;">${formatTimestamp(trade.timestamp)}</td>
                            <td style="padding: 8px;">${escapeHtml(trade.symbol || '-')}</td>
                            <td style="padding: 8px;"><span class="${actionClass}">${escapeHtml(trade.action || '-')}</span></td>
                            <td style="padding: 8px; text-align: right;">${trade.price ? '$' + parseFloat(trade.price).toFixed(4) : '-'}</td>
                            <td style="padding: 8px; text-align: right;">${trade.quantity || trade.amount || '-'}</td>
                            <td style="padding: 8px; text-align: right;" class="${pnlClass}">${pnlDisplay}</td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;

    paginationControls.innerHTML = '';
    if (totalPages > 1) {
        const prevBtn = document.createElement('button');
        prevBtn.className = 'toolbar-btn';
        prevBtn.textContent = '← Prev';
        prevBtn.disabled = tradePage === 0;
        prevBtn.onclick = () => { tradePage--; renderTradeHistory(); };
        paginationControls.appendChild(prevBtn);

        for (let i = 0; i < totalPages; i++) {
            const btn = document.createElement('button');
            btn.className = `toolbar-btn ${i === tradePage ? 'active' : ''}`;
            btn.textContent = i + 1;
            btn.onclick = () => { tradePage = i; renderTradeHistory(); };
            paginationControls.appendChild(btn);
        }

        const nextBtn = document.createElement('button');
        nextBtn.className = 'toolbar-btn';
        nextBtn.textContent = 'Next →';
        nextBtn.disabled = tradePage >= totalPages - 1;
        nextBtn.onclick = () => { tradePage++; renderTradeHistory(); };
        paginationControls.appendChild(nextBtn);
    }
}

function formatTimestamp(ts) {
    if (!ts) return '-';
    const d = new Date(ts);
    return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false
    });
}

function renderStatsCards(stats) {
    if (!stats || Object.keys(stats).length === 0) {
        return '<div class="empty-state"><p>No statistics available yet</p></div>';
    }
    const cards = [
        { title: "Total Trades", value: `${stats.total_trades || 0}`, sub: `${stats.winning_trades || 0}W / ${stats.losing_trades || 0}L`, color: "stat-neutral" },
        { title: "Win Rate", value: `${(stats.win_rate || 0).toFixed(1)}%`, color: (stats.win_rate || 0) >= 50 ? "stat-positive" : "stat-negative" },
        { title: "Total P&L", value: `${(stats.total_pnl_pct || 0) >= 0 ? '+' : ''}${(stats.total_pnl_pct || 0).toFixed(2)}%`, sub: `$${(stats.total_pnl_quote || 0).toFixed(2)}`, color: (stats.total_pnl_quote || 0) >= 0 ? "stat-positive" : "stat-negative" },
        { title: "Sharpe Ratio", value: (stats.sharpe_ratio || 0).toFixed(2), color: (stats.sharpe_ratio || 0) >= 1 ? "stat-positive" : "stat-neutral" },
        { title: "Profit Factor", value: (stats.profit_factor || 0).toFixed(2), color: (stats.profit_factor || 0) >= 1.5 ? "stat-positive" : "stat-neutral" },
        { title: "Max Drawdown", value: `${(stats.max_drawdown_pct || 0).toFixed(2)}%`, color: "stat-negative" },
    ];
    return cards.map(c => `
        <div class="stat-card-lg">
            <div class="stat-header"><span class="stat-title">${escapeHtml(c.title)}</span></div>
            <div class="stat-value-lg ${escapeHtml(c.color)}">${escapeHtml(c.value)}</div>
            ${c.sub ? `<div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">${escapeHtml(c.sub)}</div>` : ''}
        </div>
    `).join('');
}

function initPerfChart(historyData) {
    const container = document.getElementById('perf-detail-chart');
    if (!container) return;

    if (performanceChart) {
        performanceChart.destroy();
        performanceChart = null;
    }

    const options = {
        series: [{ name: 'Account Value', data: [] }],
        chart: { type: 'area', height: 280, background: '#161b22', toolbar: { show: false }, animations: { enabled: true, easing: 'easeinout', speed: 500 } },
        colors: ['#00e396'],
        dataLabels: { enabled: false },
        stroke: { curve: 'smooth', width: 2 },
        fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.7, opacityTo: 0.1, stops: [0, 90, 100] } },
        theme: { mode: 'dark' },
        xaxis: { type: 'datetime', labels: { datetimeUTC: false, format: 'dd MMM HH:mm', style: { colors: '#c9d1d9', fontSize: '12px' } } },
        yaxis: { labels: { formatter: (v) => new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v), style: { colors: '#c9d1d9', fontSize: '12px' } } },
        tooltip: { theme: 'dark' },
        grid: { borderColor: '#30363d', strokeDashArray: 4 },
        noData: { text: 'No trades yet', style: { color: '#8b949e', fontSize: '14px' } }
    };

    performanceChart = new window.ApexCharts(container, options);
    performanceChart.render();

    if (historyData.history && historyData.history.length > 0) {
        const seriesData = historyData.history.map(p => ({ x: new Date(p.time).getTime(), y: p.value }));
        performanceChart.updateSeries([{ name: 'Account Value', data: seriesData }]);
    }
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
