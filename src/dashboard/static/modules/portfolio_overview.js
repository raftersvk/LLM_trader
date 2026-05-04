/** Portfolio Overview module - Aggregated portfolio KPIs and pairs grid. */

let portfolioData = null;
let portfolioChart = null;

export async function initPortfolioOverview() {
    initPortfolioChart();
    await updatePortfolioData();
}

function initPortfolioChart() {
    const container = document.getElementById('portfolio-performance-chart');
    if (!container) return;

    const options = {
        series: [{ name: 'Portfolio Value', data: [] }],
        chart: { type: 'area', height: 250, background: '#161b22', toolbar: { show: true, tools: { download: false, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true }, autoSelected: 'zoom' }, zoom: { enabled: true, type: 'x', autoScaleYaxis: true }, animations: { enabled: true, easing: 'easeinout', speed: 500 } },
        colors: ['#00e396'],
        dataLabels: { enabled: false },
        stroke: { curve: 'smooth', width: 2 },
        fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.7, opacityTo: 0.1, stops: [0, 90, 100] } },
        theme: { mode: 'dark' },
        xaxis: { type: 'datetime', labels: { datetimeUTC: false, format: 'dd MMM HH:mm', style: { colors: '#c9d1d9', fontSize: '12px' }, rotate: -45, rotateAlways: false }, axisBorder: { show: false }, axisTicks: { show: true, color: '#30363d' } },
        yaxis: { labels: { formatter: (v) => new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v), style: { colors: '#c9d1d9', fontSize: '12px' } }, tickAmount: 5 },
        tooltip: { theme: 'dark', x: { format: 'dd MMM yyyy HH:mm' } },
        grid: { borderColor: '#30363d', strokeDashArray: 4, yaxis: { lines: { show: true } }, padding: { bottom: 10, top: 50, right: 30, left: 20 } },
        noData: { text: 'No trades yet', style: { color: '#8b949e', fontSize: '14px' } }
    };
    portfolioChart = new ApexCharts(container, options);
    portfolioChart.render();
}

export async function updatePortfolioData() {
    try {
        const response = await fetch('/api/portfolio/summary');
        portfolioData = await response.json();
        renderPortfolioKPIs(portfolioData);
        renderPairsGrid(portfolioData);
    } catch (e) {
        console.error('Failed to fetch portfolio data:', e);
    }

    try {
        const response = await fetch('/api/performance/history');
        const historyData = await response.json();
        if (portfolioChart && historyData.history && historyData.history.length > 0) {
            const seriesData = historyData.history.map(p => ({ x: new Date(p.time).getTime(), y: p.value }));
            portfolioChart.updateSeries([{ name: 'Portfolio Value', data: seriesData }]);
        }
    } catch (e) {
        console.error('Failed to fetch portfolio history:', e);
    }

    try {
        const response = await fetch('/api/monitor/costs');
        const data = await response.json();
        const el = document.getElementById('portfolio-cost');
        if (el) el.textContent = formatCost(data.total_session_cost || 0);
    } catch (e) {
        console.error('Failed to fetch costs:', e);
    }

    const el = document.getElementById('portfolio-last-updated');
    if (el) el.textContent = `Updated: ${new Intl.DateTimeFormat(navigator.language, { timeStyle: 'medium' }).format(new Date())}`;
}

function renderPortfolioKPIs(data) {
    const pnlEl = document.getElementById('portfolio-total-pnl');
    if (pnlEl) {
        const pnl = data.total_pnl_quote || 0;
        pnlEl.textContent = `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)} USDT`;
        pnlEl.className = `value ${pnl >= 0 ? 'start-green' : 'start-red'}`;
    }
    const wrEl = document.getElementById('portfolio-win-rate');
    if (wrEl) {
        const wr = data.win_rate || 0;
        wrEl.textContent = `${wr.toFixed(1)}%`;
        wrEl.className = `value highlight ${wr >= 50 ? 'start-green' : 'start-red'}`;
    }
    const tcEl = document.getElementById('portfolio-trade-count');
    if (tcEl) tcEl.textContent = `${data.total_trades || 0} trades`;

    const apEl = document.getElementById('portfolio-active-positions');
    if (apEl) apEl.textContent = data.active_positions || 0;
    const pcEl = document.getElementById('portfolio-pairs-count');
    if (pcEl) pcEl.textContent = `${data.total_pairs || 0} pairs`;
}

function renderPairsGrid(data) {
    const container = document.getElementById('pairs-status-grid');
    if (!container || !data.pairs) return;

    container.innerHTML = data.pairs.map(pair => {
        const trend = pair.trend || '--';
        const trendClass = trend === 'BULLISH' ? 'trend-bullish' : trend === 'BEARISH' ? 'trend-bearish' : 'trend-neutral';
        const action = pair.action || '--';
        const actionClass = action.includes('BUY') ? 'action-buy' : action.includes('SELL') ? 'action-sell' : 'action-hold';
        const confidence = pair.confidence || '--';
        const price = pair.current_price ? new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(pair.current_price) : '--';
        const hasPos = pair.position ? 'has-position' : 'no-position';
        const posLabel = pair.position ? (pair.position.direction || 'OPEN') : 'FLAT';

        return `
            <div class="pair-card ${hasPos}" data-symbol="${escapeHtml(pair.symbol)}" tabindex="0" role="button" aria-label="View ${escapeHtml(pair.symbol)} details">
                <div class="pair-card-header">
                    <span class="pair-card-symbol">${escapeHtml(pair.symbol)}</span>
                    <span class="pair-card-price">${price}</span>
                </div>
                <div class="pair-card-badges">
                    <span class="pair-badge ${trendClass}">${escapeHtml(trend)}</span>
                    <span class="pair-badge ${actionClass}">${escapeHtml(action)}</span>
                    <span class="pair-badge badge-confidence">${confidence !== '--' ? confidence + '%' : '--'}</span>
                </div>
                <div class="pair-card-position">${escapeHtml(posLabel)}</div>
            </div>
        `;
    }).join('');

    container.querySelectorAll('.pair-card').forEach(card => {
        const handler = () => {
            const symbol = card.getAttribute('data-symbol');
            if (symbol && window.state) {
                window.state.currentPair = symbol;
                const pairSelect = document.getElementById('pair-select');
                if (pairSelect) pairSelect.value = symbol;
                document.dispatchEvent(new CustomEvent('pair-changed', { detail: { symbol } }));
                const pairTab = document.querySelector('[data-target="tab-pair-detail"]');
                if (pairTab) pairTab.click();
            }
        };
        card.addEventListener('click', handler);
        card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); } });
    });
}

function formatCost(cost) {
    if (cost === 0 || cost === null || cost === undefined) return '$0.00';
    if (cost < 0.0001) return `$${cost.toFixed(8)}`;
    if (cost < 0.01) return `$${cost.toFixed(6)}`;
    if (cost < 1) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
