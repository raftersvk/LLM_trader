/** Performance Panel module - Detailed per-pair and cross-pair analytics. */

import { initPerformanceChart as initOverviewChart, updatePerformanceData as updateOverviewData } from './performance_chart.js?v=4.5';

let performanceChart = null;
let comparisonChart = null;

export async function initPerformancePanel() {
    await updatePerformancePanel();
}

export async function updatePerformancePanel() {
    const container = document.getElementById('performance-content');
    if (!container) return;

    try {
        const symbol = window.state?.currentPair;
        const symbolParam = symbol && symbol !== '__global__' ? `?symbol=${encodeURIComponent(symbol)}` : '';

        const [statsResp, historyResp, compareResp] = await Promise.all([
            fetch(`/api/performance/stats${symbolParam}`),
            fetch(`/api/performance/history${symbolParam}`),
            fetch('/api/performance/compare')
        ]);

        const stats = await statsResp.json();
        const historyData = await historyResp.json();
        const compareData = await compareResp.json();

        renderPerformancePanel(container, stats, historyData, compareData);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><p>Error loading performance data</p><p style="font-size: 0.85em; color: var(--accent-danger);">${escapeHtml(e.message)}</p></div>`;
    }
}

function renderPerformancePanel(container, stats, historyData, compareData) {
    const pairs = compareData.pairs || [];

    container.innerHTML = `
        <section class="panel">
            <div class="panel-header"><h3>Performance Chart</h3></div>
            <div id="perf-detail-chart" class="chart-container"></div>
        </section>
        <div class="stats-grid">
            ${renderStatsCards(stats)}
        </div>
        <section class="panel" style="margin-top: 24px;">
            <div class="panel-header"><h3>Pair Comparison</h3></div>
            <div id="perf-comparison-chart" style="height: 300px;"></div>
        </section>
    `;

    initPerfChart(historyData);
    initComparisonChart(pairs);
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

    performanceChart = new ApexCharts(container, options);
    performanceChart.render();

    if (historyData.history && historyData.history.length > 0) {
        const seriesData = historyData.history.map(p => ({ x: new Date(p.time).getTime(), y: p.value }));
        performanceChart.updateSeries([{ name: 'Account Value', data: seriesData }]);
    }
}

function initComparisonChart(pairs) {
    const container = document.getElementById('perf-comparison-chart');
    if (!container) return;

    if (comparisonChart) {
        comparisonChart.destroy();
        comparisonChart = null;
    }

    const symbols = pairs.filter(p => !p.error).map(p => p.symbol);
    const pnlData = pairs.filter(p => !p.error).map(p => p.total_pnl_pct || 0);
    const wrData = pairs.filter(p => !p.error).map(p => p.win_rate || 0);

    const options = {
        series: [
            { name: 'P&L %', data: pnlData },
            { name: 'Win Rate %', data: wrData }
        ],
        chart: { type: 'bar', height: 300, background: '#161b22', toolbar: { show: false } },
        colors: ['#00e396', '#00f0ff'],
        plotOptions: { bar: { horizontal: false, columnWidth: '55%', borderRadius: 4 } },
        dataLabels: { enabled: false },
        stroke: { show: true, width: 2, colors: ['transparent'] },
        theme: { mode: 'dark' },
        xaxis: { categories: symbols, labels: { style: { colors: '#c9d1d9', fontSize: '11px' } } },
        yaxis: { labels: { formatter: (v) => v.toFixed(1) + '%', style: { colors: '#c9d1d9', fontSize: '11px' } } },
        tooltip: { theme: 'dark', y: { formatter: (v) => v.toFixed(2) + '%' } },
        grid: { borderColor: '#30363d', strokeDashArray: 4 },
        legend: { labels: { colors: '#c9d1d9' } }
    };

    comparisonChart = new ApexCharts(container, options);
    comparisonChart.render();
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
