/** System Monitor module - Bot fleet status, API costs, countdowns. */

export async function initSystemMonitor() {
    await updateSystemData();
}

export async function updateSystemData() {
    const container = document.getElementById('system-content');
    if (!container) return;

    try {
        const [statusResp, costsResp] = await Promise.all([
            fetch('/api/system/status'),
            fetch('/api/monitor/costs')
        ]);
        const status = await statusResp.json();
        const costs = await costsResp.json();

        renderSystemMonitor(container, status, costs);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><p>Error loading system status</p><p style="font-size: 0.85em; color: var(--accent-danger);">${escapeHtml(e.message)}</p></div>`;
    }
}

function renderSystemMonitor(container, status, costs) {
    const pairs = status.pairs || {};
    const countdowns = status.countdowns || {};

    const pairsHtml = Object.entries(pairs).map(([symbol, info]) => {
        const running = info.is_running;
        const statusDot = running ? '🟢' : '🔴';
        const statusText = running ? 'Running' : 'Stopped';
        const error = info.last_error ? `<div class="system-error">${escapeHtml(info.last_error)}</div>` : '';
        const countdown = countdowns[symbol];
        const countdownText = countdown && countdown.seconds_remaining > 0
            ? formatDuration(countdown.seconds_remaining)
            : '--:--';
        const taskActive = info.task_active ? '<span class="system-badge badge-active">Task Active</span>' : '';

        return `
            <div class="system-pair-row">
                <div class="system-pair-symbol">${statusDot} ${escapeHtml(symbol)}</div>
                <div class="system-pair-status ${running ? 'status-running' : 'status-stopped'}">${statusText}</div>
                <div class="system-pair-countdown">${countdownText}</div>
                <div class="system-pair-analyses">Analyses: ${info.analysis_count || 0}</div>
                ${taskActive}
                ${error}
            </div>
        `;
    }).join('');

    const costByProvider = costs.costs_by_provider || {};
    const totalCost = costs.total_session_cost || 0;

    container.innerHTML = `
        <div class="system-grid">
            <div class="system-section">
                <h3>Bot Fleet Status</h3>
                <div class="system-kpi-row">
                    <div class="system-kpi"><span class="label">Total Pairs</span><span class="value">${status.total_pairs || 0}</span></div>
                    <div class="system-kpi"><span class="label">Active</span><span class="value" style="color: var(--accent-success);">${status.active_pairs || 0}</span></div>
                </div>
                <div class="system-pairs-list">${pairsHtml || '<div class="empty-state-text">No pairs configured</div>'}</div>
            </div>
            <div class="system-section">
                <h3>API Costs</h3>
                <div class="system-costs">
                    <div class="cost-row"><span>OpenRouter</span><span class="cost-value">${formatCost(costByProvider.openrouter || 0)}</span></div>
                    <div class="cost-row"><span>Google</span><span class="cost-value">${formatCost(costByProvider.google || 0)}</span></div>
                    <div class="cost-row cost-total"><span>Total</span><span class="cost-value">${formatCost(totalCost)}</span></div>
                    <button id="btn-reset-costs" class="btn-primary" style="margin-top: 12px;">Reset Costs</button>
                </div>
            </div>
        </div>
    `;

    const resetBtn = document.getElementById('btn-reset-costs');
    if (resetBtn) {
        resetBtn.addEventListener('click', async () => {
            try {
                await fetch('/api/monitor/costs', { method: 'POST' });
                await updateSystemData();
            } catch (e) { console.error('Failed to reset costs:', e); }
        });
    }
}

function formatDuration(seconds) {
    if (seconds < 0) seconds = 0;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
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
