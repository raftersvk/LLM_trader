import { updatePerformanceData } from './modules/performance_chart.js?v=5.1';
import { initSynapseNetwork, updateSynapses } from './modules/synapse_viewer.js?v=5.1';
import { updateLogs } from './modules/log_viewer.js?v=5.1';
import { updateVisuals } from './modules/visuals.js?v=5.1';
import { initVectorPanel, updateVectorData } from './modules/vector_panel.js?v=5.1';
import { initFullscreen } from './modules/fullscreen.js?v=5.1';
import { initWebSocket, startCountdownLoop } from './modules/websocket.js?v=5.1';
import { initPositionPanel, updatePositionData } from './modules/position_panel.js?v=5.1';
import { initUI } from './modules/ui.js?v=5.1';
import { initStatisticsPanel, updateStatisticsData } from './modules/statistics_panel.js?v=5.1';
import { initNewsPanel, updateNewsData } from './modules/news_panel.js?v=5.1';
import { initPortfolioOverview, updatePortfolioData } from './modules/portfolio_overview.js?v=5.1';
import { initPerformancePanel, updatePerformancePanel } from './modules/performance_panel.js?v=5.1';
import { initSystemMonitor, updateSystemData } from './modules/system_monitor.js?v=5.1';
import { initPairDetail, updatePairDetail } from './modules/pair_detail.js?v=5.1';
import './modules/overview.js?v=5.1';

const state = {
    isConnected: false,
    fastPollInterval: 10000,
    slowPollInterval: 30000,
    lastUpdateTime: null,
    currentPair: null
};
window.state = state;

function formatCost(cost) {
    if (cost === 0 || cost === null || cost === undefined) return '$0.00';
    if (cost < 0.0001) return `$${cost.toFixed(8)}`;
    if (cost < 0.01) return `$${cost.toFixed(6)}`;
    if (cost < 1) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
}

async function fetchCosts() {
    try {
        const response = await fetch('/api/monitor/costs');
        const data = await response.json();
        updateCostDisplay(data);
    } catch (e) {
        console.error("Failed to fetch costs", e);
    }
}

function updateCostDisplay(data) {
    const total = data.total_session_cost || 0;

    // Update main total - fix: element id is 'portfolio-cost' not 'overview-cost'
    const costEl = document.getElementById('portfolio-cost');
    if (costEl) costEl.textContent = formatCost(total);
}



async function fetchBrainStatus() {
    try {
        const url = state.currentPair && state.currentPair !== '__global__'
            ? `/api/brain/status?symbol=${encodeURIComponent(state.currentPair)}`
            : '/api/brain/status';
        const response = await fetch(url);
        const data = await response.json();
        const connStatus = document.getElementById('connection-status');
        const statusDot = document.querySelector('.status-dot');

        if (connStatus) {
            connStatus.textContent = 'Connected';
            connStatus.classList.remove('status-text', 'disconnected');
            connStatus.classList.add('status-text', 'connected');
            connStatus.style.color = ''; // Clear inline style if present
        }

        if (statusDot) {
            statusDot.classList.remove('disconnected');
            statusDot.classList.add('connected');
        }

        // Update Brain State Indicator
        // (Legacy indicator removed from UI, skipping update)

        // Direct update to Overview KPIs
        const trendEl = document.getElementById('overview-trend');
        if (trendEl) {
            trendEl.textContent = data.trend || '--';
            if (data.trend === 'BULLISH') {
                trendEl.className = 'value start-green';
            } else if (data.trend === 'BEARISH') {
                trendEl.className = 'value start-red';
            } else {
                trendEl.className = 'value'; // default color
            }
        }

        const confEl = document.getElementById('overview-conf');
        if (confEl) {
            confEl.textContent = data.confidence ? `${data.confidence}%` : '--%';
        }

        const actionEl = document.getElementById('overview-action');
        if (actionEl) {
            actionEl.textContent = data.action || 'WAITING';
            // Styling logic for action (sub-label text color usually muted, but user had color logic)
            // The previous logic colored the text. Let's keep it consistent if possible, 
            // but usually sub-labels are muted. The legacy code colored #action-val.
            // syncStatus copied text content, but NOT style. 
            // WAIT - syncStatus did NOT copy style from action-val to overview-action.
            // It only did: document.getElementById('overview-action').textContent = act;
            // So visible UI was NOT colored. I will stick to text content to match visible behavior.
        }

        state.lastUpdateTime = new Date();
        updateLastUpdated();
    } catch (e) {
        const connStatus = document.getElementById('connection-status');
        const statusDot = document.querySelector('.status-dot');

        if (connStatus) {
            connStatus.textContent = 'Disconnected';
            connStatus.classList.remove('status-text', 'connected');
            connStatus.classList.add('status-text', 'disconnected');
            connStatus.style.color = ''; // Clear inline style if present
        }

        if (statusDot) {
            statusDot.classList.remove('connected');
            statusDot.classList.add('disconnected');
        }
    }
}

async function fetchRules() {
    try {
        const response = await fetch('/api/brain/rules');
        const rules = await response.json();

        // Direct update to Rules Count KPI
        const countEl = document.getElementById('overview-rules-count');
        const hintEl = document.getElementById('overview-rules-hint');

        if (countEl) {
            const count = rules.length;
            countEl.textContent = count;

            if (hintEl) {
                hintEl.style.display = count > 0 ? 'none' : 'block';
            }
        }

    } catch (e) {
        console.error("Failed to fetch rules", e);
    }
}

function updateLastUpdated() {
    const el = document.getElementById('last-updated');
    if (el && state.lastUpdateTime) {
        el.textContent = `Updated: ${new Intl.DateTimeFormat(navigator.language, { timeStyle: 'medium' }).format(state.lastUpdateTime)}`;
    }
}

function togglePanelMinimize(panelId) {
    const panel = document.getElementById(panelId);
    if (panel) {
        const isMinimized = panel.classList.toggle('minimized');
        const btn = panel.querySelector('.toolbar-btn[title="Minimize"], .toolbar-btn[title="Expand"]');
        if (btn) {
            btn.innerHTML = isMinimized
                ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>'
                : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line></svg>';
            btn.title = isMinimized ? 'Expand' : 'Minimize';
            btn.setAttribute('aria-expanded', String(!isMinimized));

            const titleEl = panel.querySelector('h3');
            const panelName = titleEl ? titleEl.textContent.toLowerCase() : 'panel';
            btn.setAttribute('aria-label', isMinimized ? `Expand ${panelName}` : `Minimize ${panelName}`);
        }
    }
}

async function updateAll() {
    await updateFastLane();
    await updateSlowLane();
}

async function updateFastLane() {
    await fetchBrainStatus();
    await updatePositionData();
}

async function updateSlowLane() {
    await fetchRules();
    await fetchCosts();
    await updatePerformanceData();
    await updateSynapses();
    await updateLogs();
    await updateVisuals();
    await updateVectorData();
    await updateStatisticsData();
    await updateNewsData();
    await updatePortfolioData();
    await updatePerformancePanel();
    await updateSystemData();
    await updatePairDetail();
}

// Initialize application
function initApp() {
    console.log('Initializing Dashboard App...');

    // Make crucial functions global immediately
    window.updateAll = updateAll;
    window.updateVectorData = updateVectorData;
    window.updatePerformanceData = updatePerformanceData;
    window.updatePerformancePanel = updatePerformancePanel;

    window.togglePanelMinimize = togglePanelMinimize;

    // Mobile menu is fully managed by setupMobileMenu() in modules/ui.js

    try {
        // Event listeners for static buttons
        const btnMinimize = document.getElementById('btn-minimize-visuals');
        if (btnMinimize) {
            btnMinimize.addEventListener('click', () => togglePanelMinimize('panel-visuals'));
        }

        const btnLightbox = document.getElementById('btn-lightbox-visuals');
        if (btnLightbox) {
            btnLightbox.addEventListener('click', () => {
                const img = document.getElementById('analysis-chart');
                if (img && img.src && window.openLightbox) window.openLightbox(img.src);
            });
        }

        const btnCopyPrompt = document.getElementById('btn-copy-prompt');
        if (btnCopyPrompt && window.copyPromptContent) {
            btnCopyPrompt.addEventListener('click', window.copyPromptContent);
        }

        const btnCopyResponse = document.getElementById('btn-copy-response');
        if (btnCopyResponse && window.copyResponseContent) {
            btnCopyResponse.addEventListener('click', window.copyResponseContent);
        }

        initSynapseNetwork();
        initVectorPanel();
        initFullscreen();
        initPositionPanel();
        initStatisticsPanel();
        initNewsPanel();
        initPortfolioOverview();
        initPerformancePanel();
        initSystemMonitor();
        initPairDetail();
        initWebSocket();
        initUI();
        startCountdownLoop();

        // Initial update
        updateAll();

        // Start polling lanes: fast for critical status/position, slow for heavier panels.
        setInterval(updateFastLane, state.fastPollInterval);
        setInterval(updateSlowLane, state.slowPollInterval);

        // Listen for WS analysis complete
        document.addEventListener('analysis-complete', () => {
            console.log('Analysis complete, refreshing...');
            updateFastLane();
            updateSlowLane();
        });

        // Listen for pair change and refresh all panels
        document.addEventListener('pair-changed', (e) => {
            console.log('Pair changed to:', e.detail.symbol, '- refreshing all panels...');
            updateAll();
            if (window.updatePairDetail) window.updatePairDetail();
            if (window.updateVectorData) window.updateVectorData();
            if (window.updatePerformanceData) window.updatePerformanceData();
            if (window.updatePerformancePanel) window.updatePerformancePanel();
        });

        // Initialize pair selector and consolidated view button
        const pairSelect = document.getElementById('pair-select');
        const activePairsBadge = document.getElementById('active-pairs-badge');

        if (pairSelect && activePairsBadge) {
            console.log('Fetching pairs from /api/monitor/pairs...');
            fetch('/api/monitor/pairs')
                    .then(response => {
                        console.log('Response status:', response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log('Pairs data received:', JSON.stringify(data));
                        let pairs = data.pairs;
                        // Ensure pairs is an array
                        if (!Array.isArray(pairs)) {
                            console.warn('Pairs data is not an array, converting:', pairs);
                            if (typeof pairs === 'object' && pairs !== null) {
                                pairs = Object.keys(pairs);
                            } else {
                                pairs = [];
                            }
                        }
                        if (pairs.length > 0) {
                            pairs.forEach(pair => {
                                const option = document.createElement('option');
                                option.value = pair;
                                option.textContent = pair;
                                pairSelect.appendChild(option);
                            });

                            pairSelect.value = pairs[0];
                            state.currentPair = pairs[0];

                            activePairsBadge.textContent = `${pairs.length} pairs`;

                            pairSelect.addEventListener('change', (e) => {
                                const newPair = e.target.value;
                                if (newPair === '__global__') {
                                    if (window.consolidatedOverview) {
                                        window.consolidatedOverview.load();
                                    }
                                    return;
                                }
                                if (newPair && newPair !== state.currentPair) {
                                    state.currentPair = newPair;
                                    document.dispatchEvent(new CustomEvent('pair-changed', {
                                        detail: { symbol: newPair }
                                    }));
                                }
                            });
                        } else {
                            activePairsBadge.textContent = 'No pairs';
                        }
                    })
                    .catch(err => {
                        console.error('Failed to load pairs:', err);
                        activePairsBadge.textContent = 'Error';
                    });
        }

        console.log('Dashboard App Initialized');
    } catch (e) {
        console.error('Error initializing dashboard:', e);
    }
}

// Run init when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
