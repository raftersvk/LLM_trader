// TEMPORARY SIMPLIFIED VERSION - testing imports only
import { initPortfolioOverview, updatePortfolioData } from './modules/portfolio_overview.js?v=1.0';
import { initSystemMonitor, updateSystemData } from './modules/system_monitor.js?v=1.0';

console.log('[DEBUG] All imports completed successfully');
console.log('[DEBUG] initPortfolioOverview:', typeof initPortfolioOverview);
console.log('[DEBUG] initSystemMonitor:', typeof initSystemMonitor);

function initApp() {
    console.log('[DEBUG] initApp called');
    console.log('[DEBUG] Would call initPortfolioOverview and initSystemMonitor here');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}