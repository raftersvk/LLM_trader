/* Consolidated Overview Module */
class ConsolidatedOverview {
    constructor() {
        this.data = null;
        this.overlay = null;
        this.modal = null;
        this.content = null;
        this.initElements();
    }
    
    initElements() {
        // Create overlay and modal if they don't exist
        this.overlay = document.getElementById('consolidated-overlay');
        this.modal = document.getElementById('consolidated-modal');
        this.content = document.getElementById('consolidated-view');
        
        // Create elements if they don't exist
        if (!this.overlay) {
            this.overlay = document.createElement('div');
            this.overlay.id = 'consolidated-overlay';
            this.overlay.className = 'overlay';
            this.overlay.onclick = () => this.close();
            document.body.appendChild(this.overlay);
        }
        
        if (!this.modal) {
            this.modal = document.createElement('div');
            this.modal.id = 'consolidated-modal';
            this.modal.className = 'modal';
            
            const modalContent = document.createElement('div');
            modalContent.className = 'modal-content';
            
            const closeBtn = document.createElement('span');
            closeBtn.className = 'close';
            closeBtn.innerHTML = '&times;';
            closeBtn.onclick = () => this.close();
            
            this.content = document.createElement('div');
            this.content.id = 'consolidated-content';
            
            modalContent.appendChild(closeBtn);
            modalContent.appendChild(this.content);
            this.modal.appendChild(modalContent);
            document.body.appendChild(this.modal);
        }
    }
    
    async load() {
        try {
            const response = await fetch('/api/monitor/status/all');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.data = await response.json();
            this.render();
            this.open();
        } catch (error) {
            console.error('Failed to load consolidated overview:', error);
            this.content.innerHTML = '<div class="error">Failed to load data: ' + error.message + '</div>';
            this.open();
        }
    }
    
    render() {
        if (!this.data) {
            this.content.innerHTML = '<div class="loading">Loading data...</div>';
            return;
        }
        
        let totalPnL = 0;
        let totalTrades = 0;
        let winningTrades = 0;
        let activePositions = 0;
        let totalPairs = this.data.total_pairs || 0;
        let activePairs = this.data.active_pairs || 0;
        
        // Process pairs data for consolidated stats
        if (this.data.pairs) {
            Object.entries(this.data.pairs).forEach(([_symbol, state]) => {
                // We would ideally get P&L from stats, but for now we'll use what's available
                // In a real implementation, we'd fetch stats for each pair
                if (state.current_position) {
                    activePositions++;
                }
            });
        }
        
        // Get consolidated stats if available
        if (this.data.consolidated) {
            totalPnL = this.data.consolidated.total_pnl_quote || 0;
            totalTrades = this.data.consolidated.total_trades || 0;
            winningTrades = this.data.consolidated.winning_trades || 0;
        }
        
        const winRate = totalTrades > 0 ? (winningTrades / totalTrades * 100) : 0;
        
        this.content.innerHTML = `
            <div class="overview-header">
                <h2>Vue Globale - ${totalPairs} Paires</h2>
                <div class="summary-cards">
                    <div class="card">
                        <span class="label">Paires Actives</span>
                        <span class="value">${activePairs}</span>
                    </div>
                    <div class="card">
                        <span class="label">Positions Ouvertes</span>
                        <span class="value">${activePositions}</span>
                    </div>
                    <div class="card">
                        <span class="label">P&L Total</span>
                        <span class="value">${totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)} USDT</span>
                    </div>
                    <div class="card">
                        <span class="label">Trades Total</span>
                        <span class="value">${totalTrades}</span>
                    </div>
                    <div class="card">
                        <span class="label">Taux de Réussite</span>
                        <span class="value">${winRate.toFixed(1)}%</span>
                    </div>
                </div>
            </div>
            <div class="table-container">
                <table class="pairs-table">
                    <thead>
                        <tr>
                            <th>Paire</th>
                            <th>Prix Actuel</th>
                            <th>Position</th>
                            <th>P&L Non Réalisé</th>
                            <th>Statut</th>
                            <th>Dernière Analyse</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.data.pairs ? Object.entries(this.data.pairs).map(([symbol, state]) => {
                            const pos = state.current_position || {};
                            const pnl = pos.unrealized_pnl !== undefined ? pos.unrealized_pnl : 0;
                            return `
                                <tr>
                                    <td><strong>${symbol}</strong></td>
                                    <td class="price">$${state.current_price ? state.current_price.toLocaleString() : '—'}</td>
                                    <td class="position">${pos.direction ? `${pos.direction} ${pos.size || ''} ${symbol.split('/')[0]}` : 'Aucune'}</td>
                                    <td class="pnl ${pnl >= 0 ? 'positive' : 'negative'}">${pnl !== 0 ? (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + ' USDT' : '—'}</td>
                                    <td class="status">${state.is_running ? '🟢 En Cours' : '🔴 Arrêté'}</td>
                                    <td>${state.last_analysis_time ? new Date(state.last_analysis_time).toLocaleTimeString() : '—'}</td>
                                </tr>
                            `;
                        }).join('') : '<tr><td colspan="6">Aucune donnée disponible</td></tr>'}
                    </tbody>
                </table>
            </div>
        `;
    }
    
    open() {
        if (this.overlay) this.overlay.style.display = 'block';
        if (this.modal) this.modal.style.display = 'block';
        document.body.style.overflow = 'hidden'; // Prevent scrolling when modal is open
    }
    
    close() {
        if (this.overlay) this.overlay.style.display = 'none';
        if (this.modal) this.modal.style.display = 'none';
        document.body.style.overflow = 'auto'; // Restore scrolling
    }
}

// Initialize the consolidated overview when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.consolidatedOverview = new ConsolidatedOverview();
});