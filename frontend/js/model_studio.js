/**
 * model_studio.js — Model Zoo & Bot Registration Logic
 *
 * Fetches bot metadata from the REST API, renders model cards,
 * and handles bot registration, training, export, and deletion.
 */

const ModelStudio = {
    bots: [],
    _refreshInterval: null,

    /**
     * Fetch all bots from the API and render the grid.
     */
    async loadBots() {
        try {
            const res = await fetch('/api/bots');
            const data = await res.json();
            this.bots = data.bots || [];
            this.renderGrid();
        } catch (e) {
            console.error('[Studio] Failed to load bots:', e);
        }
    },

    /**
     * Start auto-refreshing the bot list every 3 seconds.
     */
    startAutoRefresh() {
        this.stopAutoRefresh();
        this.loadBots();
        this._refreshInterval = setInterval(() => this.loadBots(), 3000);
    },

    stopAutoRefresh() {
        if (this._refreshInterval) {
            clearInterval(this._refreshInterval);
            this._refreshInterval = null;
        }
    },

    /**
     * Render the model cards grid.
     */
    renderGrid() {
        const grid = document.getElementById('bots-grid');
        if (!grid) return;

        if (this.bots.length === 0) {
            grid.innerHTML = `
                <div class="bots-empty" style="grid-column: 1 / -1;">
                    <div class="bots-empty-icon">🤖</div>
                    <p>No bots registered yet.</p>
                    <p class="text-muted" style="margin-top:0.5rem; font-size:0.85rem;">
                        Click <strong>+ New Bot</strong> to create your first AI agent.
                    </p>
                </div>
            `;
            return;
        }

        grid.innerHTML = this.bots.map(bot => this._renderCard(bot)).join('');
    },

    _renderCard(bot) {
        const name = bot.display_name || bot.bot_name;
        const layers = (bot.layers || [128, 128]).join(' × ');
        const activation = (bot.activation || 'relu').toUpperCase();
        const lr = bot.learning_rate || 3e-4;
        const current = bot.current_step || 0;
        const total = bot.total_timesteps || 1;
        const pct = Math.min(100, (current / total) * 100);
        const status = bot.status || 'created';
        const isTraining = bot.is_training;
        const avgReward = bot.avg_reward || 0;
        const winRate = bot.win_rate || 0;
        const hasOnnx = bot.has_onnx;
        const hasModel = bot.has_model;

        const stepsFormatted = this._formatSteps(current);
        const totalFormatted = this._formatSteps(total);

        // Build action buttons based on state
        let actions = '';

        if (hasOnnx) {
            actions += `<button class="btn-sm play" onclick="ModelStudio.playBot('${bot.bot_name}')">▶ Play</button>`;
            actions += `<button class="btn-sm watch" onclick="ModelStudio.spectateBot('${bot.bot_name}')">📺 Watch</button>`;
        }

        if (hasModel) {
            actions += `<button class="btn-sm export" onclick="ModelStudio.exportOnnx('${bot.bot_name}')">📦 Export</button>`;
        }

        if (isTraining) {
            actions += `<button class="btn-sm stop" onclick="ModelStudio.stopTraining('${bot.bot_name}')">⏹ Stop</button>`;
        } else if (status !== 'completed') {
            const label = current > 0 ? '▶ Resume' : '🚀 Train';
            actions += `<button class="btn-sm train" onclick="ModelStudio.startTraining('${bot.bot_name}')">${label}</button>`;
        }

        actions += `<button class="btn-sm delete" onclick="ModelStudio.deleteBot('${bot.bot_name}')">🗑</button>`;

        return `
            <div class="bot-card" data-bot="${bot.bot_name}">
                <div class="bot-card-header">
                    <div class="bot-card-name">🧠 ${name}</div>
                    <span class="bot-card-status ${status}">${isTraining ? 'training' : status}</span>
                </div>
                <div class="bot-card-arch">[${layers} ${activation}]  lr: ${lr}</div>
                <div class="bot-card-progress">
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill ${status === 'completed' ? 'completed' : ''}" style="width: ${pct}%"></div>
                    </div>
                    <div class="progress-text">
                        <span>${stepsFormatted} / ${totalFormatted}</span>
                        <span>${pct.toFixed(1)}%</span>
                    </div>
                </div>
                <div class="bot-card-stats">
                    <div class="stat-item">
                        <span class="stat-label">Avg Reward</span>
                        <span class="stat-value">${avgReward >= 0 ? '+' : ''}${avgReward.toFixed(1)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Win Rate</span>
                        <span class="stat-value">${(winRate * 100).toFixed(0)}%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">ONNX</span>
                        <span class="stat-value">${hasOnnx ? '✅' : '—'}</span>
                    </div>
                </div>
                <div class="bot-card-actions">${actions}</div>
            </div>
        `;
    },

    _formatSteps(n) {
        if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
        if (n >= 1_000) return (n / 1_000).toFixed(0) + 'K';
        return String(n);
    },

    // ─── Actions ─────────────────────────────────────────────────────────

    async startTraining(botName) {
        try {
            const res = await fetch(`/api/bots/${botName}/train`, { method: 'POST' });
            const data = await res.json();
            if (!res.ok) {
                alert(data.detail || 'Failed to start training');
                return;
            }
            console.log(`[Studio] Training started for ${botName}`);
            this.loadBots();
        } catch (e) {
            console.error('[Studio] Start training failed:', e);
        }
    },

    async stopTraining(botName) {
        try {
            await fetch(`/api/bots/${botName}/stop`, { method: 'POST' });
            console.log(`[Studio] Training stopped for ${botName}`);
            this.loadBots();
        } catch (e) {
            console.error('[Studio] Stop training failed:', e);
        }
    },

    async exportOnnx(botName) {
        try {
            const res = await fetch(`/api/bots/${botName}/export`, { method: 'POST' });
            const data = await res.json();
            if (!res.ok) {
                alert(data.detail || 'Export failed');
                return;
            }
            alert(`✅ ONNX exported for ${botName}!`);
            this.loadBots();
        } catch (e) {
            console.error('[Studio] Export failed:', e);
        }
    },

    async deleteBot(botName) {
        if (!confirm(`Delete bot "${botName}" and all its files?`)) return;
        try {
            await fetch(`/api/bots/${botName}`, { method: 'DELETE' });
            console.log(`[Studio] Deleted ${botName}`);
            this.loadBots();
        } catch (e) {
            console.error('[Studio] Delete failed:', e);
        }
    },

    playBot(botName) {
        // Delegate to App to start a game against this trained bot
        App.startGameWithBot(botName, 'rl_bot');
    },

    spectateBot(botName) {
        // Delegate to App to start spectating this bot vs rule bot
        App.startGameWithBot(botName, 'spectate_rl_vs_rule');
    },

    // ─── Registration Form ───────────────────────────────────────────────

    initForm() {
        const layersEl = document.getElementById('reg-layers');
        const activationEl = document.getElementById('reg-activation');

        const updatePreview = () => {
            const layers = layersEl.value.split(',');
            const act = activationEl.value.toUpperCase();
            const mid = layers.map(l => `[${l.trim()} ${act}]`).join(' → ');
            document.getElementById('arch-preview').textContent = `obs(30) → ${mid} → action(5)`;
        };

        layersEl.addEventListener('change', updatePreview);
        activationEl.addEventListener('change', updatePreview);
        updatePreview();
    },

    async submitRegistration() {
        const name = document.getElementById('reg-name').value.trim();
        if (!name) {
            alert('Please enter a bot name');
            return false;
        }

        const layers = document.getElementById('reg-layers').value.split(',').map(Number);
        const activation = document.getElementById('reg-activation').value;
        const lr = parseFloat(document.getElementById('reg-lr').value);
        const steps = parseInt(document.getElementById('reg-steps').value);

        const payload = {
            bot_name: name,
            layers: layers,
            activation: activation,
            learning_rate: lr,
            total_timesteps: steps,
        };

        try {
            const res = await fetch('/api/bots', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) {
                alert(data.detail || 'Failed to create bot');
                return false;
            }
            console.log(`[Studio] Created bot: ${name}`);
            // Clear form
            document.getElementById('reg-name').value = '';
            return true;
        } catch (e) {
            console.error('[Studio] Registration failed:', e);
            alert('Network error');
            return false;
        }
    },
};
