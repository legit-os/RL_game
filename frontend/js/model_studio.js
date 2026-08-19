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
        const matches = bot.matches_played || 0;
        const level = bot.curriculum_level !== undefined ? bot.curriculum_level : 1;
        const levelWinRate = bot.level_win_rate !== undefined ? Number(bot.level_win_rate).toFixed(1) : (bot.win_rate ? (bot.win_rate * 100).toFixed(1) : '0.0');
        const levelMatches = bot.level_matches || 0;

        const levelTitle = level === 0 ? 'Gen 0 (Imitation)' : `Gen ${level}`;

        const matchesFormatted = this._formatSteps(matches);

        // Build action buttons based on state
        let actions = '';

        if (hasOnnx) {
            actions += `<button class="btn-sm play" onclick="ModelStudio.playBot('${bot.bot_name}')">▶ Play</button>`;
            actions += `
                <button class="btn-sm watch" onclick="ModelStudio.openTestModal('${bot.bot_name}')">⚔️ Test</button>
            `;
        }

        if (hasModel) {
            actions += `<button class="btn-sm export" onclick="ModelStudio.exportOnnx('${bot.bot_name}')">📦 Export</button>`;
        }
        
        // Graph button always available if it has generated any logs
        actions += `<button class="btn-sm" style="background: #475569; color: white;" onclick="ModelStudio.viewGraph('${bot.bot_name}')">📈 Graph</button>`;

        if (isTraining) {
            actions += `<button class="btn-sm" style="background: #3b82f6; color: white;" onclick="ModelStudio.watchTraining('${bot.bot_name}')">👁️ Watch</button>`;
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
                    <div style="display: flex; gap: 0.4rem; align-items: center;">
                        <span class="level-badge" style="background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.35); font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 12px;">🏆 Gen ${level}</span>
                        <span class="bot-card-status ${status}">${isTraining ? 'training' : status}</span>
                    </div>
                </div>
                <div class="bot-card-arch">[${layers} ${activation}]  lr: ${lr}</div>
                
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 0.5rem 0.7rem; margin-bottom: 0.8rem; font-size: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #94a3b8; font-weight: 600;">Self-Play Pool:</span>
                        <span style="color: #f1f5f9; font-weight: 700;">${levelTitle} (${5 + Math.max(0, level - 1)} opponents)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #94a3b8;">Pool Win Rate (Expand at 75%):</span>
                        <span style="color: ${Number(levelWinRate) >= 75 ? '#4ade80' : '#38bdf8'}; font-weight: 700;">${levelWinRate}% <span style="color: #64748b; font-size: 0.7rem;">(${levelMatches}/100)</span></span>
                    </div>
                </div>

                <div class="bot-card-progress">
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill ${status === 'completed' ? 'completed' : ''}" style="width: ${pct}%"></div>
                    </div>
                    <div class="progress-text">
                        <span>${matchesFormatted} Matches Played</span>
                        <span>${pct.toFixed(1)}%</span>
                    </div>
                </div>
                <div class="bot-card-stats">
                    <div class="stat-item">
                        <span class="stat-label">Generation</span>
                        <span class="stat-value">🏆 Gen ${level}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Pool Win Rate</span>
                        <span class="stat-value" style="color: ${Number(levelWinRate) >= 75 ? '#4ade80' : '#38bdf8'}">${levelWinRate}%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Reward</span>
                        <span class="stat-value">${avgReward >= 0 ? '+' : ''}${avgReward.toFixed(1)}</span>
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
        App.startGameWithBot(botName, 'rl_bot');
    },

    openTestModal(botName) {
        this.testBotTarget = botName;
        
        // Find bot and populate snapshots
        const bot = this.bots.find(b => (b.bot_name === botName || b.display_name === botName));
        const snapshotSelect = document.getElementById('test-snapshot-select');
        if (bot && bot.snapshots && bot.snapshots.length > 0) {
            snapshotSelect.innerHTML = bot.snapshots.map(s => {
                const label = s === 'model.onnx' ? 'model.onnx (Latest)' : s;
                return `<option value="${s}">${label}</option>`;
            }).join('');
            
            // Default to model.onnx if available, otherwise the last level
            const latest = bot.snapshots.includes('model.onnx') ? 'model.onnx' : bot.snapshots[bot.snapshots.length - 1];
            snapshotSelect.value = latest;
        } else {
            snapshotSelect.innerHTML = '<option value="model.onnx">model.onnx (Latest)</option>';
        }

        document.getElementById('opponent-select-modal').classList.add('active');
    },

    launchTest(opponent) {
        if (!this.testBotTarget) return;
        const snapshot = document.getElementById('test-snapshot-select').value;
        document.getElementById('opponent-select-modal').classList.remove('active');
        App.startGameWithBot(this.testBotTarget, 'spectate_rl_vs_rule', opponent, snapshot);
    },

    watchTraining(botName) {
        App.startGameWithBot(botName, 'spectate_training');
    },

    // ─── Registration Form ───────────────────────────────────────────────

    async initForm() {
        const layersEl = document.getElementById('reg-layers');
        const activationEl = document.getElementById('reg-activation');
        const baseModelEl = document.getElementById('reg-base-model');
        const levelEl = document.getElementById('reg-level');

        // Fetch imitation models
        try {
            const res = await fetch('/api/imitation/models');
            const data = await res.json();
            if (data.models && data.models.length > 0) {
                window.imitationModels = data.models; // Store globally for access
                baseModelEl.innerHTML = data.models.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
                setTimeout(() => baseModelEl.dispatchEvent(new Event('change')), 10);
            } else {
                baseModelEl.innerHTML = `<option value="">-- No Imitation Models Found --</option>`;
            }
        } catch (e) {
            console.error('[Studio] Failed to load imitation models:', e);
        }

        const updatePreview = () => {
            const layers = layersEl.value.split(',');
            const act = activationEl.value.toUpperCase();
            const mid = layers.map(l => `[${l.trim()} ${act}]`).join(' → ');
            document.getElementById('arch-preview').textContent = `obs(30) → ${mid} → action(5)`;
        };

        const updateBaseModelConstraints = () => {
            const isImitationBase = parseInt(levelEl.value) === 0;
            const selectedName = baseModelEl.value;
            
            // Handle visibility based on level
            document.getElementById('base-model-group').style.display = isImitationBase ? 'block' : 'none';

            if (isImitationBase && window.imitationModels && selectedName) {
                const selectedModel = window.imitationModels.find(m => m.name === selectedName);
                if (selectedModel) {
                    const layerStr = selectedModel.layers.join(',');
                    
                    // Add option if it doesn't exist
                    let layerOption = Array.from(layersEl.options).find(opt => opt.value === layerStr);
                    if (!layerOption) {
                        layerOption = document.createElement('option');
                        layerOption.value = layerStr;
                        layerOption.text = selectedModel.layers.join(' × ');
                        layersEl.add(layerOption);
                    }
                    
                    layersEl.value = layerStr;
                    activationEl.value = selectedModel.activation.toLowerCase();
                    
                    layersEl.disabled = true;
                    activationEl.disabled = true;
                    updatePreview();
                    return;
                }
            }
            
            layersEl.disabled = false;
            activationEl.disabled = false;
            updatePreview();
        };

        baseModelEl.addEventListener('change', updateBaseModelConstraints);
        levelEl.addEventListener('change', updateBaseModelConstraints);
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
        const useLrScheduler = document.getElementById('reg-lr-scheduler').checked;
        const steps = parseInt(document.getElementById('reg-steps').value);

        const level = parseInt(document.getElementById('reg-level').value) || 0;
        const baseModel = level === 0 ? document.getElementById('reg-base-model').value : "";

        // PPO Hyperparameters
        const clipRange = parseFloat(document.getElementById('reg-clip-range').value) || 0.2;
        const entCoef = parseFloat(document.getElementById('reg-ent-coef').value) || 0.0;
        const gamma = parseFloat(document.getElementById('reg-gamma').value) || 0.99;
        const targetKlVal = document.getElementById('reg-target-kl').value.trim();
        const targetKl = targetKlVal ? parseFloat(targetKlVal) : null;

        const payload = {
            bot_name: name,
            curriculum_level: baseModel ? 3 : 1,
            base_model: baseModel,
            layers: layers,
            activation: activation,
            learning_rate: lr,
            use_lr_scheduler: useLrScheduler,
            total_timesteps: steps,
            clip_range: clipRange,
            ent_coef: entCoef,
            gamma: gamma,
            target_kl: targetKl
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
    
    // Globals for chart instances
    _chartInstances: [],

    async viewGraph(botName) {
        // Show modal
        const modal = document.getElementById('graph-modal');
        document.getElementById('graph-modal-title').innerText = `Training Progress: ${botName}`;
        modal.classList.add('active');
        
        try {
            const res = await fetch(`/api/bots/${botName}/progress`);
            const data = await res.json();
            
            const container = document.getElementById('charts-container');
            container.innerHTML = ''; // Clear existing
            
            if (this._chartInstances && this._chartInstances.length > 0) {
                this._chartInstances.forEach(c => c.destroy());
            }
            this._chartInstances = [];
            
            if (!data || data.length === 0) {
                container.innerHTML = '<div style="color: #94a3b8; text-align: center; width: 100%; padding: 2rem;">No progress data yet.</div>';
                return;
            }
            
            // Aggregate data by unique timesteps to handle split log lines
            const aggregated = {};
            data.forEach(row => {
                const step = row["time/total_timesteps"];
                if (step === undefined) return;
                if (!aggregated[step]) aggregated[step] = {};
                Object.keys(row).forEach(key => {
                    aggregated[step][key] = row[key];
                });
            });
            
            const steps = Object.keys(aggregated).map(Number).sort((a,b) => a - b);
            const metricsMap = {}; 
            const excludeKeys = new Set(['time/total_timesteps', 'time/iterations', 'time/time_elapsed', 'time/fps']);
            
            steps.forEach((step, idx) => {
                const row = aggregated[step];
                Object.keys(row).forEach(key => {
                    if (excludeKeys.has(key)) return;
                    if (!metricsMap[key]) {
                        metricsMap[key] = new Array(steps.length).fill(null);
                    }
                    metricsMap[key][idx] = row[key];
                });
            });
            
            // Define some nice colors
            const colors = ['#34d399', '#f87171', '#60a5fa', '#fbbf24', '#c084fc', '#fb923c', '#2dd4bf'];
            let colorIdx = 0;
            
            // Create a chart for each metric
            Object.keys(metricsMap).sort().forEach(key => {
                // Create wrapper
                const wrapper = document.createElement('div');
                wrapper.style.background = 'rgba(0,0,0,0.2)';
                wrapper.style.borderRadius = '8px';
                wrapper.style.padding = '1rem';
                wrapper.style.border = '1px solid rgba(255,255,255,0.05)';
                
                const title = document.createElement('h3');
                title.style.margin = '0 0 1rem 0';
                title.style.fontSize = '0.9rem';
                title.style.color = '#cbd5e1';
                title.innerText = key;
                wrapper.appendChild(title);
                
                const canvasWrapper = document.createElement('div');
                canvasWrapper.style.position = 'relative';
                canvasWrapper.style.height = '250px';
                canvasWrapper.style.width = '100%';
                
                const canvas = document.createElement('canvas');
                canvasWrapper.appendChild(canvas);
                wrapper.appendChild(canvasWrapper);
                container.appendChild(wrapper);
                
                const ctx = canvas.getContext('2d');
                // A quick hack to ensure semi-transparent fill: 
                const colorHex = colors[colorIdx % colors.length];
                const r = parseInt(colorHex.slice(1,3), 16);
                const g = parseInt(colorHex.slice(3,5), 16);
                const b = parseInt(colorHex.slice(5,7), 16);
                const colorFill = `rgba(${r}, ${g}, ${b}, 0.1)`;
                colorIdx++;
                
                const chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: steps,
                        datasets: [{
                            label: key,
                            data: metricsMap[key],
                            borderColor: colorHex,
                            backgroundColor: colorFill,
                            fill: true,
                            spanGaps: true,
                            tension: 0.2,
                            borderWidth: 2,
                            pointRadius: 3, // Increased so single points are highly visible!
                            pointHoverRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { intersect: false, mode: 'index' },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                titleColor: '#fff',
                                bodyColor: '#cbd5e1',
                                borderColor: 'rgba(255,255,255,0.1)',
                                borderWidth: 1
                            }
                        },
                        scales: {
                            x: {
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: {
                                    color: '#94a3b8',
                                    callback: function(val, index) {
                                        return ModelStudio._formatSteps(this.getLabelForValue(val));
                                    }
                                }
                            },
                            y: {
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: { color: '#94a3b8' }
                            }
                        }
                    }
                });
                
                this._chartInstances.push(chart);
            });
            
        } catch (e) {
            console.error('[Studio] Failed to load graph:', e);
        }
    }
};
