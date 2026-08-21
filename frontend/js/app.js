/**
 * app.js — Main Application State Machine
 *
 * Manages screen transitions: MENU → STUDIO → REGISTER → GAME → RESULT
 * Initializes all modules and runs the render loop.
 */

const App = {
    currentScreen: 'menu',
    currentState: null,
    isSpectating: false,
    selectedBotName: null,
    _animFrameId: null,
    _gameStartTick: 0,

    init() {
        // Initialize modules
        Controls.init();
        Renderer.init();
        ModelStudio.initForm();
        TrainingHUD.start();

        // ─── Menu buttons ─────────────────────────────────────────────
        document.getElementById('btn-rule-bot').addEventListener('click', () => {
            this.startGame('rule_bot');
        });
        document.getElementById('btn-rl-bot').addEventListener('click', () => {
            this.openRLAgentModal();
        });

        // ─── RL Agent Modal buttons ───────────────────────────────────
        document.getElementById('btn-start-rl-game').addEventListener('click', () => {
            const botName = document.getElementById('rl-agent-bot-select').value;
            const snapshot = document.getElementById('rl-agent-snapshot-select').value;
            if (botName) {
                document.getElementById('rl-agent-select-modal').classList.remove('active');
                // Play against the selected RL agent
                this.startGameWithBot(botName, 'rl_bot', null, snapshot);
            }
        });
        document.getElementById('btn-studio').addEventListener('click', () => {
            this.showScreen('studio');
            ModelStudio.startAutoRefresh();
        });
        document.getElementById('btn-imitation').addEventListener('click', () => {
            this.showScreen('imitation');
        });

        // ─── Imitation Learning buttons ───────────────────────────────
        document.getElementById('btn-back-menu-imitation').addEventListener('click', () => {
            this.showScreen('menu');
        });
        document.getElementById('btn-play-record').addEventListener('click', () => {
            const botSelect = document.getElementById('il-bot-select');
            const selectedBot = botSelect.value;
            this.startGameWithRecording(selectedBot);
        });

        // ─── Studio buttons ───────────────────────────────────────────
        document.getElementById('btn-back-menu').addEventListener('click', () => {
            ModelStudio.stopAutoRefresh();
            this.showScreen('menu');
        });
        document.getElementById('btn-register-bot').addEventListener('click', () => {
            ModelStudio.stopAutoRefresh();
            this.showScreen('register');
        });

        // ─── Register form buttons ────────────────────────────────────
        document.getElementById('btn-cancel-register').addEventListener('click', () => {
            this.showScreen('studio');
            ModelStudio.startAutoRefresh();
        });
        document.getElementById('btn-submit-register').addEventListener('click', async () => {
            const success = await ModelStudio.submitRegistration();
            if (success) {
                this.showScreen('studio');
                ModelStudio.startAutoRefresh();
            }
        });

        // ─── Result screen buttons ────────────────────────────────────
        document.getElementById('btn-play-again').addEventListener('click', () => {
            this._updateStatus('Ready', 'connected');
            this.showScreen('menu');
        });
        document.getElementById('btn-back-studio').addEventListener('click', () => {
            this.showScreen('studio');
            ModelStudio.startAutoRefresh();
        });

        // ─── Imitation recording result actions ───────────────────────
        document.getElementById('btn-save-recording').addEventListener('click', async () => {
            const statusEl = document.getElementById('result-save-status');
            statusEl.style.display = 'block';
            statusEl.style.color = '#38bdf8';
            statusEl.textContent = 'Saving match recording...';
            try {
                const res = await fetch('/api/imitation/save', { method: 'POST' });
                const data = await res.json();
                if (res.ok) {
                    statusEl.style.color = '#4ade80';
                    statusEl.textContent = `✅ Saved ${data.steps} steps to ${data.filename}!`;
                    document.getElementById('btn-save-recording').disabled = true;
                    document.getElementById('btn-discard-recording').disabled = true;
                } else {
                    statusEl.style.color = '#f87171';
                    statusEl.textContent = `❌ ${data.detail || 'Failed to save'}`;
                }
            } catch (e) {
                statusEl.style.color = '#f87171';
                statusEl.textContent = '❌ Network error saving recording';
            }
        });

        document.getElementById('btn-discard-recording').addEventListener('click', async () => {
            const statusEl = document.getElementById('result-save-status');
            statusEl.style.display = 'block';
            statusEl.style.color = '#94a3b8';
            statusEl.textContent = 'Discarding recording...';
            try {
                await fetch('/api/imitation/discard', { method: 'POST' });
                statusEl.textContent = '🗑️ Recording discarded.';
                document.getElementById('btn-save-recording').disabled = true;
                document.getElementById('btn-discard-recording').disabled = true;
            } catch (e) {
                statusEl.textContent = '🗑️ Recording discarded.';
            }
        });

        // ─── Training bar watch button ────────────────────────────────
        document.getElementById('training-bar-watch').addEventListener('click', () => {
            const botName = TrainingHUD.getActiveBotName();
            if (botName) {
                this.startGameWithBot(botName, 'spectate_training');
            }
        });

        // Show menu
        this._updateStatus('Ready', 'connected');
    },

    showScreen(name) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        const screen = document.getElementById(name + '-screen');
        if (screen) {
            screen.classList.add('active');
        }
        this.currentScreen = name;
    },

    async openRLAgentModal() {
        const modal = document.getElementById('rl-agent-select-modal');
        const botSelect = document.getElementById('rl-agent-bot-select');
        
        botSelect.innerHTML = '<option>Loading...</option>';
        modal.classList.add('active');

        try {
            const res = await fetch('/api/bots');
            const data = await res.json();
            const trainedBots = (data.bots || []).filter(b => b.has_onnx);
            
            if (trainedBots.length === 0) {
                botSelect.innerHTML = '<option value="">No trained bots found</option>';
                return;
            }

            botSelect.innerHTML = trainedBots.map(b => `<option value="${b.bot_name}">${b.bot_name}</option>`).join('');
        } catch (e) {
            botSelect.innerHTML = '<option value="">Error loading bots</option>';
        }
    },

    /**
     * Start a game with the default model path (legacy mode).
     */
    startGame(mode) {
        this.selectedBotName = null;
        this._launchGame(mode, null, false);
    },

    /**
     * Start a game or spectate session with a specific trained bot.
     */
    startGameWithBot(botName, mode, opponent = null, snapshot = null) {
        this.selectedBotName = botName;
        this._launchGame(mode, botName, false, opponent, snapshot);
    },

    /**
     * Start an Imitation Learning recording session against a rule bot.
     */
    startGameWithRecording(ruleBotType) {
        this.selectedBotName = ruleBotType;
        this._launchGame('record', ruleBotType, true);
    },

    _launchGame(mode, botName, isRecording = false, opponent = null, snapshot = null) {
        this.isSpectating = mode.startsWith('spectate');
        this.isRecording = isRecording;

        // Update HUD labels
        const playerLabel = document.getElementById('hud-label-player');
        const enemyLabel = document.getElementById('hud-label-enemy');
        if (this.isSpectating) {
            playerLabel.textContent = botName ? botName.toUpperCase() : 'RL BOT';
            enemyLabel.textContent = 'RULE BOT';
        } else {
            playerLabel.textContent = 'YOU';
            enemyLabel.textContent = botName ? botName.toUpperCase() : 'BOT';
        }

        // Configure renderer
        Renderer.spectateMode = this.isSpectating;
        Renderer.playerLabel = playerLabel.textContent;
        Renderer.enemyLabel = enemyLabel.textContent;

        // Show/hide joystick zones and spectator banner
        const jLeft = document.getElementById('joystick-left');
        const jRight = document.getElementById('joystick-right');
        const specBanner = document.getElementById('spectator-banner');
        if (this.isSpectating) {
            jLeft.style.display = 'none';
            jRight.style.display = 'none';
            if (specBanner) specBanner.style.display = 'block';
        } else {
            jLeft.style.display = '';
            jRight.style.display = '';
            if (specBanner) specBanner.style.display = 'none';
        }

        // Stop studio refresh when entering game
        ModelStudio.stopAutoRefresh();

        this.showScreen('game');
        this.currentState = null;
        this._gameStartTick = 0;
        
        // Reset HUD explicitly so it doesn't show previous game HP if connection fails
        this._updateHUD({ p: [0, 0, 100], e: [0, 0, 100], t: 0, bs: [] });

        // Connect WebSocket
        GameSocket.connect(
            mode,
            botName,
            this.isSpectating,
            isRecording,
            opponent,
            snapshot,
            // onStateUpdate
            (state) => {
                this.currentState = state;
                if (this._gameStartTick === 0 && state.t) {
                    this._gameStartTick = state.t;
                }
                this._updateHUD(state);
            },
            // onGameOver
            (winner, score) => {
                console.log(`[App] Game over: Winner ${winner}, Score ${score}`);
                setTimeout(() => {
                    this._stopRenderLoop();
                    GameSocket.disconnect();
                    this._showResult(winner, score);
                }, 300);
            },
            // onDisconnect
            (isError) => {
                this._stopRenderLoop();
                if (isError) {
                    this._updateStatus('Disconnected', 'error');
                }
            }
        );

        // Start render loop
        this._startRenderLoop();
    },

    _startRenderLoop() {
        this._stopRenderLoop();
        const loop = () => {
            if (!this.isSpectating) {
                Controls.updateKeyboard();
            }
            Renderer.render(this.currentState);
            this._animFrameId = requestAnimationFrame(loop);
        };
        this._animFrameId = requestAnimationFrame(loop);
    },

    _stopRenderLoop() {
        if (this._animFrameId) {
            cancelAnimationFrame(this._animFrameId);
            this._animFrameId = null;
        }
    },

    _updateHUD(state) {
        if (!state) return;

        const playerHP = state.p ? state.p[2] : 100;
        const enemyHP = state.e ? state.e[2] : 100;

        // HP bars
        const pFill = document.getElementById('hp-fill-player');
        const eFill = document.getElementById('hp-fill-enemy');
        const pText = document.getElementById('hp-text-player');
        const eText = document.getElementById('hp-text-enemy');

        if (pFill) pFill.style.width = Math.max(0, playerHP) + '%';
        if (eFill) eFill.style.width = Math.max(0, enemyHP) + '%';
        if (pText) pText.textContent = Math.max(0, Math.round(playerHP));
        if (eText) eText.textContent = Math.max(0, Math.round(enemyHP));

        // Timer
        const timer = document.getElementById('hud-timer');
        if (timer && state.t) {
            const elapsed = state.t - this._gameStartTick;
            const seconds = Math.floor(elapsed / 60);
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            timer.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
        }
    },

    _showResult(winner, score) {
        this.showScreen('result');
        const title = document.getElementById('result-title');
        const sub = document.getElementById('result-sub');
        const scoreEl = document.getElementById('result-score');

        if (this.isSpectating) {
            title.textContent = winner === 1 ? 'BLUE WINS' : 'RED WINS';
            title.className = winner === 1 ? 'result-title victory' : 'result-title defeat';
            sub.textContent = winner === 1 ? 'RL Bot eliminated the Rule Bot!' : 'Rule Bot eliminated the RL Bot!';
        } else if (winner === 1) {
            title.textContent = 'VICTORY';
            title.className = 'result-title victory';
            sub.textContent = 'Enemy eliminated!';
        } else if (winner === 0) {
            title.textContent = 'TIMEOUT';
            title.className = 'result-title';
            title.style.color = '#eab308';
            sub.textContent = 'Time limit reached.';
        } else {
            title.textContent = 'DEFEAT';
            title.className = 'result-title defeat';
            sub.textContent = 'You were eliminated.';
        }
        
        if (score !== undefined && !this.isSpectating) {
            scoreEl.textContent = `Score: ${score.toFixed(2)}`;
            scoreEl.style.display = 'block';
        } else {
            scoreEl.style.display = 'none';
        }

        const saveBox = document.getElementById('result-save-box');
        const saveStatus = document.getElementById('result-save-status');
        const saveBtn = document.getElementById('btn-save-recording');
        const discardBtn = document.getElementById('btn-discard-recording');

        if (this.isRecording && saveBox) {
            saveBox.style.display = 'block';
            if (saveStatus) saveStatus.style.display = 'none';
            if (saveBtn) saveBtn.disabled = false;
            if (discardBtn) discardBtn.disabled = false;

            // Fetch step count of captured game
            fetch('/api/imitation/pending')
                .then(res => res.json())
                .then(info => {
                    const stepsEl = document.getElementById('result-record-steps');
                    if (stepsEl) stepsEl.textContent = info.steps || 0;
                })
                .catch(() => {});
        } else if (saveBox) {
            saveBox.style.display = 'none';
        }

        this.showScreen('result');
    },

    _updateStatus(text, className) {
        const el = document.getElementById('connection-status');
        if (el) {
            el.textContent = text;
            el.className = 'status ' + (className || '');
        }
    },
};

// ─── Bootstrap ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
