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
            this.startGame('rl_bot');
        });
        document.getElementById('btn-studio').addEventListener('click', () => {
            this.showScreen('studio');
            ModelStudio.startAutoRefresh();
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

        // ─── Training bar watch button ────────────────────────────────
        document.getElementById('training-bar-watch').addEventListener('click', () => {
            const botName = TrainingHUD.getActiveBotName();
            if (botName) {
                this.startGameWithBot(botName, 'spectate_rl_vs_rule');
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

    /**
     * Start a game with the default model path (legacy mode).
     */
    startGame(mode) {
        this.selectedBotName = null;
        this._launchGame(mode, null);
    },

    /**
     * Start a game or spectate session with a specific trained bot.
     */
    startGameWithBot(botName, mode) {
        this.selectedBotName = botName;
        this._launchGame(mode, botName);
    },

    _launchGame(mode, botName) {
        this.isSpectating = mode.startsWith('spectate');

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

        // Connect WebSocket
        GameSocket.connect(
            mode,
            botName,
            this.isSpectating,
            // onStateUpdate
            (state) => {
                this.currentState = state;
                if (this._gameStartTick === 0 && state.t) {
                    this._gameStartTick = state.t;
                }
                this._updateHUD(state);
            },
            // onGameOver
            (winner) => {
                console.log(`[App] Game over: Winner ${winner}`);
                setTimeout(() => {
                    this._stopRenderLoop();
                    GameSocket.disconnect();
                    this._showResult(winner);
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

    _showResult(winner) {
        const title = document.getElementById('result-title');
        const sub = document.getElementById('result-sub');

        if (this.isSpectating) {
            title.textContent = winner === 1 ? 'BLUE WINS' : 'RED WINS';
            title.className = winner === 1 ? 'result-title victory' : 'result-title defeat';
            sub.textContent = winner === 1 ? 'RL Bot eliminated the Rule Bot!' : 'Rule Bot eliminated the RL Bot!';
        } else if (winner === 1) {
            title.textContent = 'VICTORY';
            title.className = 'result-title victory';
            sub.textContent = 'Enemy eliminated!';
        } else {
            title.textContent = 'DEFEAT';
            title.className = 'result-title defeat';
            sub.textContent = 'You were eliminated.';
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
