/**
 * websocket.js — WebSocket Connection Manager
 *
 * Manages the WebSocket connection to the Python server.
 * Sends human input at a fixed rate (not on every touch move).
 * Receives game state and passes it to the renderer.
 * Supports spectator mode (no input sending).
 */

const GameSocket = {
    ws: null,
    isConnected: false,
    latestState: null,
    _sendInterval: null,
    _isSpectating: false,

    SEND_RATE: 60,  // Input send rate (Hz)

    /**
     * Connect to the WebSocket server.
     * @param {string} mode - "rule_bot", "rl_bot", "spectate_rl_vs_rule", etc.
     * @param {string|null} botName - Name of a specific trained bot to use
     * @param {boolean} isSpectating - Whether this is a spectator session
     * @param {boolean} isRecording - Whether to record the game data
     * @param {Function} onStateUpdate - callback(stateDict)
     * @param {Function} onGameOver - callback(winner)
     * @param {Function} onDisconnect - callback(isError)
     */
    connect(mode, botName, isSpectating, isRecording, opponent, snapshot, onStateUpdate, onGameOver, onDisconnect) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        let url = `${protocol}//${host}/ws`;

        if (mode === 'spectate_training' && botName) {
            url = `${protocol}//${host}/ws/training/${botName}`;
        }

        this._isSpectating = isSpectating;

        console.log(`[WS] Connecting to ${url} | mode: ${mode} | bot: ${botName} | spectate: ${isSpectating}`);
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
            console.log('[WS] Connected');
            this.isConnected = true;

            // Send init message with mode and optional bot name
            const initMsg = { mode: mode, record: isRecording };
            if (botName) initMsg.bot = botName;
            if (opponent) initMsg.opponent = opponent;
            if (snapshot) initMsg.snapshot = snapshot;
            this.ws.send(JSON.stringify(initMsg));

            // Only send input in play mode (not spectating)
            if (!isSpectating) {
                this._startSendLoop();
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.latestState = data;

                // Training spectator may send a status message when not training
                if (data.status === 'not_training') {
                    onStateUpdate({ p: [0, 15, 100], e: [0, -15, 100], t: 0, bs: [] }); // Dummy state
                    return;
                }

                // Check for game over
                if (data.go !== undefined) {
                    onGameOver(data.go, data.score);
                    return;
                }

                onStateUpdate(data);
            } catch (e) {
                console.warn('[WS] Failed to parse:', e);
            }
        };

        this.ws.onclose = () => {
            console.log('[WS] Disconnected');
            this.isConnected = false;
            this._stopSendLoop();
            onDisconnect(false);
        };

        this.ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            onDisconnect(true);
        };
    },

    /**
     * Send human input to the server at a fixed rate.
     */
    _startSendLoop() {
        this._sendInterval = setInterval(() => {
            if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
                return;
            }

            const payload = Controls.getPayload();
            this.ws.send(JSON.stringify(payload));
        }, 1000 / this.SEND_RATE);
    },

    _stopSendLoop() {
        if (this._sendInterval) {
            clearInterval(this._sendInterval);
            this._sendInterval = null;
        }
    },

    disconnect() {
        this._stopSendLoop();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
        this.latestState = null;
    },
};
