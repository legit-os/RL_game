/**
 * training_hud.js — Global Training Status Bar
 *
 * Polls the /api/training/active endpoint to detect active training jobs
 * and displays a persistent top bar with progress info.
 */

const TrainingHUD = {
    _pollInterval: null,
    _activeBotName: null,

    /**
     * Start polling for active training jobs every 3 seconds.
     */
    start() {
        this.stop();
        this._poll();
        this._pollInterval = setInterval(() => this._poll(), 3000);
    },

    stop() {
        if (this._pollInterval) {
            clearInterval(this._pollInterval);
            this._pollInterval = null;
        }
    },

    async _poll() {
        try {
            const res = await fetch('/api/training/active');
            const data = await res.json();
            const active = data.active || [];

            if (active.length > 0) {
                const job = active[0]; // Show the first active job
                this._activeBotName = job.bot_name;
                this._show(job);
            } else {
                this._activeBotName = null;
                this._hide();
            }
        } catch (e) {
            // Silently ignore polling errors
        }
    },

    _show(job) {
        const bar = document.getElementById('training-bar');
        if (!bar) return;

        const nameEl = document.getElementById('training-bar-name');
        const statsEl = document.getElementById('training-bar-stats');

        const current = job.current_step || 0;
        const total = job.total_timesteps || 1;
        const pct = ((current / total) * 100).toFixed(1);
        const reward = job.avg_reward || 0;

        nameEl.textContent = `⚡ ${job.bot_name}`;
        statsEl.textContent = `${this._fmt(current)} / ${this._fmt(total)} (${pct}%) | Reward: ${reward >= 0 ? '+' : ''}${reward.toFixed(1)}`;

        const watchBtn = document.getElementById('training-bar-watch');
        if (watchBtn) {
            if (job.has_onnx) {
                watchBtn.style.display = 'inline-block';
                watchBtn.textContent = 'Watch Live';
                watchBtn.disabled = false;
            } else {
                watchBtn.style.display = 'inline-block';
                watchBtn.textContent = 'Training...';
                watchBtn.disabled = true;
            }
        }

        bar.classList.add('visible');
    },

    _hide() {
        const bar = document.getElementById('training-bar');
        if (bar) bar.classList.remove('visible');
    },

    _fmt(n) {
        if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
        if (n >= 1_000) return (n / 1_000).toFixed(0) + 'K';
        return String(n);
    },

    getActiveBotName() {
        return this._activeBotName;
    },
};
