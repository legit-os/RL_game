/**
 * renderer.js — HTML5 Canvas Game Renderer
 *
 * Draws the game arena, players, bullets, HP bars, and joystick indicators.
 * All positions are in game-world coordinates (-10 to 10).
 * Handles high-DPI (Retina/Mobile) displays and viewport scaling.
 */

const Renderer = {
    canvas: null,
    ctx: null,
    width: 0,
    height: 0,

    // World→screen scale
    scale: 1,
    offsetX: 0,
    offsetY: 0,

    // Dynamic labels & spectator mode
    spectateMode: false,
    playerLabel: 'YOU',
    enemyLabel: 'BOT',

    // Colors (match CSS variables)
    COLORS: {
        bg: '#0a0e1a',
        grid: 'rgba(79, 195, 247, 0.05)',
        gridBorder: 'rgba(79, 195, 247, 0.2)',
        player: '#4fc3f7',
        playerGlow: 'rgba(79, 195, 247, 0.4)',
        enemy: '#ff5252',
        enemyGlow: 'rgba(255, 82, 82, 0.4)',
        bullet: '#ffab40',
        bulletGlow: 'rgba(255, 171, 64, 0.5)',
        hpGreen: '#69f0ae',
        hpRed: '#ff5252',
        joystickBase: 'rgba(79, 195, 247, 0.18)',
        joystickStick: 'rgba(79, 195, 247, 0.65)',
        joystickBorder: 'rgba(79, 195, 247, 0.4)',
    },

    MAP_SIZE: 10.0,      // Game world is [-10, 10]
    PLAYER_RADIUS: 0.35, // Visual radius (matches hitbox)

    init() {
        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas.getContext('2d');
        this._resize();
        window.addEventListener('resize', () => this._resize());
        window.addEventListener('orientationchange', () => setTimeout(() => this._resize(), 200));
    },

    _resize() {
        const dpr = window.devicePixelRatio || 1;
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.canvas.style.width = this.width + 'px';
        this.canvas.style.height = this.height + 'px';
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Calculate scale to fit the 20x20 game world into the screen
        const worldSize = this.MAP_SIZE * 2;
        const padding = 20;
        this.scale = Math.min(
            (this.width - padding * 2) / worldSize,
            (this.height - padding * 2) / worldSize
        );
        this.offsetX = this.width / 2;
        this.offsetY = this.height / 2;
    },

    /**
     * Convert game-world coordinates to screen pixels.
     */
    worldToScreen(wx, wy) {
        return {
            x: this.offsetX + wx * this.scale,
            y: this.offsetY + wy * this.scale,
        };
    },

    /**
     * Main render function. Called every frame.
     * @param {Object} gameState - { p: [x,y,hp], e: [x,y,hp], b: [[x,y],...], t: tick }
     */
    render(gameState) {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);

        // Background
        ctx.fillStyle = this.COLORS.bg;
        ctx.fillRect(0, 0, this.width, this.height);

        this._drawArena(ctx);

        if (gameState) {
            this._drawBullets(ctx, gameState.b || []);
            this._drawPlayer(ctx, gameState.p, true);
            this._drawPlayer(ctx, gameState.e, false);
        }

        if (!this.spectateMode) {
            this._drawJoysticks(ctx);
        }
    },

    _drawArena(ctx) {
        const tl = this.worldToScreen(-this.MAP_SIZE, -this.MAP_SIZE);
        const br = this.worldToScreen(this.MAP_SIZE, this.MAP_SIZE);
        const w = br.x - tl.x;
        const h = br.y - tl.y;

        // Arena background
        ctx.fillStyle = 'rgba(20, 25, 41, 0.7)';
        ctx.fillRect(tl.x, tl.y, w, h);

        // Grid lines
        ctx.strokeStyle = this.COLORS.grid;
        ctx.lineWidth = 1;
        for (let i = -this.MAP_SIZE; i <= this.MAP_SIZE; i += 2) {
            const from = this.worldToScreen(i, -this.MAP_SIZE);
            const to = this.worldToScreen(i, this.MAP_SIZE);
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.stroke();

            const fromH = this.worldToScreen(-this.MAP_SIZE, i);
            const toH = this.worldToScreen(-this.MAP_SIZE, i);
            const toH2 = this.worldToScreen(this.MAP_SIZE, i);
            ctx.beginPath();
            ctx.moveTo(fromH.x, fromH.y);
            ctx.lineTo(toH2.x, toH2.y);
            ctx.stroke();
        }

        // Arena border
        ctx.strokeStyle = this.COLORS.gridBorder;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(tl.x, tl.y, w, h);
    },

    _drawPlayer(ctx, playerData, isHuman) {
        if (!playerData) return;

        const [wx, wy, hp] = playerData;
        const pos = this.worldToScreen(wx, wy);
        const r = Math.max(12, this.PLAYER_RADIUS * this.scale);
        const color = isHuman ? this.COLORS.player : this.COLORS.enemy;
        const glow = isHuman ? this.COLORS.playerGlow : this.COLORS.enemyGlow;

        // Outer Glow
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r * 2.2, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();

        // Player Body Circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Inner highlight
        ctx.beginPath();
        ctx.arc(pos.x - r * 0.25, pos.y - r * 0.25, r * 0.35, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.fill();

        // HP bar above player
        const barW = Math.max(40, r * 2.8);
        const barH = 5;
        const barY = pos.y - r - 14;
        const hpFrac = Math.max(0, hp / 100);

        // HP Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(pos.x - barW / 2, barY, barW, barH);

        // HP Fill
        const hpColor = hpFrac > 0.5 ? this.COLORS.hpGreen :
                        hpFrac > 0.25 ? this.COLORS.bullet : this.COLORS.hpRed;
        ctx.fillStyle = hpColor;
        ctx.fillRect(pos.x - barW / 2, barY, barW * hpFrac, barH);

        // Player Label
        ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.font = 'bold 11px Outfit, sans-serif';
        ctx.textAlign = 'center';
        const label = isHuman ? this.playerLabel : this.enemyLabel;
        ctx.fillText(label, pos.x, barY - 4);
    },

    _drawBullets(ctx, bullets) {
        for (const bullet of bullets) {
            if (!bullet || bullet.length < 2) continue;
            const [wx, wy] = bullet;
            const pos = this.worldToScreen(wx, wy);
            const r = 4;

            // Glow
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, r * 2.5, 0, Math.PI * 2);
            ctx.fillStyle = this.COLORS.bulletGlow;
            ctx.fill();

            // Core
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
            ctx.fillStyle = this.COLORS.bullet;
            ctx.fill();

            // White center
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, r * 0.5, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
        }
    },

    _drawJoysticks(ctx) {
        // Left joystick
        const left = Controls.getLeftStick();
        if (left.active) {
            this._drawJoystickVisual(ctx, left.originX, left.originY, left.dx, left.dy, '#4fc3f7');
        }

        // Right joystick
        const right = Controls.getRightStick();
        if (right.active) {
            this._drawJoystickVisual(ctx, right.originX, right.originY, right.dx, right.dy, '#ffab40');
        }
    },

    _drawJoystickVisual(ctx, ox, oy, dx, dy, accentColor) {
        const baseR = 55;
        const stickR = 24;

        // Outer base ring
        ctx.beginPath();
        ctx.arc(ox, oy, baseR, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(26, 32, 53, 0.6)';
        ctx.fill();
        ctx.strokeStyle = accentColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Inner stick handle
        ctx.beginPath();
        ctx.arc(ox + dx, oy + dy, stickR, 0, Math.PI * 2);
        ctx.fillStyle = accentColor;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
    },
};
