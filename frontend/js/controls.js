/**
 * controls.js — Dual Virtual Joystick for Touch Input
 *
 * Left joystick  → movement (move_x, move_y)
 * Right joystick → aim direction (aim_x, aim_y) + auto-fire when deflected
 *
 * Exposes a global `Controls` object with current input state.
 */

const Controls = {
    // Current input state (read by websocket.js every tick)
    moveX: 0,
    moveY: 0,
    aimX: 0,
    aimY: 0,
    shootTrigger: -1,  // -1 = not shooting, 1 = shooting

    // Joystick visual state
    _leftActive: false,
    _rightActive: false,
    _leftOrigin: { x: 0, y: 0 },
    _rightOrigin: { x: 0, y: 0 },
    _leftTouchId: null,
    _rightTouchId: null,
    
    // Buffer for reliable firing
    _fireNextPayload: false,
    _fireAimX: 0,
    _fireAimY: 0,

    // Canvas element references for drawing joystick indicators
    _leftZone: null,
    _rightZone: null,

    // Joystick config
    MAX_RADIUS: 60,      // Max pixel distance from touch origin
    DEAD_ZONE: 8,        // Pixels of dead zone in center
    FIRE_THRESHOLD: 0.3, // Aim deflection needed to fire

    init() {
        this._leftZone = document.getElementById('joystick-left');
        this._rightZone = document.getElementById('joystick-right');

        // Touch events on the joystick zones
        this._leftZone.addEventListener('touchstart', (e) => this._onTouchStart(e, 'left'), { passive: false });
        this._rightZone.addEventListener('touchstart', (e) => this._onTouchStart(e, 'right'), { passive: false });

        document.addEventListener('touchmove', (e) => this._onTouchMove(e), { passive: false });
        document.addEventListener('touchend', (e) => this._onTouchEnd(e), { passive: false });
        document.addEventListener('touchcancel', (e) => this._onTouchEnd(e), { passive: false });

        // Keyboard fallback for desktop testing
        this._keys = {};
        document.addEventListener('keydown', (e) => { this._keys[e.key.toLowerCase()] = true; });
        document.addEventListener('keyup', (e) => { this._keys[e.key.toLowerCase()] = false; });
    },

    /**
     * Update keyboard-based input (for desktop testing).
     * Called each frame from the game loop.
     */
    updateKeyboard() {
        // WASD = movement
        let kx = 0, ky = 0;
        if (this._keys['a']) kx -= 1;
        if (this._keys['d']) kx += 1;
        if (this._keys['w']) ky -= 1;
        if (this._keys['s']) ky += 1;

        // Only override if no touch is active
        if (!this._leftActive) {
            this.moveX = kx;
            this.moveY = ky;
        }

        // Arrow keys = aim + auto-fire
        let ax = 0, ay = 0;
        if (this._keys['arrowleft']) ax -= 1;
        if (this._keys['arrowright']) ax += 1;
        if (this._keys['arrowup']) ay -= 1;
        if (this._keys['arrowdown']) ay += 1;

        if (!this._rightActive) {
            this.aimX = ax;
            this.aimY = ay;
            // On keyboard, we can keep the continuous fire if both keys pressed, 
            // but to match hold-to-aim: shootTrigger = 1 when keys are released?
            // For simplicity on keyboard, let's just allow continuous fire or spacebar to fire.
            // But we're mostly focused on touch for mobile.
            this.shootTrigger = (Math.abs(ax) + Math.abs(ay)) > 0 ? 1 : -1;
        }
    },

    _onTouchStart(e, side) {
        e.preventDefault();
        const touch = e.changedTouches[0];

        if (side === 'left' && this._leftTouchId === null) {
            this._leftTouchId = touch.identifier;
            this._leftOrigin = { x: touch.clientX, y: touch.clientY };
            this._leftActive = true;
        } else if (side === 'right' && this._rightTouchId === null) {
            this._rightTouchId = touch.identifier;
            this._rightOrigin = { x: touch.clientX, y: touch.clientY };
            this._rightActive = true;
        }
    },

    _onTouchMove(e) {
        e.preventDefault();

        for (const touch of e.changedTouches) {
            if (touch.identifier === this._leftTouchId) {
                this._processJoystick(touch, this._leftOrigin, 'left');
            } else if (touch.identifier === this._rightTouchId) {
                this._processJoystick(touch, this._rightOrigin, 'right');
            }
        }
    },

    _onTouchEnd(e) {
        for (const touch of e.changedTouches) {
            if (touch.identifier === this._leftTouchId) {
                this._leftTouchId = null;
                this._leftActive = false;
                this.moveX = 0;
                this.moveY = 0;
            } else if (touch.identifier === this._rightTouchId) {
                this._rightTouchId = null;
                this._rightActive = false;
                
                // If they released the joystick and they had aimed far enough, FIRE!
                let mag = Math.sqrt(this.aimX * this.aimX + this.aimY * this.aimY);
                if (mag > this.FIRE_THRESHOLD) {
                    // Buffer the fire event so it's guaranteed to be sent in the next WS payload
                    this._fireNextPayload = true;
                    this._fireAimX = this.aimX;
                    this._fireAimY = this.aimY;
                }
                
                this.aimX = 0;
                this.aimY = 0;
                this.shootTrigger = -1;
            }
        }
    },

    _processJoystick(touch, origin, side) {
        let dx = touch.clientX - origin.x;
        let dy = touch.clientY - origin.y;
        let dist = Math.sqrt(dx * dx + dy * dy);

        // Dead zone
        if (dist < this.DEAD_ZONE) {
            if (side === 'left') {
                this.moveX = 0;
                this.moveY = 0;
            } else {
                this.aimX = 0;
                this.aimY = 0;
                this.shootTrigger = -1;
            }
            return;
        }

        // Normalize to [-1, 1] within MAX_RADIUS
        let nx = dx / this.MAX_RADIUS;
        let ny = dy / this.MAX_RADIUS;
        let mag = Math.sqrt(nx * nx + ny * ny);
        if (mag > 1) {
            nx /= mag;
            ny /= mag;
        }

        if (side === 'left') {
            this.moveX = nx;
            this.moveY = ny;
        } else {
            this.aimX = nx;
            this.aimY = ny;
            // DO NOT Auto-fire while holding. Just aim.
            // Fire happens on touchend.
            this.shootTrigger = -1;
        }
    },

    /**
     * Get joystick positions for visual rendering.
     */
    getLeftStick() {
        return {
            active: this._leftActive,
            originX: this._leftOrigin.x,
            originY: this._leftOrigin.y,
            dx: this.moveX * this.MAX_RADIUS,
            dy: this.moveY * this.MAX_RADIUS,
        };
    },

    getRightStick() {
        return {
            active: this._rightActive,
            originX: this._rightOrigin.x,
            originY: this._rightOrigin.y,
            dx: this.aimX * this.MAX_RADIUS,
            dy: this.aimY * this.MAX_RADIUS,
        };
    },

    /**
     * Get compact input payload for WebSocket.
     */
    getPayload() {
        let ax = this.aimX;
        let ay = this.aimY;
        let st = this.shootTrigger > 0 ? 1 : 0;
        
        // If a fire event was buffered on release, override this payload
        if (this._fireNextPayload) {
            ax = this._fireAimX;
            ay = this._fireAimY;
            st = 1;
            this._fireNextPayload = false; // Consume the event
        }
        
        return {
            mx: Math.round(this.moveX * 100) / 100,
            my: Math.round(this.moveY * 100) / 100,
            ax: Math.round(ax * 100) / 100,
            ay: Math.round(ay * 100) / 100,
            st: st,
        };
    }
};
