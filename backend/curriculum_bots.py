import numpy as np

class Level1Bot:
    """Stands still, doesn't attack."""
    def predict(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    def reset(self):
        pass

class Level2Bot:
    """Stands still, tracks enemy and attacks."""
    def predict(self, obs: np.ndarray) -> np.ndarray:
        angle = obs[8] * np.pi
        aim_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        
        ammo = obs[1]
        shoot = 1.0 if ammo > 0.1 else -1.0
        
        return np.array([0.0, 0.0, aim_dir[0], aim_dir[1], shoot], dtype=np.float32)
    def reset(self):
        pass

class Level3Bot:
    """Moves randomly sometimes, tracks and attacks."""
    def __init__(self):
        self.move_dir = np.zeros(2, dtype=np.float32)
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        # Move randomly 5% of the time, hold direction for a while
        if np.random.rand() < 0.05:
            angle = np.random.uniform(0, 2*np.pi)
            self.move_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            
        angle = obs[8] * np.pi
        aim_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        
        ammo = obs[1]
        shoot = 1.0 if ammo > 0.1 else -1.0
        
        return np.array([self.move_dir[0], self.move_dir[1], aim_dir[0], aim_dir[1], shoot], dtype=np.float32)
    def reset(self):
        self.move_dir = np.zeros(2, dtype=np.float32)

class Level4Bot:
    """Moves constantly, tracks and attacks."""
    def __init__(self):
        self.move_dir = np.array([0.0, 1.0], dtype=np.float32)
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        # Change direction 2% of the time
        if np.random.rand() < 0.02:
            angle = np.random.uniform(0, 2*np.pi)
            self.move_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            
        angle = obs[8] * np.pi
        aim_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        
        ammo = obs[1]
        shoot = 1.0 if ammo > 0.1 else -1.0
        
        return np.array([self.move_dir[0], self.move_dir[1], aim_dir[0], aim_dir[1], shoot], dtype=np.float32)
    def reset(self):
        angle = np.random.uniform(0, 2*np.pi)
        self.move_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

class Level5Bot:
    """Aggressively closes distance, tracks and attacks."""
    def __init__(self):
        self.tick_count = 0
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.tick_count += 1
        rel_pos = obs[5:7]
        dist = obs[7]
        angle = obs[8] * np.pi
        
        # Move towards player if far away, strafe if close
        if dist > 0.25:
            move_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        else:
            move_dir = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float32)
            if self.tick_count % 60 < 30:
                move_dir = -move_dir
                
        aim_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        
        ammo = obs[1]
        shoot = 1.0 if dist < 0.45 and ammo > 0.1 else -1.0
        
        return np.array([move_dir[0], move_dir[1], aim_dir[0], aim_dir[1], shoot], dtype=np.float32)
        
    def reset(self):
        self.tick_count = 0
