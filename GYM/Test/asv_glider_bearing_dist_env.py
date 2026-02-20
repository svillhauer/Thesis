import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AsvGliderBearingEnv(gym.Env):
    """
    Updated Environment with Normalized Observations and Dense Rewards.
    - ASV knows its own GPS position and measures bearing/distance to glider.
    - Observations are normalized to [-1, 1] for stable RL training.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Time / world setup
        self.dt = 0.1
        self.max_steps = 1000
        self.world_size = 200.0 

        # ASV velocity command (vx_cmd, vy_cmd)
        self.max_cmd = 2.0 
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), # Normalized actions
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation: [bearing, dist, curr_u, curr_v, x_asv, y_asv]
        # All limits set to 1.0 because we normalize in _get_obs
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # Internal state
        self.asv_pos = None
        self.glider_pos = None
        self.glider_vel = None
        self.t = None
        self.step_count = None
        self.prev_dist = None

    def ocean_current(self, pos, t):
        x, y = pos
        k = 2 * np.pi / (2 * self.world_size)
        u = 0.5 * np.sin(k * y)
        v = 0.2 * np.cos(0.1 * t)
        return np.array([u, v], dtype=np.float32)

    def bearing_to_glider(self):
        rel = self.glider_pos - self.asv_pos
        return np.arctan2(rel[1], rel[0])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Start ASV and Glider in more varied positions to encourage exploration
        self.asv_pos = np.random.uniform(-50, 50, size=2).astype(np.float32)
        self.glider_pos = np.random.uniform(-50, 50, size=2).astype(np.float32)

        theta = np.random.uniform(-np.pi, np.pi)
        speed = 0.4
        self.glider_vel = speed * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        self.t = 0.0
        self.step_count = 0
        self.prev_dist = np.linalg.norm(self.glider_pos - self.asv_pos)

        return self._get_obs(), {}

    def _get_obs(self):
        """Normalizes all observations to approximately [-1, 1]."""
        rel = self.glider_pos - self.asv_pos
        dist = np.linalg.norm(rel)
        bearing = self.bearing_to_glider()
        current = self.ocean_current(self.asv_pos, self.t)

        # Normalization factors
        max_dist = np.sqrt(2) * self.world_size
        
        obs = np.array([
            bearing / np.pi,               # [-1, 1]
            (dist / max_dist) * 2.0 - 1.0, # Scaled to [-1, 1]
            current[0] / 1.0,              # Current is roughly 0.5
            current[1] / 1.0,
            self.asv_pos[0] / self.world_size,
            self.asv_pos[1] / self.world_size
        ], dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)

    def step(self, action):
        # Scale action from [-1, 1] to actual velocity [-3, 3]
        actual_action = action * self.max_cmd
        actual_action = np.clip(actual_action, -self.max_cmd, self.max_cmd)

        current = self.ocean_current(self.asv_pos, self.t)
        true_vel = actual_action + current

        self.asv_pos += true_vel * self.dt
        self.glider_pos += self.glider_vel * self.dt

        self.t += self.dt
        self.step_count += 1

        rel = self.glider_pos - self.asv_pos
        dist = np.linalg.norm(rel)

        # --- DENSE REWARD SHAPING ---
        # 1. Constant pull towards target (linear)
        r_dist = -0.01 * dist 
        
        # 2. Progress reward (differential)
        # Gives a strong signal: "Am I getting closer right now?"
        r_progress = (self.prev_dist - dist) * 20.0 
        self.prev_dist = dist

        # 3. Precision reward (Gaussian)
        # Only kicks in when very close (sigma=5m)
        r_close = np.exp(-(dist**2) / (2 * 5.0**2))

        # 4. Efficiency penalty
        r_energy = -0.01 * np.sum(action**2)

        reward = r_dist + r_progress + 2.0 * r_close + r_energy

        # Terminal conditions
        terminated = False
        truncated = False

        if np.any(np.abs(self.asv_pos) > self.world_size):
            reward -= 50.0  # Heavy penalty for leaving bounds
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

if __name__ == "__main__":
    env = AsvGliderBearingEnv()
    obs, _ = env.reset()
    print(f"Normalized Obs: {obs}")
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, _, _, _ = env.step(action)
        print(f"Action: {action} -> Reward: {reward:.4f}")


# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np


# class AsvGliderBearingEnv(gym.Env):
#     """
#     Environment-aware ASV-glider tracking with BEARING-ONLY sensing.
#     - ASV knows its own GPS position (x_asv, y_asv)
#     - ASV measures ONLY the bearing angle to the glider (azimuth)
#     - Ocean currents affect the ASV but are observable locally
#     - Glider moves with unknown dynamics (constant-velocity model)

#     Here we also provide distance in the observation, which you can
#     interpret as coming from a state estimator or USBL in the real system.
#     """

#     metadata = {"render.modes": ["human"]}

#     def __init__(self):
#         super().__init__()

#         # Time / world setup
#         self.dt = 0.1
#         self.max_steps = 1000
#         self.world_size = 200.0  # domain [-200, 200]

#         # ASV velocity command (vx_cmd, vy_cmd)
#         max_cmd = 3.0 # 2-3 m/s, 0.25-0.5 m/s for the glider 
#         self.action_space = spaces.Box(
#             low=np.array([-max_cmd, -max_cmd], dtype=np.float32),
#             high=np.array([max_cmd, max_cmd], dtype=np.float32),
#         )

#         # Observation: [bearing_angle, distance, current_u, current_v, x_asv, y_asv]
#         max_dist = np.sqrt(2) * self.world_size
#         obs_max = np.array(
#             [np.pi, max_dist, 2.0, 2.0, self.world_size, self.world_size],
#             dtype=np.float32,
#         )
#         self.observation_space = spaces.Box(
#             low=-obs_max,
#             high=obs_max,
#             dtype=np.float32,
#         )

#         # Internal state
#         self.asv_pos = None
#         self.glider_pos = None
#         self.glider_vel = None
#         self.t = None
#         self.step_count = None
#         self.prev_dist = None  # <--- NEW: for progress reward

#     # ------------------ Ocean current model ------------------
#     def ocean_current(self, pos, t):
#         """Simple smooth current field."""
#         x, y = pos
#         k = 2 * np.pi / (2 * self.world_size)
#         u = 0.5 * np.sin(k * y)
#         v = 0.2 * np.cos(0.1 * t)
#         return np.array([u, v], dtype=np.float32)

#     # ------------------ Bearing measurement ------------------
#     def bearing_to_glider(self):
#         """Return azimuth angle from ASV to glider."""
#         rel = self.glider_pos - self.asv_pos
#         angle = np.arctan2(rel[1], rel[0])  # [-pi, pi]
#         return angle.astype(np.float32)

#     # ------------------ Reset ------------------
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         if seed is not None:
#             np.random.seed(seed)

#         self.asv_pos = np.random.uniform(-5, 5, size=2).astype(np.float32)
#         self.glider_pos = np.random.uniform(-5, 5, size=2).astype(np.float32)

#         # Glider unknown drift velocity
#         theta = np.random.uniform(-np.pi, np.pi)
#         speed = 0.4
#         self.glider_vel = speed * np.array(
#             [np.cos(theta), np.sin(theta)], dtype=np.float32
#         )

#         self.t = 0.0
#         self.step_count = 0

#         # initialize previous distance for progress reward
#         rel = self.glider_pos - self.asv_pos
#         self.prev_dist = float(np.linalg.norm(rel))

#         obs = self._get_obs()
#         info = {}
#         return obs, info

#     # ------------------ Observation ------------------
#     def _get_obs(self):
#         bearing = self.bearing_to_glider()
#         current = self.ocean_current(self.asv_pos, self.t)
#         rel = self.glider_pos - self.asv_pos
#         dist = np.linalg.norm(rel)

#         obs = np.array(
#             [
#                 bearing,
#                 dist,          # distance term
#                 current[0],
#                 current[1],
#                 self.asv_pos[0],
#                 self.asv_pos[1],
#             ],
#             dtype=np.float32,
#         )
#         return obs

#     # ------------------ Step ------------------
#     def step(self, action):
#         action = np.clip(action, self.action_space.low, self.action_space.high)

#         # ASV motion
#         current = self.ocean_current(self.asv_pos, self.t)
#         true_vel = action + current

#         self.asv_pos = self.asv_pos + true_vel * self.dt
#         self.glider_pos = self.glider_pos + self.glider_vel * self.dt

#         self.t += self.dt
#         self.step_count += 1

#         # Distance after moving
#         rel = self.glider_pos - self.asv_pos
#         dist = float(np.linalg.norm(rel))

#         # -------- reward shaping --------
#         r_progress = self.prev_dist - dist
#         self.prev_dist = dist

#         # Positive closeness reward (bounded in (0, 1])
#         sigma = 3.0  # meters-ish scale: bigger = less picky
#         r_close = float(np.exp(-dist / sigma))

#         # Smaller penalties (otherwise they dominate when r_progress ~ 0)
#         r_energy = -0.01 * float(np.linalg.norm(action))
#         r_curr   = -0.01 * float(np.linalg.norm(current))

#         # Remove the always-negative distance bias (optional, but usually yes)
#         # r_dist_bias = -0.01 * dist

#         reward = 1.0 * r_close + 0.5 * r_progress + r_energy + r_curr
#         # --------------------------------


#         terminated = False
#         truncated = False

#         # leaving the domain is bad
#         if np.any(np.abs(self.asv_pos) > self.world_size):
#             reward -= 20.0
#             terminated = True

#         if self.step_count >= self.max_steps:
#             truncated = True

#         obs = self._get_obs()
#         info = {}
#         return obs, reward, terminated, truncated, info


# # Quick random rollout to check nothing crashes
# if __name__ == "__main__":
#     env = AsvGliderBearingEnv()
#     obs, info = env.reset()
#     print("Initial observation:", obs)

#     for step in range(20):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)

#         print(f"Step {step}: reward={reward:.3f}")
#         print(
#             f"t={env.t:.1f}, "
#             f"ASV={env.asv_pos}, "
#             f"Glider={env.glider_pos}, "
#             f"Bearing={env.bearing_to_glider():.2f}"
#         )

#         if terminated or truncated:
#             print("Episode finished, resetting env...\n")
#             obs, info = env.reset()
