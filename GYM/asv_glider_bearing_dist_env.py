import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AsvGliderBearingEnv(gym.Env):
    """
    Environment-aware ASV-glider tracking with BEARING-ONLY sensing.
    - ASV knows its own GPS position (x_asv, y_asv)
    - ASV measures ONLY the bearing angle to the glider (azimuth)
    - Ocean currents affect the ASV but are observable locally
    - Glider moves with unknown dynamics (constant-velocity model)

    Here we also provide distance in the observation, which you can
    interpret as coming from a state estimator or USBL in the real system.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Time / world setup
        self.dt = 0.1
        self.max_steps = 1000
        self.world_size = 200.0  # domain [-200, 200]

        # ASV velocity command (vx_cmd, vy_cmd)
        max_cmd = 3.0 # 2-3 m/s, 0.25-0.5 m/s for the glider 
        self.action_space = spaces.Box(
            low=np.array([-max_cmd, -max_cmd], dtype=np.float32),
            high=np.array([max_cmd, max_cmd], dtype=np.float32),
        )

        # Observation: [bearing_angle, distance, current_u, current_v, x_asv, y_asv]
        max_dist = np.sqrt(2) * self.world_size
        obs_max = np.array(
            [np.pi, max_dist, 2.0, 2.0, self.world_size, self.world_size],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-obs_max,
            high=obs_max,
            dtype=np.float32,
        )

        # Internal state
        self.asv_pos = None
        self.glider_pos = None
        self.glider_vel = None
        self.t = None
        self.step_count = None
        self.prev_dist = None  # <--- NEW: for progress reward

    # ------------------ Ocean current model ------------------
    def ocean_current(self, pos, t):
        """Simple smooth current field."""
        x, y = pos
        k = 2 * np.pi / (2 * self.world_size)
        u = 0.5 * np.sin(k * y)
        v = 0.2 * np.cos(0.1 * t)
        return np.array([u, v], dtype=np.float32)

    # ------------------ Bearing measurement ------------------
    def bearing_to_glider(self):
        """Return azimuth angle from ASV to glider."""
        rel = self.glider_pos - self.asv_pos
        angle = np.arctan2(rel[1], rel[0])  # [-pi, pi]
        return angle.astype(np.float32)

    # ------------------ Reset ------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.asv_pos = np.random.uniform(-5, 5, size=2).astype(np.float32)
        self.glider_pos = np.random.uniform(-5, 5, size=2).astype(np.float32)

        # Glider unknown drift velocity
        theta = np.random.uniform(-np.pi, np.pi)
        speed = 0.4
        self.glider_vel = speed * np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float32
        )

        self.t = 0.0
        self.step_count = 0

        # initialize previous distance for progress reward
        rel = self.glider_pos - self.asv_pos
        self.prev_dist = float(np.linalg.norm(rel))

        obs = self._get_obs()
        info = {}
        return obs, info

    # ------------------ Observation ------------------
    def _get_obs(self):
        bearing = self.bearing_to_glider()
        current = self.ocean_current(self.asv_pos, self.t)
        rel = self.glider_pos - self.asv_pos
        dist = np.linalg.norm(rel)

        obs = np.array(
            [
                bearing,
                dist,          # distance term
                current[0],
                current[1],
                self.asv_pos[0],
                self.asv_pos[1],
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------ Step ------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # ASV motion
        current = self.ocean_current(self.asv_pos, self.t)
        true_vel = action + current

        self.asv_pos = self.asv_pos + true_vel * self.dt
        self.glider_pos = self.glider_pos + self.glider_vel * self.dt

        self.t += self.dt
        self.step_count += 1

        # Distance after moving
        rel = self.glider_pos - self.asv_pos
        dist = float(np.linalg.norm(rel))

        # -------- reward shaping --------
        # progress term: positive if we got closer than last step
        r_progress = self.prev_dist - dist
        self.prev_dist = dist

        # small penalties on control effort and strong currents
        r_energy = -0.05 * float(np.linalg.norm(action))
        r_curr = -0.05 * float(np.linalg.norm(current))

        # optional tiny distance penalty so it prefers to stay close once it gets there
        r_dist_bias = -0.01 * dist

        reward = r_progress + r_energy + r_curr + r_dist_bias
        # --------------------------------

        terminated = False
        truncated = False

        # leaving the domain is bad
        if np.any(np.abs(self.asv_pos) > self.world_size):
            reward -= 20.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info


# Quick random rollout to check nothing crashes
if __name__ == "__main__":
    env = AsvGliderBearingEnv()
    obs, info = env.reset()
    print("Initial observation:", obs)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step}: reward={reward:.3f}")
        print(
            f"t={env.t:.1f}, "
            f"ASV={env.asv_pos}, "
            f"Glider={env.glider_pos}, "
            f"Bearing={env.bearing_to_glider():.2f}"
        )

        if terminated or truncated:
            print("Episode finished, resetting env...\n")
            obs, info = env.reset()
