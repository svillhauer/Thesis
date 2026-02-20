# pip install gymnasium
# pip install gymnasium[classic-control]
# pip install gymnasium[box2d]  # optional, but useful

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def wrap_angle(angle: float) -> float:
    """Map any angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class AsvGliderBearingEnv(gym.Env):
    """
    Environment-aware ASV-glider tracking with BEARING-ONLY sensing.

    - ASV knows its own GPS position (x_asv, y_asv)
    - ASV measures ONLY the bearing angle to the glider (azimuth)
    - Ocean currents affect the ASV but are observable locally
    - Glider moves with unknown dynamics (constant-velocity model)

    Reward is based ONLY on bearing geometry (no range term):
        * keep bearing small in magnitude (target roughly "ahead")
        * keep bearing changes small (avoid losing target)
        * small penalties for control effort and strong currents
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Time / world setup
        self.dt = 0.1
        self.max_steps = 1000
        self.world_size = 200.0  # domain [-200, 200]

        # ASV velocity command (vx_cmd, vy_cmd)
        max_cmd = 3.0
        self.action_space = spaces.Box(
            low=np.array([-max_cmd, -max_cmd], dtype=np.float32),
            high=np.array([max_cmd, max_cmd], dtype=np.float32),
        )

        # Observation: [bearing_angle, current_u, current_v, x_asv, y_asv]
        obs_max = np.array(
            [np.pi, 2.0, 2.0, self.world_size, self.world_size], dtype=np.float32
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
        self.prev_bearing = None  # for bearing-rate term in reward

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
        """Return azimuth angle from ASV to glider (in [-pi, pi])."""
        rel = self.glider_pos - self.asv_pos
        angle = np.arctan2(rel[1], rel[0])
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

        # initialize previous bearing for reward shaping
        self.prev_bearing = self.bearing_to_glider()

        obs = self._get_obs()
        info = {}
        return obs, info

    # ------------------ Observation ------------------
    def _get_obs(self):
        bearing = self.bearing_to_glider()
        current = self.ocean_current(self.asv_pos, self.t)

        # include ASV GPS position + local current + bearing
        obs = np.array(
            [
                bearing,
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

        # ASV motion: commanded velocity + current drift
        current = self.ocean_current(self.asv_pos, self.t)
        true_vel = action + current

        self.asv_pos = self.asv_pos + true_vel * self.dt
        self.glider_pos = self.glider_pos + self.glider_vel * self.dt

        self.t += self.dt
        self.step_count += 1

        # ----- Bearing-only reward -----
        bearing = self.bearing_to_glider()

        # change in bearing (rad/s), wrapped to [-pi, pi]
        if self.prev_bearing is None:
            bearing_change = 0.0
        else:
            bearing_change = wrap_angle(bearing - self.prev_bearing) / self.dt
        self.prev_bearing = bearing

        # 1. Alignment: keep bearing small in magnitude (target roughly ahead)
        r_align = -abs(bearing)  # best is 0, worst ≈ -pi

        # 2. Stability: penalize large bearing rate (losing or whipping LOS)
        r_stable = -0.1 * abs(bearing_change)

        # 3. Control effort penalty
        r_energy = -0.05 * np.linalg.norm(action)

        # 4. Strong-current penalty (optional)
        r_curr = -0.05 * np.linalg.norm(current)

        reward = r_align + r_stable + r_energy + r_curr

        # Termination / truncation
        terminated = False
        truncated = False

        # leave domain -> episode ends with extra penalty
        if np.any(np.abs(self.asv_pos) > self.world_size):
            reward -= 20.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    # ------------------ Render (text) ------------------
    def render(self, mode="human"):
        print(
            f"t={self.t:.1f}, "
            f"ASV={self.asv_pos}, "
            f"Glider={self.glider_pos}, "
            f"Bearing={self.bearing_to_glider():.2f}"
        )


# Random policy rollout for a quick sanity check
if __name__ == "__main__":
    env = AsvGliderBearingEnv()
    obs, info = env.reset()
    print("Initial observation:", obs)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step}: reward={reward:.3f}")
        env.render()

        done = terminated or truncated
        if done:
            print("Episode finished, resetting env...\n")
            obs, info = env.reset()
