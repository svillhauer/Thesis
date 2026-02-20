import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import your env class (adjust module name if you renamed the file)
from asv_glider_bearing_only_env import AsvGliderBearingEnv  # :contentReference[oaicite:1]{index=1}


def make_current_grid(env, n=25, margin=0.95):
    """Precompute a grid for quiver arrows."""
    lim = env.world_size * margin
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    return X, Y, P


def sample_currents(env, grid_points, t):
    """Vectorized sampling of env.ocean_current at many points."""
    U = np.zeros((grid_points.shape[0],), dtype=np.float32)
    V = np.zeros((grid_points.shape[0],), dtype=np.float32)
    for i, p in enumerate(grid_points):
        u, v = env.ocean_current(p, t)
        U[i] = u
        V[i] = v
    return U, V


def run_animation(
    seed=0,
    steps=600,
    grid_n=25,
    interval_ms=33,
    trail_len=200,
    policy="circle",
):
    """
    policy:
      - "random": random actions
      - "circle": smooth circular-ish commanded velocity (looks nice for demos)
      - "toward_bearing": naive steering using bearing only (demo-y, not optimal)
    """
    env = AsvGliderBearingEnv()
    obs, info = env.reset(seed=seed)

    # --- Matplotlib setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-env.world_size, env.world_size)
    ax.set_ylim(-env.world_size, env.world_size)
    ax.set_title("ASV–Glider Bearing-Only Env (Currents + Trajectories)")

    # Domain border
    ax.plot(
        [-env.world_size, env.world_size, env.world_size, -env.world_size, -env.world_size],
        [-env.world_size, -env.world_size, env.world_size, env.world_size, -env.world_size],
        linewidth=1,
    )

    # Current field quiver (updated in time)
    X, Y, P = make_current_grid(env, n=grid_n)
    U0, V0 = sample_currents(env, P, t=env.t)
    quiv = ax.quiver(X, Y, U0.reshape(X.shape), V0.reshape(Y.shape))

    # ASV / Glider markers
    asv_pt, = ax.plot([], [], marker="o", markersize=8, linestyle="None")
    glider_pt, = ax.plot([], [], marker="^", markersize=8, linestyle="None")

    # Trails
    asv_trail, = ax.plot([], [], linewidth=1)
    glider_trail, = ax.plot([], [], linewidth=1)

    # Bearing line (line of sight)
    los_line, = ax.plot([], [], linewidth=1, linestyle="--")

    # Text overlay
    hud = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    # History buffers
    asv_hist = []
    glider_hist = []

    # --- Simple demo policies ---
    def choose_action(policy_name, obs_vec, t):
        # obs = [bearing, current_u, current_v, x_asv, y_asv]
        bearing = float(obs_vec[0])
        # current = obs_vec[1:3]  # available if you want it

        if policy_name == "random":
            return env.action_space.sample()

        if policy_name == "circle":
            # Smooth command that makes the ASV move around (nice for presentations)
            vx = 2.0 * np.cos(0.25 * t)
            vy = 2.0 * np.sin(0.25 * t)
            return np.array([vx, vy], dtype=np.float32)

        if policy_name == "toward_bearing":
            # Naive bearing-only steering: try to point "forward" to reduce |bearing|
            # Here "forward" is +x direction; we rotate a preferred direction opposite bearing.
            desired_heading = -bearing
            speed = 2.0
            return np.array([speed * np.cos(desired_heading), speed * np.sin(desired_heading)], dtype=np.float32)

        raise ValueError(f"Unknown policy: {policy_name}")

    def init():
        asv_pt.set_data([], [])
        glider_pt.set_data([], [])
        asv_trail.set_data([], [])
        glider_trail.set_data([], [])
        los_line.set_data([], [])
        hud.set_text("")
        return quiv, asv_pt, glider_pt, asv_trail, glider_trail, los_line, hud

    def update(frame_idx):
        nonlocal obs

        action = choose_action(policy, obs, env.t)
        obs, reward, terminated, truncated, info = env.step(action)

        # Record history
        asv_hist.append(env.asv_pos.copy())
        glider_hist.append(env.glider_pos.copy())

        # Keep trails bounded
        if len(asv_hist) > trail_len:
            asv_hist.pop(0)
        if len(glider_hist) > trail_len:
            glider_hist.pop(0)

        asv_arr = np.array(asv_hist)
        glider_arr = np.array(glider_hist)

        # Update points
        asv_pt.set_data(env.asv_pos[0], env.asv_pos[1])
        glider_pt.set_data(env.glider_pos[0], env.glider_pos[1])

        # Update trails
        asv_trail.set_data(asv_arr[:, 0], asv_arr[:, 1])
        glider_trail.set_data(glider_arr[:, 0], glider_arr[:, 1])

        # Update LOS/bearing line
        los_line.set_data(
            [env.asv_pos[0], env.glider_pos[0]],
            [env.asv_pos[1], env.glider_pos[1]],
        )

        # Update currents (time-varying v component in your model)
        U, V = sample_currents(env, P, t=env.t)
        quiv.set_UVC(U.reshape(X.shape), V.reshape(Y.shape))

        # HUD
        bearing = float(obs[0])
        cu, cv = float(obs[1]), float(obs[2])
        hud.set_text(
            f"t = {env.t:6.2f} s\n"
            f"bearing = {bearing:+.3f} rad\n"
            f"local current = [{cu:+.2f}, {cv:+.2f}] m/s\n"
            f"reward = {reward:+.3f}\n"
            f"action = [{action[0]:+.2f}, {action[1]:+.2f}]"
        )

        done = terminated or truncated
        if done:
            # Re-seed reset so it looks stable across reruns
            env.reset(seed=seed)
            asv_hist.clear()
            glider_hist.clear()

        return quiv, asv_pt, glider_pt, asv_trail, glider_trail, los_line, hud

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=steps,
        interval=interval_ms,
        blit=True,
    )

    plt.show()


if __name__ == "__main__":
    run_animation(
        seed=2,
        steps=1200,
        grid_n=27,
        interval_ms=33,
        trail_len=250,
        policy="circle",   # try "toward_bearing" too
    )
