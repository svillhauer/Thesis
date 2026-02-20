"""
make_currents_gif.py
Creates an *animated* GIF showing the time-varying currents.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from asv_glider_bearing_only_env import AsvGliderBearingEnv


def make_currents_gif(
    out_path="varying_currents.gif",
    grid_n=25,
    frames=120,
    fps=12,
    seed=2,
    margin=0.95,
):
    env = AsvGliderBearingEnv()
    env.reset(seed=seed)

    # Sample grid
    lim = env.world_size * margin
    xs = np.linspace(-lim, lim, grid_n)
    ys = np.linspace(-lim, lim, grid_n)
    X, Y = np.meshgrid(xs, ys)

    def compute_currents(t):
        U = np.zeros_like(X, dtype=np.float32)
        V = np.zeros_like(Y, dtype=np.float32)
        for i in range(grid_n):
            for j in range(grid_n):
                u, v = env.ocean_current(
                    np.array([X[i, j], Y[i, j]], dtype=np.float32),
                    t,
                )
                U[i, j] = u
                V[i, j] = v
        return U, V

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-env.world_size, env.world_size)
    ax.set_ylim(-env.world_size, env.world_size)
    ax.set_title("Time-varying Ocean Currents")

    U0, V0 = compute_currents(env.t)
    quiv = ax.quiver(X, Y, U0, V0)

    def update(_k):
        env.t += env.dt
        U, V = compute_currents(env.t)
        quiv.set_UVC(U, V)
        return (quiv,)

    anim = FuncAnimation(fig, update, frames=frames, blit=True)

    # ✅ Explicit PillowWriter makes a real multi-frame GIF
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved animated GIF: {out_path}")


if __name__ == "__main__":
    make_currents_gif()
