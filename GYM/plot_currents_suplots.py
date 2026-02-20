import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from asv_glider_bearing_only_env import AsvGliderBearingEnv


def plot_currents(times=(0, 10, 20, 30), grid_n=25, margin=0.95, save_path=None):
    env = AsvGliderBearingEnv()
    env.reset(seed=2)

    # Grid
    lim = env.world_size * margin
    xs = np.linspace(-lim, lim, grid_n)
    ys = np.linspace(-lim, lim, grid_n)
    X, Y = np.meshgrid(xs, ys)

    def sample(t):
        U = np.zeros_like(X, dtype=np.float32)
        V = np.zeros_like(Y, dtype=np.float32)
        for i in range(grid_n):
            for j in range(grid_n):
                u, v = env.ocean_current(np.array([X[i, j], Y[i, j]], dtype=np.float32), float(t))
                U[i, j] = u
                V[i, j] = v
        mag = np.sqrt(U**2 + V**2)
        return U, V, mag

    # Compute all fields first for consistent color scale
    fields = [sample(t) for t in times]
    mags = [f[2] for f in fields]
    vmin = float(np.min(mags))
    vmax = float(np.max(mags))

    # ---- Layout: 2x2 plots + 1 colorbar column ----
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.06])

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])  # colorbar spans both rows

    last_q = None
    for ax, t, (U, V, mag) in zip(axes, times, fields):
        # length + color = magnitude
        q = ax.quiver(
            X, Y, U, V, mag,
            cmap="viridis",
            clim=(vmin, vmax),
            angles="xy",
            scale_units="xy",
            scale=None,        # <-- key: arrow length follows magnitude
            width=0.004,
            pivot="mid",
        )
        last_q = q

        ax.set_title(f"t = {t}s")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-env.world_size, env.world_size)
        ax.set_ylim(-env.world_size, env.world_size)

    # Single shared colorbar
    cbar = fig.colorbar(last_q, cax=cax)
    cbar.set_label("Current speed (m/s)")

    fig.suptitle("Ocean currents over time")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_currents(times=(0, 10, 20, 30), grid_n=25, save_path=None)
