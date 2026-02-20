from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- EDIT THESE PATHS -------------------------
BASE_DIR = Path("/home/svillhauer/Desktop/Thesis/GYM")

TRAJ = {
    "PPO (recurrent)": {
        "asv": "PPO/new/asv_traj.csv",
        "glider": "PPO/new/glider_traj.csv",
    },
    "SAC (recurrent)": {
        "asv": "SAC/asv_traj.csv",
        "glider": "SAC/glider_traj.csv",
    },
    "TD3 (recurrent)": {
        "asv": "TD3/asv_traj.csv",
        "glider": "TD3/glider_traj.csv",
    },
}

OUT_PNG = "traj_compare_side_by_side.png"
OUT_PDF = "traj_compare_side_by_side.pdf"
DOWNSAMPLE = 1
# ------------------------------------------------------------------


def load_traj(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError(f"Trajectory must have at least two columns: {csv_path}")

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()

    traj = np.vstack([x, y]).T
    traj = traj[~np.isnan(traj).any(axis=1)]

    if DOWNSAMPLE > 1:
        traj = traj[::DOWNSAMPLE]

    return traj


def set_common_limits(axes, trajs, pad_frac=0.06):
    xs = np.concatenate([t[:, 0] for t in trajs])
    ys = np.concatenate([t[:, 1] for t in trajs])

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    xpad = pad_frac * (xmax - xmin + 1e-9)
    ypad = pad_frac * (ymax - ymin + 1e-9)

    for ax in axes:
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)


def main():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    algos = list(TRAJ.keys())
    # fig, axes = plt.subplots(1, len(algos), figsize=(5.2 * len(algos), 5.0), dpi=170, constrained_layout=True)
    fig, axes = plt.subplots(1, len(algos), figsize=(5.2 * len(algos), 5.0), dpi=170)

    if len(algos) == 1:
        axes = [axes]

    all_trajs = []
    loaded = {}

    for algo in algos:
        asv = load_traj(BASE_DIR / TRAJ[algo]["asv"])
        gl  = load_traj(BASE_DIR / TRAJ[algo]["glider"])
        loaded[algo] = (asv, gl)
        all_trajs.extend([asv, gl])

    for ax, algo in zip(axes, algos):
        asv_traj, glider_traj = loaded[algo]

        # --- Paths ---
        ax.plot(asv_traj[:, 0], asv_traj[:, 1],
                color="blue", linewidth=2.6, label="ASV path")
        ax.plot(glider_traj[:, 0], glider_traj[:, 1],
                color="red", linestyle="--", linewidth=2.6, label="Glider path")

        # --- Start markers ---
        ax.scatter(asv_traj[0, 0], asv_traj[0, 1],
                   color="blue", s=55, marker="o", label="ASV start", zorder=5)
        ax.scatter(glider_traj[0, 0], glider_traj[0, 1],
                   color="red", s=55, marker="o", label="Glider start", zorder=5)

        # --- End markers ---
        ax.scatter(asv_traj[-1, 0], asv_traj[-1, 1],
                   color="blue", s=75, marker="x", label="ASV end", zorder=6)
        ax.scatter(glider_traj[-1, 0], glider_traj[-1, 1],
                   color="red", s=75, marker="x", label="Glider end", zorder=6)

        ax.set_title(algo)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal", adjustable="box")

        final_sep = np.linalg.norm(asv_traj[-1] - glider_traj[-1])
        ax.text(
            0.02, 0.02,
            f"Final sep: {final_sep:.2f} m",
            transform=ax.transAxes,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.85),
        )

    set_common_limits(axes, all_trajs)

    # One shared legend (cleaner)
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)

    # fig.legend(h2, l2, loc="upper center", ncol=3, frameon=True, framealpha=0.95)
    fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.90),  # ↓ move legend down
    ncol=3,
    frameon=True,
    framealpha=0.95
)

    # fig.suptitle("ASV vs Glider Trajectories — Side-by-Side Algorithm Comparison", y=1.03)
    fig.suptitle(
    "ASV vs Glider Trajectories — RL Algorithm Comparison",
    fontsize=15,
    y=0.98
    )

    # fig.tight_layout(rect=[0, 0, 1, 0.92])


    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    plt.show()


if __name__ == "__main__":
    main()
