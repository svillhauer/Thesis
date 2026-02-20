from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- USER SETTINGS -----------------------------
# Update these paths to where each algorithm saved its trajectories.
# Each entry should point to two CSVs: ASV trajectory and glider trajectory.
TRAJ_LOGS = {
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

BASE_DIR = Path("/home/svillhauer/Desktop/Thesis/GYM")
TITLE = "ASV vs Glider Trajectories (One Evaluation Rollout)"
OUT_PNG = "traj_compare.png"
OUT_PDF = "traj_compare.pdf"

# If your CSVs are long/noisy, plot every k-th point (1 = all points)
DOWNSAMPLE = 1
# ------------------------------------------------------------------------


X_CANDIDATES = ["x", "x_pos", "x_position", "x_asv", "asv_x", "pos_x"]
Y_CANDIDATES = ["y", "y_pos", "y_position", "y_asv", "asv_y", "pos_y"]


def find_xy_columns(df: pd.DataFrame):
    """Try to infer x/y columns from a dataframe."""
    cols_lower = {c.lower(): c for c in df.columns}

    x_col = None
    y_col = None

    for c in X_CANDIDATES:
        if c in cols_lower:
            x_col = cols_lower[c]
            break

    for c in Y_CANDIDATES:
        if c in cols_lower:
            y_col = cols_lower[c]
            break

    # Fallback: if first two columns are numeric, assume they are x,y
    if x_col is None or y_col is None:
        if df.shape[1] >= 2:
            c0, c1 = df.columns[0], df.columns[1]
            if pd.api.types.is_numeric_dtype(df[c0]) and pd.api.types.is_numeric_dtype(df[c1]):
                x_col, y_col = c0, c1

    if x_col is None or y_col is None:
        raise ValueError(f"Could not infer x/y columns from columns={list(df.columns)}")

    return x_col, y_col


def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    x_col, y_col = find_xy_columns(df)
    x = pd.to_numeric(df[x_col], errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").dropna().to_numpy(dtype=float)

    # Align lengths if NaNs caused different drops (conservative)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if DOWNSAMPLE > 1:
        x = x[::DOWNSAMPLE]
        y = y[::DOWNSAMPLE]
    return x, y


def main():
    # Pretty style (no explicit colors; matplotlib default cycle is fine)
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    algos = list(TRAJ_LOGS.keys())
    n = len(algos)

    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), dpi=160, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        asv_path = BASE_DIR / TRAJ_LOGS[algo]["asv"]
        gl_path  = BASE_DIR / TRAJ_LOGS[algo]["glider"]

        x_asv, y_asv = load_xy(asv_path)
        x_gl,  y_gl  = load_xy(gl_path)

        # Plot trajectories
        ax.plot(x_asv, y_asv, linewidth=2.2, label="ASV")
        ax.plot(x_gl,  y_gl,  linewidth=2.2, linestyle="--", label="Glider")

        # Start/End markers
        ax.scatter([x_asv[0]], [y_asv[0]], s=40, marker="o", label="ASV start")
        ax.scatter([x_asv[-1]], [y_asv[-1]], s=55, marker="X", label="ASV end")

        ax.scatter([x_gl[0]], [y_gl[0]], s=40, marker="o", label="Glider start")
        ax.scatter([x_gl[-1]], [y_gl[-1]], s=55, marker="X", label="Glider end")

        ax.set_title(algo)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")

        # Tighten limits with a little padding so it looks clean
        all_x = np.concatenate([x_asv, x_gl])
        all_y = np.concatenate([y_asv, y_gl])
        pad_x = 0.05 * (all_x.max() - all_x.min() + 1e-9)
        pad_y = 0.05 * (all_y.max() - all_y.min() + 1e-9)
        ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
        ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

        # Keep legend minimal per panel (optional): comment these two lines if too busy
        ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.suptitle(TITLE, y=1.02)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    plt.show()


if __name__ == "__main__":
    main()
