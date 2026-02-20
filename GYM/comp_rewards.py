from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- USER SETTINGS -----------------------------
LOGS = {
    "PPO (recurrent)": "PPO/new/rewards.csv",
    "SAC (recurrent)": "SAC/rewards.csv",
    "TD3 (recurrent)": "TD3/rewards.csv",
}

BASE_DIR = Path("/home/svillhauer/Desktop/Thesis/GYM")
DOWNSAMPLE = 1
TITLE = "Recurrent RL Comparison on ASV–Glider Tracking Task"
OUT_PNG = "rl_compare_raw.png"
OUT_PDF = "rl_compare_raw.pdf"
# ------------------------------------------------------------------------


REWARD_COL_CANDIDATES = ["reward", "episode_reward", "return", "ep_rew_mean", "ep_return", "rewards"]

# Distinct but subtle blues
BLUE_SHADES = {
    "PPO (recurrent)": "#1f77b4",   # classic blue
    "SAC (recurrent)": "#4c9be8",   # lighter blue
    "TD3 (recurrent)": "#0b3c5d",   # darker blue
}


def read_rewards(csv_path: Path):
    df = pd.read_csv(csv_path)

    if df.shape[1] == 1:
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        return s.dropna().reset_index(drop=True)

    cols_lower = {c.lower(): c for c in df.columns}
    for cand in REWARD_COL_CANDIDATES:
        if cand in cols_lower:
            s = pd.to_numeric(df[cols_lower[cand]], errors="coerce")
            return s.dropna().reset_index(drop=True)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        s = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
        return s.dropna().reset_index(drop=True)

    raise ValueError(f"No numeric reward column found in {csv_path}")


def ds(s: pd.Series):
    return s.iloc[::DOWNSAMPLE].reset_index(drop=True).to_numpy(dtype=float)


def align_lengths(series_dict):
    min_len = min(len(v) for v in series_dict.values())
    return {k: v[:min_len] for k, v in series_dict.items()}


def main():
    rewards = {
        algo: ds(read_rewards(BASE_DIR / path))
        for algo, path in LOGS.items()
    }

    rewards = align_lengths(rewards)

    # Styling
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

    fig = plt.figure(figsize=(11, 6.5), dpi=160)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # -------- Panel (a): PPO + SAC + TD3 --------
    for algo in ["PPO (recurrent)", "SAC (recurrent)", "TD3 (recurrent)"]:
        y = rewards[algo]
        x = np.arange(len(y))
        ax1.plot(
            x, y,
            linewidth=1.8,
            color=BLUE_SHADES[algo],
            label=algo
        )

    ax1.set_title("Episode Return (raw) — All Algorithms")
    # ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")

    # -------- Panel (b): SAC + TD3 only --------
    for algo in ["SAC (recurrent)", "TD3 (recurrent)"]:
        y = rewards[algo]
        x = np.arange(len(y))
        ax2.plot(
            x, y,
            linewidth=1.8,
            color=BLUE_SHADES[algo],
            label=algo
        )

    ax2.set_title("Episode Return (raw) — Excluding PPO")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Return")

    # Legends
    leg1 = ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    for lh in leg1.legendHandles:
        lh.set_linewidth(3.0)

    # leg2 = ax2.legend(loc="upper left", frameon=True, framealpha=0.9)
    # for lh in leg2.legendHandles:
    #     lh.set_linewidth(3.0)

    fig.suptitle(TITLE, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    plt.show()


if __name__ == "__main__":
    main()
