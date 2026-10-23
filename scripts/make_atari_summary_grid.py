"""Combines the per-game atari_results_<env>.npz bar charts (from
evaluate_atari_policies.py) into a single 2x2 grid image for the README.
Pure plotting from already-saved data -- no TabPFN/environment calls.

    uv run python scripts/make_atari_summary_grid.py
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from atari_envs import DATA_DIR, MEDIA_DIR, SUPPORTED_ATARI_ENVIRONMENTS

DEFAULT_ENVS = ["pong", "breakout", "spaceinvaders", "mspacman"]


def plot_env(ax, env_key):
    data = np.load(DATA_DIR / f"atari_results_{env_key}.npz")
    labels = ["Expert\n(PPO)", "TabPFN-BC", "Random"]
    keys = ["expert_returns", "tabpfn_returns", "random_returns"]
    means = [data[k].mean() for k in keys]
    stds = [data[k].std() for k in keys]
    colors = ["#4C78A8", "#F58518", "#B0B0B0"]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, zorder=2)

    top = max(m + s for m, s in zip(means, stds))
    bottom = min(m - s for m, s in zip(means, stds))
    span = top - bottom if top != bottom else max(abs(top), 1.0)
    text_offset = 0.05 * span
    ylim_pad = 0.25 * span
    ax.set_ylim(bottom - ylim_pad, top + ylim_pad)

    for bar, mean, std in zip(bars, means, stds):
        label_y = mean + std + text_offset if mean >= 0 else mean - std - text_offset
        va = "bottom" if mean >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{mean:.1f}",
                 ha="center", va=va, fontsize=9, fontweight="bold", zorder=3)

    ax.set_title(env_key, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)


def main(env_keys):
    n = len(env_keys)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, env_key in zip(axes, env_keys):
        plot_env(ax, env_key)
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle(
        "Atari: expert vs TabPFN-cloned (CNN embedding) vs random -- mean episode return (± std)\n"
        "20 parallel full episodes per game (breakout truncated at 2,500 steps -- see README)",
        fontsize=13, y=0.98,
    )
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEDIA_DIR / "atari_results_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS, choices=SUPPORTED_ATARI_ENVIRONMENTS,
                         help="which games to include, in grid order (default: %(default)s)")
    args = parser.parse_args()
    main(args.envs)
