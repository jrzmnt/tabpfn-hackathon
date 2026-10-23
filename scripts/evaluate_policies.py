"""Closed-loop evaluation: expert PPO vs TabPFN-BC vs random policy, on
any environment registered in scripts/envs.py.

TabPFN inference cost is dominated by a fixed per-call overhead (~0.5s),
almost independent of query batch size (see README). A naive step-by-step
rollout of N_EPISODES sequential 1000-step episodes would need
N_EPISODES * 1000 * n_action_dims predict() calls -- tens of thousands,
i.e. hours. Instead we run N_ENVS episodes *in parallel* (one vectorized
env per "episode") and cap episode length, so each timestep needs exactly
n_action_dims predict() calls (batched across all parallel envs) and the
total call count is EPISODE_LEN * n_action_dims regardless of N_ENVS.

    uv run python scripts/evaluate_policies.py --env halfcheetah
    uv run python scripts/evaluate_policies.py --env hopper
"""
import argparse
import os
import time
from pathlib import Path

# Avoids rebuilding the transformer from scratch on every TabPFNRegressor()
# instantiation below (one per action dimension, same checkpoint each time) --
# must be set before any TabPFN model load. ~30% faster in our measurements.
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")

import numpy as np
from dotenv import load_dotenv
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from envs import ENVIRONMENTS, load_expert, make_venv, resolve_env

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MEDIA_DIR = ROOT_DIR / "media"

N_ENVS = 20  # parallel episodes per policy (plays the role of "~20 episodes")
DEFAULT_EPISODE_LEN = 200  # capped well below the default 1000 to keep TabPFN inference tractable
N_DEMO_TRANSITIONS = 2000  # TabPFN context budget -- subsample from the 20k collected
SEED = 0


def rollout(venv, policy_fn, n_steps, label):
    obs = venv.reset()
    returns = np.zeros(venv.num_envs)
    t0 = time.time()
    for step in range(n_steps):
        action = policy_fn(obs)
        obs, reward, dones, infos = venv.step(action)
        returns += reward
        if (step + 1) % max(1, n_steps // 10) == 0:
            print(f"{label} step {step + 1}/{n_steps} ({time.time() - t0:.0f}s elapsed)")
    print(f"{label}: {n_steps} steps x {venv.num_envs} parallel envs done in {time.time() - t0:.0f}s")
    return returns


def main(env_key, episode_len=DEFAULT_EPISODE_LEN):
    spec = resolve_env(env_key)

    data = np.load(DATA_DIR / f"demos_{env_key}.npz")
    states, actions = data["states"], data["actions"]
    n_dims = actions.shape[1]
    print(f"[{env_key}] loaded {states.shape[0]} demo transitions")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(states.shape[0], size=min(N_DEMO_TRANSITIONS, states.shape[0]), replace=False)
    X_train, y_train = states[idx], actions[idx]
    print(f"fitting TabPFN on {X_train.shape[0]} transitions ({X_train.shape[1]} obs dims -> {n_dims} action dims)")

    t0 = time.time()
    regressors = []
    for d in range(n_dims):
        reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST)
        reg.fit(X_train, y_train[:, d])
        regressors.append(reg)
    print(f"fit done (in-context, no training loop) in {time.time()-t0:.1f}s")

    def tabpfn_policy(obs_batch):
        preds = np.stack([reg.predict(obs_batch) for reg in regressors], axis=1)
        return preds

    expert_model = load_expert(spec)
    action_low = expert_model.action_space.low
    action_high = expert_model.action_space.high

    def random_policy(obs_batch):
        return rng.uniform(action_low, action_high, size=(obs_batch.shape[0], n_dims)).astype(np.float32)

    def expert_policy(obs_batch):
        action, _ = expert_model.predict(obs_batch, deterministic=True)
        return action

    print(f"\n=== [{env_key}] evaluating expert ({spec.algo.__name__}) policy: {N_ENVS} parallel episodes x {episode_len} steps ===")
    venv = make_venv(spec, N_ENVS, max_episode_steps=episode_len)
    expert_returns = rollout(venv, expert_policy, episode_len, "expert")
    venv.close()

    print(f"\n=== [{env_key}] evaluating TabPFN-BC policy: {N_ENVS} parallel episodes x {episode_len} steps ===")
    venv = make_venv(spec, N_ENVS, max_episode_steps=episode_len)
    tabpfn_returns = rollout(venv, tabpfn_policy, episode_len, "tabpfn-bc")
    venv.close()

    print(f"\n=== [{env_key}] evaluating random policy: {N_ENVS} parallel episodes x {episode_len} steps ===")
    venv = make_venv(spec, N_ENVS, max_episode_steps=episode_len)
    random_returns = rollout(venv, random_policy, episode_len, "random")
    venv.close()

    print(f"\n--- [{env_key}] summary (mean return +/- std over {N_ENVS} episodes, {episode_len} steps each) ---")
    print(f"expert ({spec.algo.__name__}): {expert_returns.mean():.1f} +/- {expert_returns.std():.1f}")
    print(f"tabpfn-bc:    {tabpfn_returns.mean():.1f} +/- {tabpfn_returns.std():.1f}")
    print(f"random:       {random_returns.mean():.1f} +/- {random_returns.std():.1f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if episode_len == DEFAULT_EPISODE_LEN else f"_ep{episode_len}"
    results_path = DATA_DIR / f"results_{env_key}{suffix}.npz"
    np.savez(
        results_path,
        expert_returns=expert_returns,
        tabpfn_returns=tabpfn_returns,
        random_returns=random_returns,
        episode_len=episode_len,
        n_envs=N_ENVS,
    )
    print(f"\nsaved {results_path}")

    plot_results(env_key, spec.env_id, spec.algo.__name__, suffix)


def plot_results(env_key, env_id, algo_name, suffix=""):
    import matplotlib.pyplot as plt

    data = np.load(DATA_DIR / f"results_{env_key}{suffix}.npz")
    labels = [f"Expert ({algo_name})", "TabPFN-BC", "Random"]
    keys = ["expert_returns", "tabpfn_returns", "random_returns"]
    means = [data[k].mean() for k in keys]
    stds = [data[k].std() for k in keys]
    colors = ["#4C78A8", "#F58518", "#B0B0B0"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, zorder=2)

    top = max(m + s for m, s in zip(means, stds))
    bottom = min(m - s for m, s in zip(means, stds))
    span = top - bottom
    text_offset = 0.04 * span  # gap between error-bar cap and value label
    ylim_pad = 0.20 * span  # extra headroom so the label text itself doesn't clip the title/x-axis
    ax.set_ylim(bottom - ylim_pad, top + ylim_pad)

    for bar, mean, std in zip(bars, means, stds):
        label_y = mean + std + text_offset if mean >= 0 else mean - std - text_offset
        va = "bottom" if mean >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{mean:.0f}",
                 ha="center", va=va, fontsize=11, fontweight="bold", zorder=3)

    ax.set_ylabel("Episode return (mean ± std)")
    ax.set_title(
        f"{env_id} ({int(data['episode_len'])}-step episodes)\n"
        "expert vs TabPFN-cloned vs random",
        pad=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEDIA_DIR / f"results_{env_key}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), default="halfcheetah",
                         help="which MuJoCo environment to evaluate on")
    parser.add_argument("--episode-len", type=int, default=DEFAULT_EPISODE_LEN,
                         help=f"steps per episode; results/plot go to a *_ep<N> suffixed file when "
                              f"this differs from the default (default: {DEFAULT_EPISODE_LEN})")
    args = parser.parse_args()
    main(args.env, args.episode_len)
