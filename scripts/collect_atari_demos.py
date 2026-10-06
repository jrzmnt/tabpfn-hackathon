"""Rolls out a PPO Atari expert and records (CNN-embedding, discrete action)
pairs for TabPFN-BC -- the embedding is the expert's own NatureCNN
feature-extractor output (512-dim), not the raw pixel observation (see
scripts/atari_envs.py's module docstring for why).

    uv run python scripts/collect_atari_demos.py --env pong
    uv run python scripts/collect_atari_demos.py --env breakout --episodes 10
"""
import argparse
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from atari_envs import (
    DATA_DIR,
    SUPPORTED_ATARI_ENVIRONMENTS,
    extract_features,
    load_atari_expert,
    make_atari_venv,
    resolve_atari_env,
)

N_EPISODES = 20


def collect(env_key, n_episodes=N_EPISODES, out_path=None):
    spec = resolve_atari_env(env_key)
    out_path = Path(out_path) if out_path else DATA_DIR / f"atari_demos_{env_key}.npz"

    if out_path.exists():
        print(f"[{env_key}] already collected: {out_path} exists -- skipping (delete it to re-collect)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_atari_expert(spec)
    venv = make_atari_venv(spec, n_envs=1)

    embeddings, actions = [], []
    returns = []

    with Progress(
        TextColumn(f"[bold blue]{env_key}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("episodes"),
        TextColumn("• {task.fields[status]}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("collect", total=n_episodes, status="starting...")
        obs = venv.reset()
        for ep in range(n_episodes):
            done = False
            ep_return = 0.0
            while not done:
                feats = extract_features(model, obs)
                action, _ = model.predict(obs, deterministic=True)
                embeddings.append(feats[0])
                actions.append(action[0])
                obs, reward, dones, infos = venv.step(action)
                ep_return += reward[0]
                done = bool(dones[0])
            returns.append(ep_return)
            progress.update(
                task, advance=1,
                status=f"episode {ep}: return={ep_return:.1f}, steps so far={len(embeddings)}",
            )

    embeddings = np.array(embeddings, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    print(f"[{env_key}] collected {embeddings.shape[0]} transitions, embedding_dim={embeddings.shape[1]}, "
          f"n_actions={len(np.unique(actions))}")
    print(f"[{env_key}] expert return: mean={np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    np.savez(out_path, states=embeddings, actions=actions, expert_returns=returns)
    print(f"[{env_key}] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=SUPPORTED_ATARI_ENVIRONMENTS, default="pong",
                         help="which Atari game's expert to roll out")
    parser.add_argument("--episodes", type=int, default=N_EPISODES, help="number of episodes to collect")
    parser.add_argument("--out", default=None, help="output .npz path (default: data/atari_demos_<env>.npz)")
    args = parser.parse_args()
    collect(args.env, n_episodes=args.episodes, out_path=args.out)


if __name__ == "__main__":
    main()
