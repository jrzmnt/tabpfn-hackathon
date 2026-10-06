"""Loads a pretrained PPO expert for a MuJoCo environment (rl-baselines3-zoo
checkpoints on the HuggingFace Hub) and rolls it out -- with the matching
VecNormalize observation stats -- to collect (state, action) demonstration
pairs for behavior cloning. Generic across environments: pick one with
--env, or add a new entry to scripts/envs.py's ENVIRONMENTS.

    uv run python scripts/collect_demos.py --env halfcheetah
    uv run python scripts/collect_demos.py --env hopper --episodes 20
"""
import argparse
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from envs import ENVIRONMENTS, load_expert, make_venv, resolve_env

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_EPISODES = 20


def collect(env_key, n_episodes=N_EPISODES, out_path=None):
    spec = resolve_env(env_key)
    out_path = Path(out_path) if out_path else DATA_DIR / f"demos_{env_key}.npz"

    if out_path.exists():
        print(f"[{env_key}] already collected: {out_path} exists -- skipping (delete it to re-collect)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_expert(spec)
    venv = make_venv(spec)

    states, actions = [], []
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
        for ep in range(n_episodes):
            obs = venv.reset()
            done = False
            ep_return = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                states.append(obs[0].copy())
                actions.append(action[0].copy())
                obs, reward, dones, infos = venv.step(action)
                ep_return += reward[0]
                done = bool(dones[0])
            returns.append(ep_return)
            progress.update(
                task, advance=1,
                status=f"episode {ep}: return={ep_return:.1f}, steps so far={len(states)}",
            )

    states = np.array(states)
    actions = np.array(actions)
    print(f"[{env_key}] collected {states.shape[0]} transitions, obs_dim={states.shape[1]}, act_dim={actions.shape[1]}")
    print(f"[{env_key}] expert return: mean={np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    np.savez(out_path, states=states, actions=actions, expert_returns=returns)
    print(f"[{env_key}] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), default="halfcheetah",
                         help="which MuJoCo environment's expert to roll out")
    parser.add_argument("--episodes", type=int, default=N_EPISODES, help="number of episodes to collect")
    parser.add_argument("--out", default=None, help="output .npz path (default: data/demos_<env>.npz)")
    args = parser.parse_args()
    collect(args.env, n_episodes=args.episodes, out_path=args.out)


if __name__ == "__main__":
    main()
