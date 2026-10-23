"""Unified data collector: pass --env <name> and --n-transitions <N>, and
it rolls out the matching expert (MuJoCo PPO/SAC or Atari PPO) until at
least N (state, action) pairs are collected -- whole episodes at a time,
so the last one may slightly overshoot N, but it never starts a new
episode once the target is reached. This matters because what TabPFN
actually needs is the size of the context, not how many episodes it took
to gather it, and each env's episode length varies wildly (compare
invertedpendulum's ~200-step episodes to mspacman's ~291 or breakout's
~5,745 -- "20 episodes" means very different amounts of data per env).

Auto-detects whether <name> is a MuJoCo env (scripts/envs.py's
ENVIRONMENTS) or an Atari game (scripts/atari_envs.py's
ATARI_ENVIRONMENTS) and dispatches to the matching collection logic
(continuous states + VecNormalize-or-not for MuJoCo; CNN-embedding states
+ discrete actions for Atari) -- output paths and format match what
scripts/tabpfn_bc.py / scripts/atari_tabpfn_bc.py already expect
(data/demos_<env>.npz / data/atari_demos_<env>.npz), so nothing downstream
needs to change.

    uv run python scripts/collect_data.py --env halfcheetah --n-transitions 5000
    uv run python scripts/collect_data.py --env pong --n-transitions 10000
"""
import argparse
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

import atari_envs
import envs

DEFAULT_N_TRANSITIONS = 20000


def collect_mujoco(env_key, n_transitions, out_path, progress, task):
    spec = envs.resolve_env(env_key)
    model = envs.load_expert(spec)
    venv = envs.make_venv(spec)

    states, actions, returns = [], [], []
    while len(states) < n_transitions:
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
            progress.update(task, completed=min(len(states), n_transitions),
                             status=f"episode {len(returns)}, return so far={ep_return:.1f}")
        returns.append(ep_return)
    venv.close()

    states = np.array(states)
    actions = np.array(actions)
    np.savez(out_path, states=states, actions=actions, expert_returns=returns)
    return states, actions, returns


def collect_atari(env_key, n_transitions, out_path, progress, task):
    spec = atari_envs.resolve_atari_env(env_key)
    model = atari_envs.load_atari_expert(spec)
    venv = atari_envs.make_atari_venv(spec, n_envs=1)

    embeddings, actions, returns = [], [], []
    obs = venv.reset()
    while len(embeddings) < n_transitions:
        done = False
        ep_return = 0.0
        while not done:
            feats = atari_envs.extract_features(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            embeddings.append(feats[0])
            actions.append(action[0])
            obs, reward, dones, infos = venv.step(action)
            ep_return += reward[0]
            done = bool(dones[0])
            progress.update(task, completed=min(len(embeddings), n_transitions),
                             status=f"episode {len(returns)}, return so far={ep_return:.1f}")
        returns.append(ep_return)
    venv.close()

    embeddings = np.array(embeddings, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    np.savez(out_path, states=embeddings, actions=actions, expert_returns=returns)
    return embeddings, actions, returns


def collect(env_key, n_transitions=DEFAULT_N_TRANSITIONS, out_path=None):
    is_mujoco = env_key in envs.ENVIRONMENTS
    is_atari = env_key in atari_envs.ATARI_ENVIRONMENTS
    if is_mujoco and envs.ENVIRONMENTS[env_key] is None:
        envs.resolve_env(env_key)  # raises the standard "unsupported" error with the reason
    if not is_mujoco and not is_atari:
        all_envs = sorted(set(envs.SUPPORTED_ENVIRONMENTS) | set(atari_envs.SUPPORTED_ATARI_ENVIRONMENTS))
        raise ValueError(f"--env {env_key} not found in either registry. Supported: {all_envs}")

    prefix = "demos" if is_mujoco else "atari_demos"
    out_path = Path(out_path) if out_path else atari_envs.DATA_DIR / f"{prefix}_{env_key}.npz"

    if out_path.exists():
        print(f"[{env_key}] already collected: {out_path} exists -- skipping (delete it to re-collect)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn(f"[bold blue]{env_key}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("transitions"),
        TextColumn("• {task.fields[status]}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("collect", total=n_transitions, status="starting...")
        if is_mujoco:
            states, actions, returns = collect_mujoco(env_key, n_transitions, out_path, progress, task)
        else:
            states, actions, returns = collect_atari(env_key, n_transitions, out_path, progress, task)

    print(f"[{env_key}] collected {states.shape[0]} transitions (requested >= {n_transitions}), "
          f"state_dim={states.shape[1]}")
    print(f"[{env_key}] expert return over {len(returns)} episodes: mean={np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"[{env_key}] saved {out_path}")


def main():
    all_envs = sorted(set(envs.SUPPORTED_ENVIRONMENTS) | set(atari_envs.SUPPORTED_ATARI_ENVIRONMENTS))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=all_envs,
                         help="any MuJoCo env (scripts/envs.py) or Atari game (scripts/atari_envs.py)")
    parser.add_argument("--n-transitions", type=int, default=DEFAULT_N_TRANSITIONS,
                         help=f"minimum number of (state, action) pairs to collect, "
                              f"whole episodes at a time (default: {DEFAULT_N_TRANSITIONS})")
    parser.add_argument("--out", "--path", dest="out", default=None,
                         help="output .npz path (default: data/[atari_]demos_<env>.npz)")
    args = parser.parse_args()
    collect(args.env, n_transitions=args.n_transitions, out_path=args.out)


if __name__ == "__main__":
    main()
