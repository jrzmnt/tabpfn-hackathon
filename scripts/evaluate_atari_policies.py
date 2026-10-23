"""Closed-loop evaluation: expert PPO vs TabPFN-BC vs random policy, on any
Atari game registered in scripts/atari_envs.py. Same parallel-episode design
as scripts/evaluate_policies.py, for the same reason: TabPFN's per-call cost
is roughly fixed regardless of batch size, so running N_ENVS episodes in
parallel and doing one batched predict() call per step keeps total
inference calls bounded by how many steps the rollout actually takes, not
multiplied by N_ENVS.

Unlike the MuJoCo pipeline (which caps every episode at a fixed length),
this runs each of the N_ENVS parallel episodes to its own true end (done=True
from the env, i.e. a full game/life, not an artificial cutoff), masking
further reward accumulation per env once it's done so a faster env's next
auto-reset episode doesn't leak into this one's return. MAX_STEPS is a
safety cap, not a target -- real episodes are expected to finish well
before it. This is slower than a fixed short cap (worth it for a
scientifically meaningful full-episode return) but still only pays for
however long the *slowest* of the N_ENVS episodes takes, since N_ENVS
parallelism is free here (see the docstring paragraph above).

TabPFN-BC's "state" is the expert's own frozen CNN feature-extractor output
(512-dim), not raw pixels -- see scripts/atari_envs.py's module docstring.

    uv run python scripts/evaluate_atari_policies.py --env pong
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")

import numpy as np
from dotenv import load_dotenv
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from atari_envs import (
    DATA_DIR,
    MEDIA_DIR,
    SUPPORTED_ATARI_ENVIRONMENTS,
    extract_features,
    load_atari_expert,
    make_atari_venv,
    resolve_atari_env,
)

N_ENVS = 20  # free to raise -- doesn't add TabPFN calls, only more episodes' worth of stats
DEFAULT_MAX_STEPS = 8000  # safety cap for a genuine full episode -- comfortably above the longest
                           # average episode length we've measured (breakout's ~5,745 steps); real
                           # episodes should finish well before this. Pass --max-steps lower for a
                           # documented, deliberate truncation (see README re: breakout's runtime).
N_DEMO_TRANSITIONS = 500  # smaller context than MuJoCo's 2000; see README for the per-call-cost tradeoff
SEED = 0


def rollout(venv, policy_fn, max_steps, label):
    obs = venv.reset()
    returns = np.zeros(venv.num_envs)
    active = np.ones(venv.num_envs, dtype=bool)  # False once that env's episode has ended
    episode_lengths = np.zeros(venv.num_envs, dtype=int)
    t0 = time.time()
    step = 0
    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, dones, infos = venv.step(action)
        returns += reward * active  # don't count reward from an env's next (auto-reset) episode
        episode_lengths += active.astype(int)
        active &= ~np.asarray(dones)
        if (step + 1) % 20 == 0 or not active.any():
            print(f"{label} step {step + 1}/{max_steps} ({time.time() - t0:.0f}s elapsed, "
                  f"{int(active.sum())}/{venv.num_envs} episodes still running)")
        if not active.any():
            break
    print(f"{label}: stopped after {step + 1} steps ({time.time() - t0:.0f}s); "
          f"episode lengths mean={episode_lengths.mean():.0f} max={episode_lengths.max()}")
    return returns


def main(env_key, max_steps):
    spec = resolve_atari_env(env_key)

    data = np.load(DATA_DIR / f"atari_demos_{env_key}.npz")
    states, actions = data["states"], data["actions"]
    print(f"[{env_key}] loaded {states.shape[0]} demo transitions")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(states.shape[0], size=min(N_DEMO_TRANSITIONS, states.shape[0]), replace=False)
    X_train, y_train = states[idx], actions[idx]

    t0 = time.time()
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)
    clf.fit(X_train, y_train)
    print(f"[{env_key}] fit done (in-context, no training loop) in {time.time()-t0:.1f}s")

    expert_model = load_atari_expert(spec)
    n_actions = int(actions.max()) + 1

    def expert_policy(obs_batch):
        action, _ = expert_model.predict(obs_batch, deterministic=True)
        return action

    def tabpfn_policy(obs_batch):
        feats = extract_features(expert_model, obs_batch)
        return clf.predict(feats)

    def random_policy(obs_batch):
        return rng.integers(0, n_actions, size=obs_batch.shape[0])

    print(f"\n=== [{env_key}] evaluating expert (PPO) policy: {N_ENVS} parallel full episodes (max {max_steps} steps) ===")
    venv = make_atari_venv(spec, N_ENVS)
    expert_returns = rollout(venv, expert_policy, max_steps, "expert")
    venv.close()

    print(f"\n=== [{env_key}] evaluating TabPFN-BC policy: {N_ENVS} parallel full episodes (max {max_steps} steps) ===")
    venv = make_atari_venv(spec, N_ENVS)
    tabpfn_returns = rollout(venv, tabpfn_policy, max_steps, "tabpfn-bc")
    venv.close()

    print(f"\n=== [{env_key}] evaluating random policy: {N_ENVS} parallel full episodes (max {max_steps} steps) ===")
    venv = make_atari_venv(spec, N_ENVS)
    random_returns = rollout(venv, random_policy, max_steps, "random")
    venv.close()

    print(f"\n--- [{env_key}] summary (mean return +/- std over {N_ENVS} full episodes) ---")
    print(f"expert (PPO): {expert_returns.mean():.1f} +/- {expert_returns.std():.1f}")
    print(f"tabpfn-bc:    {tabpfn_returns.mean():.1f} +/- {tabpfn_returns.std():.1f}")
    print(f"random:       {random_returns.mean():.1f} +/- {random_returns.std():.1f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_path = DATA_DIR / f"atari_results_{env_key}.npz"
    np.savez(
        results_path,
        expert_returns=expert_returns, tabpfn_returns=tabpfn_returns, random_returns=random_returns,
        max_steps=max_steps, n_envs=N_ENVS,
    )
    print(f"\nsaved {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=SUPPORTED_ATARI_ENVIRONMENTS, default="pong")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                         help=f"safety cap on episode length; pass a lower value for a deliberate, "
                              f"documented truncation on very long episodes like breakout's "
                              f"(default: {DEFAULT_MAX_STEPS}, comfortably above any game's real episode length)")
    args = parser.parse_args()
    main(args.env, args.max_steps)
