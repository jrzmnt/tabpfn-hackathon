"""Records a side-by-side GIF comparing the PPO expert, the TabPFN-BC
policy, and a random policy, on any environment registered in
scripts/envs.py.

    uv run python scripts/record_comparison_video.py --env halfcheetah
    uv run python scripts/record_comparison_video.py --env hopper
"""
import argparse
import os
from pathlib import Path

# Avoids rebuilding the transformer from scratch on every TabPFNRegressor()
# instantiation below (one per action dimension, same checkpoint each time) --
# must be set before any TabPFN model load. ~30% faster in our measurements.
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")

import numpy as np
import gymnasium as gym
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from envs import ENVIRONMENTS, load_expert, make_venv, resolve_env

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MEDIA_DIR = ROOT_DIR / "media"

N_STEPS = 150  # kept short -- each TabPFN-BC step needs n_action_dims predict() calls (~0.5s each)
N_DEMO_TRANSITIONS = 2000
SEED = 0


def rollout_frames(spec, policy_fn, n_steps, label):
    venv = make_venv(spec, render_mode="rgb_array")
    # VecNormalize wraps the DummyVecEnv in .venv; without it (e.g. humanoid),
    # `venv` is already that DummyVecEnv, which has .envs directly.
    render_env = getattr(venv, "venv", venv).envs[0]
    obs = venv.reset()
    frames = [render_env.render()]
    ep_return = 0.0
    for step in range(n_steps):
        action = policy_fn(obs)
        obs, reward, dones, infos = venv.step(action)
        frames.append(render_env.render())
        ep_return += reward[0]
        if (step + 1) % 5 == 0:
            print(f"{label} step {step + 1}/{n_steps} return={ep_return:.1f}")
        if dones[0]:
            break
    venv.close()
    return frames, ep_return


def fit_tabpfn_policy(env_key):
    data = np.load(DATA_DIR / f"demos_{env_key}.npz")
    states, actions = data["states"], data["actions"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(states.shape[0], size=min(N_DEMO_TRANSITIONS, states.shape[0]), replace=False)
    X_train, y_train = states[idx], actions[idx]

    regressors = []
    for d in range(y_train.shape[1]):
        reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST)
        reg.fit(X_train, y_train[:, d])
        regressors.append(reg)

    def policy(obs_batch):
        return np.stack([reg.predict(obs_batch) for reg in regressors], axis=1)

    return policy


def label_frame(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return img


def make_side_by_side_gif(panels, out_path, fps):
    """panels: list of (frames, return, label) tuples, rendered left to right."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(frames) for frames, _, _ in panels)
    duration_ms = int(1000 / fps)
    gif_frames = []
    for i in range(n):
        labeled = [
            label_frame(frames[i], f"{label}  return={ret:.0f}")
            for frames, ret, label in panels
        ]
        combined = Image.new("RGB", (sum(img.width for img in labeled), labeled[0].height))
        x = 0
        for img in labeled:
            combined.paste(img, (x, 0))
            x += img.width
        gif_frames.append(combined)
    gif_frames[0].save(
        out_path, save_all=True, append_images=gif_frames[1:], duration=duration_ms, loop=0
    )
    print(f"saved {out_path} ({n} frames, {fps} fps)")


def main(env_key):
    spec = resolve_env(env_key)
    rng = np.random.default_rng(SEED)

    expert_model = load_expert(spec)

    def expert_policy(obs_batch):
        action, _ = expert_model.predict(obs_batch, deterministic=True)
        return action

    action_low = expert_model.action_space.low
    action_high = expert_model.action_space.high

    def random_policy(obs_batch):
        return rng.uniform(action_low, action_high, size=(obs_batch.shape[0], len(action_low))).astype(np.float32)

    print(f"=== [{env_key}] recording expert ({spec.algo.__name__}) ===")
    expert_frames, expert_return = rollout_frames(spec, expert_policy, N_STEPS, "expert")

    print(f"\n=== [{env_key}] recording random policy ===")
    random_frames, random_return = rollout_frames(spec, random_policy, N_STEPS, "random")

    print(f"\n=== [{env_key}] fitting TabPFN-BC policy (in-context, no training loop) ===")
    tabpfn_policy = fit_tabpfn_policy(env_key)

    print(f"\n=== [{env_key}] recording TabPFN-BC ===")
    tabpfn_frames, tabpfn_return = rollout_frames(spec, tabpfn_policy, N_STEPS, "tabpfn-bc")

    probe_env = gym.make(spec.env_id, **spec.env_kwargs)
    fps = probe_env.metadata.get("render_fps", 20)
    probe_env.close()

    panels = [
        (random_frames, random_return, "Random"),
        (expert_frames, expert_return, f"Expert ({spec.algo.__name__})"),
        (tabpfn_frames, tabpfn_return, "TabPFN-BC"),
    ]
    out_path = MEDIA_DIR / f"comparison_{env_key}.gif"
    make_side_by_side_gif(panels, out_path, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), default="halfcheetah",
                         help="which MuJoCo environment to record")
    args = parser.parse_args()
    main(args.env)
