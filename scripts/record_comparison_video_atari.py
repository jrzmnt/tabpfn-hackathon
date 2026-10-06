"""Records a side-by-side GIF comparing Random, Expert (PPO), and TabPFN-BC
gameplay for an Atari game registered in scripts/atari_envs.py. TabPFN-BC
acts on the expert's own frozen CNN feature-extractor output (see
scripts/atari_envs.py's module docstring), not raw pixels; only the
rendered *video* uses raw pixels, for a human-viewable GIF.

Each rollout stops at the real episode end (done=True) or --max-steps,
whichever comes first -- --max-steps is a safety cap, not a target. Real
episodes here can run long (our collected Pong demos average ~1,658 steps
per episode): showing a full episode makes for a much more honest GIF, but
also a much bigger file and a much slower TabPFN-BC recording (each step
is one TabPFNClassifier.predict() call, ~3-8s). Pick --max-steps to trade
those off; there's no single right answer.

    uv run python scripts/record_comparison_video_atari.py --env pong
    uv run python scripts/record_comparison_video_atari.py --env pong --max-steps 1800
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw
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

load_dotenv()

DEFAULT_MAX_STEPS = 500  # safety cap, not a target -- see module docstring for the size/runtime tradeoff
N_DEMO_TRANSITIONS = 500
SEED = 0


def rollout_frames(spec, policy_fn, n_steps, label):
    venv = make_atari_venv(spec, n_envs=1, render_mode="rgb_array")
    render_env = venv.venv.envs[0]  # underlying gym.Env, wrapped by VecFrameStack
    obs = venv.reset()
    frames = [render_env.render()]
    ep_return = 0.0
    for step in range(n_steps):
        action = policy_fn(obs)
        obs, reward, dones, infos = venv.step(action)
        frames.append(render_env.render())
        ep_return += reward[0]
        if (step + 1) % 10 == 0:
            print(f"{label} step {step + 1}/{n_steps} return={ep_return:.1f}")
        if dones[0]:
            break
    venv.close()
    return frames, ep_return


def label_frame(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return img


def make_side_by_side_gif(panels, out_path, fps=15):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(frames) for frames, _, _ in panels)
    duration_ms = int(1000 / fps)
    gif_frames = []
    for i in range(n):
        labeled = [label_frame(frames[i], f"{label}  return={ret:.0f}") for frames, ret, label in panels]
        combined = Image.new("RGB", (sum(img.width for img in labeled), labeled[0].height))
        x = 0
        for img in labeled:
            combined.paste(img, (x, 0))
            x += img.width
        gif_frames.append(combined)
    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:], duration=duration_ms, loop=0)
    print(f"saved {out_path} ({n} frames, {fps} fps)")


def main(env_key, max_steps):
    spec = resolve_atari_env(env_key)
    rng = np.random.default_rng(SEED)

    expert_model = load_atari_expert(spec)
    n_actions = expert_model.action_space.n

    def expert_policy(obs_batch):
        action, _ = expert_model.predict(obs_batch, deterministic=True)
        return action

    def random_policy(obs_batch):
        return rng.integers(0, n_actions, size=obs_batch.shape[0])

    print(f"=== [{env_key}] recording random policy (max {max_steps} steps) ===")
    random_frames, random_return = rollout_frames(spec, random_policy, max_steps, "random")

    print(f"\n=== [{env_key}] recording expert (PPO) (max {max_steps} steps) ===")
    expert_frames, expert_return = rollout_frames(spec, expert_policy, max_steps, "expert")

    print(f"\n=== [{env_key}] fitting TabPFN-BC policy (in-context, no training loop) ===")
    data = np.load(DATA_DIR / f"atari_demos_{env_key}.npz")
    states, actions = data["states"], data["actions"]
    idx = rng.choice(states.shape[0], size=min(N_DEMO_TRANSITIONS, states.shape[0]), replace=False)
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)
    clf.fit(states[idx], actions[idx])

    def tabpfn_policy(obs_batch):
        feats = extract_features(expert_model, obs_batch)
        return clf.predict(feats)

    print(f"\n=== [{env_key}] recording TabPFN-BC (max {max_steps} steps) ===")
    tabpfn_frames, tabpfn_return = rollout_frames(spec, tabpfn_policy, max_steps, "tabpfn-bc")

    panels = [
        (random_frames, random_return, "Random"),
        (expert_frames, expert_return, "Expert (PPO)"),
        (tabpfn_frames, tabpfn_return, "TabPFN-BC"),
    ]
    out_path = MEDIA_DIR / f"comparison_{env_key}.gif"
    make_side_by_side_gif(panels, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=SUPPORTED_ATARI_ENVIRONMENTS, default="pong",
                         help="which Atari game to record")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                         help=f"safety cap on rollout length; real episodes end earlier via done=True "
                              f"(default: {DEFAULT_MAX_STEPS})")
    args = parser.parse_args()
    main(args.env, args.max_steps)
