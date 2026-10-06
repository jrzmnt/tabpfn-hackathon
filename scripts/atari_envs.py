"""Registry + shared helpers for the Atari CNN-embedding TabPFN-BC pipeline
(collect_atari_demos.py, atari_tabpfn_bc.py, evaluate_atari_policies.py) --
separate from scripts/envs.py's continuous-control MuJoCo registry because
Atari differs in every dimension that matters here:

- **Discrete, not continuous, actions** -- these use TabPFNClassifier, one
  classifier total (not N regressors, one per action dimension).
- **Pixel, not low-dim vector, observations.** The raw frame-stacked
  observation is (84, 84, 4) = 28,224 dims if flattened -- 81x
  HumanoidStandup's 348 dims, which already showed severe R^2 degradation
  under TabPFN's in-context learning (see the main README). Feeding raw
  pixels here was not expected to work and wasn't attempted at scale.
  Instead, the "state" TabPFN sees is the expert PPO policy's own NatureCNN
  feature-extractor output (512-dim) -- extracted fresh at every step from
  both the expert AND the TabPFN-BC policy (TabPFN-BC has no vision of its
  own; it reuses the expert's frozen CNN purely as a fixed feature
  extractor, the same way the rest of this project treats a MuJoCo state
  vector as already given). This changes what's being tested: not "can
  TabPFN learn from pixels" but "can TabPFN's in-context learning clone a
  policy from a fixed, high-quality state representation." A held-out
  offline check on Pong got 85.5% action-prediction accuracy (vs. 20.5% for
  a majority-class baseline) with this approach, which is why it was worth
  building out.

All 4 games are public rl-baselines3-zoo PPO checkpoints, no VecNormalize
(Atari zoo policies train on raw/clipped-reward pixel envs, not normalized
vectors) -- unlike scripts/envs.py, there's nothing here to toggle per env.
"""
from dataclasses import dataclass
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from huggingface_sb3 import load_from_hub

gym.register_envs(ale_py)  # registers "*NoFrameskip-v4" etc. used by the zoo checkpoints below


@dataclass(frozen=True)
class AtariEnvSpec:
    env_id: str
    repo_id: str
    model_filename: str


ATARI_ENVIRONMENTS = {
    "pong": AtariEnvSpec("PongNoFrameskip-v4", "sb3/ppo-PongNoFrameskip-v4", "ppo-PongNoFrameskip-v4.zip"),
    "breakout": AtariEnvSpec("BreakoutNoFrameskip-v4", "sb3/ppo-BreakoutNoFrameskip-v4", "ppo-BreakoutNoFrameskip-v4.zip"),
    "spaceinvaders": AtariEnvSpec("SpaceInvadersNoFrameskip-v4", "sb3/ppo-SpaceInvadersNoFrameskip-v4", "ppo-SpaceInvadersNoFrameskip-v4.zip"),
    "mspacman": AtariEnvSpec("MsPacmanNoFrameskip-v4", "sb3/ppo-MsPacmanNoFrameskip-v4", "ppo-MsPacmanNoFrameskip-v4.zip"),
}

SUPPORTED_ATARI_ENVIRONMENTS = sorted(ATARI_ENVIRONMENTS)


def resolve_atari_env(env_key: str) -> AtariEnvSpec:
    if env_key not in ATARI_ENVIRONMENTS:
        raise ValueError(f"--env {env_key} not in {SUPPORTED_ATARI_ENVIRONMENTS}")
    return ATARI_ENVIRONMENTS[env_key]


def load_atari_expert(spec: AtariEnvSpec) -> PPO:
    model_path = load_from_hub(repo_id=spec.repo_id, filename=spec.model_filename)
    return PPO.load(model_path)


def make_atari_venv(spec: AtariEnvSpec, n_envs: int = 1, seed: int = 0, render_mode: str | None = None):
    """SB3's standard Atari preprocessing (NoopReset, MaxAndSkip, resize to
    84x84 grayscale, episodic life, reward clipping) + 4-frame stack --
    matching what the zoo checkpoints were trained on. render_mode="rgb_array"
    enables render() on the returned VecEnv's underlying envs, for video."""
    env_kwargs = {"render_mode": render_mode} if render_mode else None
    venv = make_atari_env(spec.env_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    venv = VecFrameStack(venv, n_stack=4)
    return venv


def extract_features(model: PPO, obs: np.ndarray) -> np.ndarray:
    """Runs obs through the expert's own frozen NatureCNN feature extractor,
    returning a (batch, 512) array -- the "state" TabPFN-BC is fit/evaluated
    on, in place of the raw (84, 84, 4) pixel observation."""
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        feats = model.policy.extract_features(obs_t, model.policy.features_extractor)
    return feats.cpu().numpy()


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"
