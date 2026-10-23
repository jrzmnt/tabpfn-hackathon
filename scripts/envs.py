"""Registry of MuJoCo environments usable across the pipeline scripts
(collect_demos.py, evaluate_policies.py, record_comparison_video.py,
split_and_render_gifs.py) -- one source of truth for env id / HF Hub repo /
algorithm / env_kwargs, so every script targets the same environment
definitions.

Two checkpoint sources are mixed here:

1. **rl-baselines3-zoo PPO experts** (`sb3/ppo-*-v3`), trained against the
   old mujoco-py "-v3" environments with a VecNormalize wrapper (halfcheetah,
   hopper, walker2d). gymnasium's current mujoco bindings ("-v4"/"-v5")
   sometimes changed the default observation layout, so `env_kwargs` restores
   the original obs shape the checkpoint (and its VecNormalize stats) expects
   -- e.g. walker2d needs `exclude_current_positions_from_observation=False`.
   `ant`'s old-v3 obs doesn't have a public-kwarg fix (its mujoco-py version
   excluded only the x root coordinate from qpos and included contact forces
   differently than any current option), so it's `None` (unsupported) rather
   than silently loading against the wrong observation space. Separately,
   `walker2d`'s checkpoint has *matching* obs dims but still falls almost
   immediately on the new engine (physics transfer issue, not a dimension
   one) -- see the README's Limitations section.

2. **farama-minari experts**, trained directly on the *current* v5 engine
   (no stale-physics risk): `humanoidstandup`, `invertedpendulum`, and
   `swimmer` (this last one replacing the disabled rl-zoo `sb3/ppo-Swimmer-v3`,
   whose old-v3 obs also didn't have a public-kwarg fix). None of these ship
   a `vec_normalize.pkl`, and their action ranges aren't all `[-1, 1]`
   (humanoidstandup: `Box(-0.4, 0.4, (17,))`; invertedpendulum:
   `Box(-3, 3, (1,))`) -- `use_vecnorm=False` and action bounds always read
   from the loaded model's `action_space` (never assumed) handle both.
"""
from dataclasses import dataclass, field

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from huggingface_sb3 import load_from_hub


@dataclass(frozen=True)
class EnvSpec:
    env_id: str  # gymnasium env id
    repo_id: str  # HF Hub repo id
    model_filename: str  # model .zip filename inside the repo
    env_kwargs: dict = field(default_factory=dict)
    algo: type = PPO  # stable_baselines3 algorithm class used to .load() the checkpoint
    use_vecnorm: bool = True  # whether the repo ships a vec_normalize.pkl to load


ENVIRONMENTS = {
    "halfcheetah": EnvSpec("HalfCheetah-v4", "sb3/ppo-HalfCheetah-v3", "ppo-HalfCheetah-v3.zip"),
    "hopper": EnvSpec("Hopper-v4", "sb3/ppo-Hopper-v3", "ppo-Hopper-v3.zip"),
    "walker2d": EnvSpec("Walker2d-v4", "sb3/ppo-Walker2d-v3", "ppo-Walker2d-v3.zip",
                         env_kwargs={"exclude_current_positions_from_observation": False}),
    "humanoidstandup": EnvSpec("HumanoidStandup-v5", "farama-minari/HumanoidStandup-v5-PPO-expert",
                                "humanoidstandup-v5-ppo-expert.zip", use_vecnorm=False),
    "invertedpendulum": EnvSpec("InvertedPendulum-v5", "farama-minari/InvertedPendulum-v5-SAC-expert",
                                 "invertedpendulum-v5-sac-expert.zip", algo=SAC, use_vecnorm=False),
    "swimmer": EnvSpec("Swimmer-v5", "farama-minari/Swimmer-v5-PPO-expert",
                        "swimmer-v5-PPO-expert.zip", use_vecnorm=False),
    "ant": None,  # unsupported -- see module docstring (obs dim 27 vs checkpoint's 112)
}

SUPPORTED_ENVIRONMENTS = sorted(k for k, v in ENVIRONMENTS.items() if v is not None)


def resolve_env(env_key: str) -> EnvSpec:
    """Returns the EnvSpec for env_key, or raises ValueError with a clear
    explanation if it's unsupported."""
    entry = ENVIRONMENTS.get(env_key)
    if entry is None:
        raise ValueError(
            f"--env {env_key} is not supported: no available checkpoint's observation "
            "space could be matched to current gymnasium mujoco envs with a public kwarg "
            "(see the module docstring in scripts/envs.py). Supported: "
            f"{SUPPORTED_ENVIRONMENTS}"
        )
    return entry


def load_expert(spec: EnvSpec):
    """Downloads and loads the expert checkpoint for `spec` with its
    matching stable_baselines3 algorithm class."""
    model_path = load_from_hub(repo_id=spec.repo_id, filename=spec.model_filename)
    return spec.algo.load(model_path)


def make_venv(spec: EnvSpec, n_envs: int = 1, **gym_kwargs):
    """Builds an n_envs-wide DummyVecEnv for `spec`, wrapped in the matching
    VecNormalize stats when the checkpoint was trained with one (see
    `EnvSpec.use_vecnorm`). Extra gym_kwargs (e.g. render_mode,
    max_episode_steps) are merged with spec.env_kwargs."""
    kwargs = {**spec.env_kwargs, **gym_kwargs}
    venv = DummyVecEnv([(lambda: gym.make(spec.env_id, **kwargs)) for _ in range(n_envs)])
    if spec.use_vecnorm:
        vecnorm_path = load_from_hub(repo_id=spec.repo_id, filename="vec_normalize.pkl")
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    return venv
