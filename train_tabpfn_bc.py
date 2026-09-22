"""Fits TabPFN-3.5 (open weights) as a behavior-cloned policy directly on
the (state, action) expert demonstrations -- no training loop, just
in-context learning -- then evaluates it in the real environment and
compares against the expert.
"""
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from huggingface_sb3 import load_from_hub
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

ENV_ID = "HalfCheetah-v4"
REPO_ID = "sb3/ppo-HalfCheetah-v3"
N_EVAL_EPISODES = 10
N_DEMO_TRANSITIONS = 2000  # TabPFN context budget -- subsample from the 20k collected


def main():
    data = np.load("demos_halfcheetah.npz")
    states, actions, expert_returns = data["states"], data["actions"], data["expert_returns"]
    print(f"loaded {states.shape[0]} transitions, expert mean return={expert_returns.mean():.1f}")

    rng = np.random.default_rng(0)
    idx = rng.choice(states.shape[0], size=min(N_DEMO_TRANSITIONS, states.shape[0]), replace=False)
    X_train, y_train = states[idx], actions[idx]
    print(f"fitting TabPFN on {X_train.shape[0]} transitions ({X_train.shape[1]} obs dims -> {y_train.shape[1]} action dims)")

    # one regressor per action dimension (TabPFNRegressor is single-output)
    regressors = []
    for d in range(y_train.shape[1]):
        reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST)
        reg.fit(X_train, y_train[:, d])
        regressors.append(reg)
    print("fit done (in-context, no training loop)")

    def tabpfn_policy(obs_batch):
        preds = np.stack([reg.predict(obs_batch) for reg in regressors], axis=1)
        return preds

    vecnorm_path = load_from_hub(repo_id=REPO_ID, filename="vec_normalize.pkl")
    venv = DummyVecEnv([lambda: gym.make(ENV_ID)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    tabpfn_returns = []
    for ep in range(N_EVAL_EPISODES):
        obs = venv.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = tabpfn_policy(obs)
            obs, reward, dones, infos = venv.step(action)
            ep_return += reward[0]
            done = bool(dones[0])
        tabpfn_returns.append(ep_return)
        print(f"tabpfn-bc episode {ep}: return={ep_return:.1f}")

    print(f"\nexpert return:    mean={expert_returns.mean():.1f} +/- {expert_returns.std():.1f}")
    print(f"tabpfn-bc return: mean={np.mean(tabpfn_returns):.1f} +/- {np.std(tabpfn_returns):.1f}")

    np.savez("results.npz", expert_returns=expert_returns, tabpfn_returns=tabpfn_returns)


if __name__ == "__main__":
    main()
