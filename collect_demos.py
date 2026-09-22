"""Loads a pretrained PPO expert for HalfCheetah and rolls it out (with the
correct VecNormalize observation stats) to collect (state, action)
demonstration pairs for behavior cloning.
"""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from huggingface_sb3 import load_from_hub

ENV_ID = "HalfCheetah-v4"
REPO_ID = "sb3/ppo-HalfCheetah-v3"
N_EPISODES = 20


def main():
    model_path = load_from_hub(repo_id=REPO_ID, filename="ppo-HalfCheetah-v3.zip")
    vecnorm_path = load_from_hub(repo_id=REPO_ID, filename="vec_normalize.pkl")
    model = PPO.load(model_path)

    venv = DummyVecEnv([lambda: gym.make(ENV_ID)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    states, actions = [], []
    returns = []

    for ep in range(N_EPISODES):
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
        print(f"episode {ep}: return={ep_return:.1f}, steps so far={len(states)}")

    states = np.array(states)
    actions = np.array(actions)
    print(f"\ncollected {states.shape[0]} transitions, obs_dim={states.shape[1]}, act_dim={actions.shape[1]}")
    print(f"expert return: mean={np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    np.savez("demos_halfcheetah.npz", states=states, actions=actions, expert_returns=returns)
    print("saved demos_halfcheetah.npz")


if __name__ == "__main__":
    main()
