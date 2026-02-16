
import mo_gymnasium as mo_gym
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
class RewardScalarizationWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sum(reward))


ACTION_SET = [
    0, 128, 129, 130, 131, # 0 is for nothing , and rest are right +jump,run
    1, 2, # 1 is jump and 2 is run
    64, 65, 66, # left +jump,run
    16 #up
]

def make_env():
    env = mo_gym.make("mo-supermario-v0", render_mode="human")
    env = RewardScalarizationWrapper(env)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=True)
    return env


env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4, channels_order="last")

obs = env.reset()

for step in range(1000):
    action = [np.random.choice(ACTION_SET)]
    obs, rewards, dones, infos = env.step(action)

    if dones[0]:
        obs = env.reset()

env.close()

