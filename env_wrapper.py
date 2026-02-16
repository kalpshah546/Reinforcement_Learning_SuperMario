import mo_gymnasium as mo_gym
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import RecordVideo

class RewardScalarizationWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sum(reward))

class ActionMappingWrapper(gym.ActionWrapper):
    def __init__(self, env, action_set):
        super().__init__(env)
        self.action_set = action_set
        self.action_space = gym.spaces.Discrete(len(action_set))

    def action(self, act):
        return self.action_set[act]

ACTION_SET = [
    0, 128, 129, 130, 131,
    1, 2,
    64, 65, 66,
    16
]
#reducing the action set to speed up the learning process
# ACTION_SET = [
#     128,        
#     129,        
#     130,        
#     131         
# ]

def make_env(render=False):
    env = mo_gym.make(
        "mo-supermario-v0",
        render_mode="human" if render else None
    )
    env = RewardScalarizationWrapper(env)
    env = ActionMappingWrapper(env, ACTION_SET)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=True)
    return env

def make_vec_env(render=False):
    env = DummyVecEnv([lambda: make_env(render)])
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    return env


def make_eval_env(video_folder):
    env = mo_gym.make("mo-supermario-v0", render_mode="rgb_array")
    env = RewardScalarizationWrapper(env)
    env = ActionMappingWrapper(env, ACTION_SET)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=True)

    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,
        name_prefix="ppo_eval"
    )
    return env

def make_eval_vec_env(video_folder):
    env = DummyVecEnv([lambda: make_eval_env(video_folder)])
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    return env