
import mo_gymnasium as mo_gym
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, RecordVideo

class RewardScalarizationWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sum(reward))

ACTION_SET = [0, 1, 128, 129, 130, 131]

env = mo_gym.make(
    "mo-supermario-v0",
    render_mode="rgb_array"   # REQUIRED for video
)

env = RewardScalarizationWrapper(env)
env = ResizeObservation(env, (84, 84))
env = GrayscaleObservation(env, keep_dim=True)

env = RecordVideo(
    env,
    video_folder="./videos",
    episode_trigger=lambda ep: ep == 0, 
    disable_logger=True
)

obs, info = env.reset()

for step in range(500):
    action = np.random.choice(ACTION_SET)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
print("Mario video recorded successfully")
