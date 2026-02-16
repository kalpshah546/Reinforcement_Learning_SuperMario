from stable_baselines3 import PPO
from env_wrapper import make_vec_env

env = make_vec_env(render=True)

model = PPO.load("models/best_model", env=env)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
