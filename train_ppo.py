from stable_baselines3 import PPO
from env_wrapper import make_vec_env, make_eval_vec_env
import os
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

def get_latest_checkpoint(folder="models"):
    checkpoints = [
        f for f in os.listdir(folder)
        if f.startswith("ppo_mario_") and f.endswith("_steps.zip")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("_")[2]))
    return os.path.join(folder, checkpoints[-1])

env = make_vec_env(render=False)
eval_env = make_eval_vec_env(video_folder="videos")

checkpoint_path = get_latest_checkpoint()

if checkpoint_path:
    print(f"Resuming training from {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env)
else:
    print("Starting training from scratch")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        verbose=1,
        tensorboard_log="logs/ppo_tb"
    )

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="models/",
    name_prefix="ppo_mario"
)

eval_callback = EvalCallback(
    eval_env,
    eval_freq=100_000,
    n_eval_episodes=1,
    deterministic=True,
    best_model_save_path="models/",
    log_path="logs/eval/"
)

model.learn(
    total_timesteps=100_000,
    callback=[checkpoint_callback, eval_callback]
)

model.save("models/ppo_mario_final")
