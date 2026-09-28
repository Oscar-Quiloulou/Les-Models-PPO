import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

model_dir = "models/"
os.makedirs(model_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,           # Sauvegarder tous les 10 000 steps
    save_path=model_dir,
    name_prefix="ppo_cyberbattle"
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000, callback=checkpoint_callback)

# Reprise depuis un checkpoint
model = PPO.load("models/ppo_cyberbattle_50000_steps.zip", env=env)
model.learn(total_timesteps=200_000)
