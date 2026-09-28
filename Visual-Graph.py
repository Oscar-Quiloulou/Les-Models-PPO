from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class RewardLoggerCallback(BaseCallback):
    """Enregistre les récompenses par épisode."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self):
        self.current_rewards += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True

callback = RewardLoggerCallback()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000, callback=callback)

plt.plot(callback.episode_rewards)
plt.xlabel("Épisode")
plt.ylabel("Récompense cumulée")
plt.title("Courbe d'apprentissage PPO")
plt.show()
