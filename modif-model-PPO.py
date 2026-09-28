import torch
from stable_baselines3 import PPO
import gym

env = gym.make("nasim:TinySmall-v2")

# Définir une architecture personnalisée
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256],              # Acteur : 2 couches de 256 neurones
        vf=[256, 256]               # Critique : 2 couches de 256 neurones
    ),
    activation_fn=torch.nn.ReLU
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=2.5e-4,   # Taux d'apprentissage
    n_steps=1024,           # Steps avant mise à jour
    batch_size=128,         # Taille du mini-batch
    clip_range=0.2,         # Ratio de clipping PPO
    gamma=0.99              # Facteur de discount
)
model.learn(total_timesteps=1_000_000)
