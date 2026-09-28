from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from cyberbattle._env.cyberbattle_env import CyberBattleChain
from cyberbattle._env.flatten_wrapper import FlattenObservation

# 1. Créer l'environnement CyberBattleChain
env = CyberBattleChain(
    size=10,  # 10 machines dans la chaîne
    attacker_goal=...,
    defender_goal=...
)

# 2. Aplatir l'observation (dictionnaire → vecteur)
env_as_gym = FlattenObservation(env)

# 3. Vectoriser l'environnement
vec_env = DummyVecEnv([lambda: env_as_gym])

# 4. Initialiser et entraîner PPO
model_ppo = PPO("MultiInputPolicy", vec_env, verbose=1)
model_ppo.learn(total_timesteps=100_000)

# 5. Sauvegarder
model_ppo.save("ppo_trained_cyberbattlechain")
