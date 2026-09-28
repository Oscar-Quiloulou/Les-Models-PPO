from stable_baselines3 import PPO
import gym

# 1. Charger le modèle sauvegardé
model = PPO.load("ppo_nasim_tinysmall")

# 2. Créer l'environnement
env = gym.make("nasim:TinySmall-v2")

# 3. Lancer l'agent (mode déterministe = meilleure action connue)
obs, _ = env.reset()
done = False
actions_log = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions_log.append(action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# 4. Analyser le chemin d'attaque
print("Actions exécutées :", actions_log)
env.render(mode='readable')
