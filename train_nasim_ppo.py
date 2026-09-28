import gym
import nasim
from stable_baselines3 import PPO

if __name__ == "__main__":
    # 1. Créer l'environnement de simulation
    env = gym.make("nasim:TinySmall-v2")

    # 2. Initialiser l'agent PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    # 3. Entraîner l'agent
    model.learn(total_timesteps=500_000)

    # 4. Sauvegarder le modèle
    model.save("ppo_nasim_tinysmall")

    # 5. Tester l'agent
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Récompense totale : {total_reward}")
