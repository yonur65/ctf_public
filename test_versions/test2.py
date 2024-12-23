import gymnasium as gym
import gym_cap  # gym_cap environment
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import gym_cap.heuristic as policy

class CompatibleEnv(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        output = self.env.reset(**kwargs)
        if isinstance(output, tuple):
            return output[0], {}  # Return observation and an empty info dict
        return output, {}

# Çevreyi sarmalayın
env = CompatibleEnv(gym.make('cap-v0'))

# PPO modelini oluşturun
model = PPO("MlpPolicy", env, verbose=1)

# Eğitim adımları ve kazanma oranlarını kaydetmek için değişkenler
num_games = 10000  # Toplam oyun sayısı (artırıldı)
evaluation_interval = 30  # Değerlendirme aralığı
timesteps_per_game = 1000  # Her oyun için zaman adımı

total_rewards = []

#print(env.MAX_STEP)

env.CONTROL_ALL = False


# Modeli eğitin ve kazanma oranlarını kaydedin
for game in range(1, num_games + 1):
    obs, _ = env.reset(
        map_size=20,
        policy_red=getattr(policy, 'Zeros')(),
        policy_blue=getattr(policy, 'Roomba')()
    )
    
    # Modeli belirli bir aralıkla eğitin
    if game % evaluation_interval == 0:
        model.learn(total_timesteps=timesteps_per_game * evaluation_interval)
        
    # Test Oyunları ve Toplam Ödül Hesaplama
    total_reward = 0
    for _ in range(10):  # Her bir eğitim değerlendirmesinde 10 oyun test etme
        obs, _ = env.reset(
            map_size=20,
            policy_red=getattr(policy, 'Zeros')(),
            policy_blue=getattr(policy, 'Roomba')()
        )
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, done, info = env.step(action)
            env.render()
            episode_reward += reward
        total_reward += episode_reward
    
    avg_reward = total_reward / 10.0  # Her bir değerlendirme oyununda ortalama ödül
    total_rewards.append(avg_reward)
    print(f"Evaluation Game {game}/{num_games}: Average Reward = {avg_reward:.2f}")

# Modeli kaydedin
model.save("ppo_ctf_model")

# Ortalama Ödül Grafiğini Oluşturun
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(total_rewards) + 1) * evaluation_interval, total_rewards, marker='o', linestyle='-', color='b')
plt.title('Average Reward per Evaluation Interval')
plt.xlabel('Training Steps')
plt.ylabel('Average Reward')
plt.grid(True)
plt.savefig('test2_blue_train_10K_.png')
plt.show()

# Ortamı kapatın
env.close()
