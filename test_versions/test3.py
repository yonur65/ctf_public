import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.atari import flag_capture_v2
from stable_baselines3 import PPO
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from stable_baselines3.common.vec_env import VecMonitor

# Oyun ortamını Gym uyumlu hale getir
env = flag_capture_v2.parallel_env()
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, 1, num_cpus=3, base_class='stable_baselines3')

# Ortamı izlemek için monitor ekleyelim
env = VecMonitor(env)

# Kırmızı takımın PPO modeli (saldırı)
red_team_model = PPO("CnnPolicy", env, verbose=0)
# Mavi takımın PPO modeli (savunma)
blue_team_model = PPO("CnnPolicy", env, verbose=0)

# Eğitim Süreci
episodes = 30000  # Toplam episode sayısı
max_steps = 500  # Her episode için maksimum adım sayısı
red_team_rewards = []
blue_team_rewards = []
test_rewards = []

for episode in range(episodes):
    obs = env.reset()
    red_team_episode_reward = 0
    blue_team_episode_reward = 0

    if (episode + 1) % 5 == 0:  # Her 5 episodda bir eğitilmemiş mavi takım ile savaş
        for step in range(max_steps):
            # Kırmızı takım bayrağı kapmaya çalışır
            red_action = red_team_model.predict(obs[0])[0]
            # Mavi takım rastgele hareket eder (eğitilmemiş model)
            blue_action = env.action_space.sample()

            actions = [red_action, blue_action]

            obs, rewards, done, infos = env.step(actions)
            
            red_team_episode_reward += rewards[0]

        test_rewards.append(red_team_episode_reward)  # Test sonuçlarını kaydet
        print(f"Test Episode {episode + 1}: Red Team Reward = {red_team_episode_reward}")
    else:
        for step in range(max_steps):
            # Kırmızı takım bayrağı kapmaya çalışır
            red_action = red_team_model.predict(obs[0])[0]
            # Mavi takım savunma yapar (eğitilen model)
            blue_action = blue_team_model.predict(obs[1])[0]

            actions = [red_action, blue_action]

            obs, rewards, done, infos = env.step(actions)
            
            red_team_episode_reward += rewards[0]
            blue_team_episode_reward += rewards[1]

        red_team_rewards.append(red_team_episode_reward)
        blue_team_rewards.append(blue_team_episode_reward)

# Eğitim ve Test Süreci Sonuçlarını Görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(red_team_rewards) + 1), red_team_rewards, label='Training: Red Team Rewards (vs. Trained Blue)')
plt.plot(range(5, episodes + 1, 5), test_rewards, label='Testing: Red Team Rewards (vs. Untrained Blue)', marker='o')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training and Testing Performance of Red Team')
plt.legend()
plt.show()
plt.savefig('test3.png')
