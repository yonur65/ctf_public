import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.atari import flag_capture_v2
from stable_baselines3 import PPO
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from stable_baselines3.common.vec_env import VecMonitor

# Oyun ortamını Gym uyumlu hale getir
env = flag_capture_v2.parallel_env()
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

# Ortamı izlemek için monitor ekleyelim
env = VecMonitor(env)

# Kırmızı takımın PPO modeli (saldırı)
red_team_model = PPO("CnnPolicy", env, verbose=0)
# Mavi takımın PPO modeli (savunma)
blue_team_model = PPO("CnnPolicy", env, verbose=0)

# Eğitim Süreci
episodes = 15  # Toplam episode sayısı
max_steps = 10000  # Her episode için maksimum adım sayısı
red_team_rewards = []
blue_team_rewards = []
test_rewards = []

def is_near_opponent(red_pos, blue_pos):
    # Ajanların birbirine olan mesafesini kontrol eder
    return np.linalg.norm(np.array(red_pos) - np.array(blue_pos)) < 1.5

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

            # Bayrağın güncel konumunu al
            flag_position = infos['flag_position']  # Ortamdan bayrak pozisyonunu çek

            # Ajanların pozisyon bilgilerini al
            red_pos = infos['player_red']['position']
            blue_pos = infos['player_blue']['position']

            # Bayrağa yakınlık ödülü
            red_distance_to_flag = np.linalg.norm(np.array(red_pos) - np.array(flag_position))
            blue_distance_to_flag = np.linalg.norm(np.array(blue_pos) - np.array(flag_position))

            # Bayrağa yeterince yakınsa ekstra ödül ver
            if red_distance_to_flag < 2.0:
                rewards[0] += 0.1  # Kırmızı takım ekstra ödül kazanır

            if blue_distance_to_flag < 2.0:
                rewards[1] += 0.1  # Mavi takım ekstra ödül kazanır

            # Rakibe yakın olma ödülü
            if is_near_opponent(red_pos, blue_pos):
                rewards[0] += 0.2  # Kırmızı takım ekstra ödül kazanır
                rewards[1] += 0.2  # Mavi takım da ödül kazanır

            # Ödülleri toplama
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

            # Bayrağın güncel konumunu al
            flag_position = infos['flag_position']  # Ortamdan bayrak pozisyonunu çek

            # Ajanların pozisyon bilgilerini al
            red_pos = infos['player_red']['position']
            blue_pos = infos['player_blue']['position']

            # Bayrağa yakınlık ödülü
            red_distance_to_flag = np.linalg.norm(np.array(red_pos) - np.array(flag_position))
            blue_distance_to_flag = np.linalg.norm(np.array(blue_pos) - np.array(flag_position))

            # Bayrağa yeterince yakınsa ekstra ödül ver
            if red_distance_to_flag < 2.0:
                rewards[0] += 0.1  # Kırmızı takım ekstra ödül kazanır

            if blue_distance_to_flag < 2.0:
                rewards[1] += 0.1  # Mavi takım ekstra ödül kazanır

            # Rakibe yakın olma ödülü
            if is_near_opponent(red_pos, blue_pos):
                rewards[0] += 0.2  # Kırmızı takım ekstra ödül kazanır
                rewards[1] += 0.2  # Mavi takım da ödül kazanır

            # Ödülleri toplama
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
plt.savefig('test4.png')