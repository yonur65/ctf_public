import gym
from pettingzoo.atari import flag_capture_v2
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import pettingzoo_env_to_vec_env_v1, resize_v1, color_reduction_v0, frame_stack_v1

# Oyun ortamını oluştur ve AEC'den paralel ortama çevir
env = flag_capture_v2.env()  # AEC formatında başlatılıyor
env = aec_to_parallel(env)  # Paralel ortama çevir


# Ortamın şekil boyutlarını küçült ve vektörleştirilmiş hale getir
env = resize_v1(env, x_size=84, y_size=84)  # Gözlem boyutlarını düşürüyoruz
env = color_reduction_v0(env, mode='full')  # Gözlemleri siyah-beyaza dönüştürüyoruz
env = frame_stack_v1(env, 4)  # Gözlem geçmişini (frame stack) kullanarak 4 çerçeve saklıyoruz


# Gym ile uyumlu hale getir
env = pettingzoo_env_to_vec_env_v1(env)

# PPO modelini oluştur ve GPU kullanacak şekilde ayarla
model = PPO("CnnPolicy", env, verbose=1, device="cuda")  # GPU'yu kullanmak için device="cuda" eklendi

# Eğitimden önce kırmızı takımın başarısı
n_games = 1
red_wins_before = 0

for game in range(n_games):
    obs = env.reset()
    dones = [False] * env.num_envs  # Tüm ajanlar için done durumu
    while not all(dones):  # Tüm ajanlar bitene kadar devam et
        actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])  # Rastgele aksiyonlar üret
        obs, rewards, terms, dones, infos = env.step(actions)
        # `terms` ve `truncs` birleştirerek tüm ajanların tamamlanıp tamamlanmadığını kontrol et
        #dones = {agent: terms[agent] or truncs[agent] for agent in env.possible_agents}
    print(rewards)
    if rewards[0] > rewards[1]:  # Kırmızı takımın kazanıp kazanmadığını kontrol et
        red_wins_before += 1
print(f"Red team wins before training: {red_wins_before}/{n_games}")

# Modeli eğit
model.learn(total_timesteps=1000000)

# Eğitimden sonra kırmızı takımın başarısı
red_wins_after = 0

for game in range(n_games):
    obs = env.reset()
    dones = [False] * env.num_envs  # Ajan başına done durumu
    while not all(dones):  # Tüm ajanlar bitene kadar devam et
        actions = {}
        for agent in env.possible_agents:
            if not dones[agent]:
                if 'red' in agent:
                    action, _ = model.predict(obs[agent])  # Modelden aksiyon tahmini al
                    actions[agent] = action
                else:
                    actions[agent] = env.action_space.sample()  # Diğer ajanlar için rastgele aksiyon
        obs, rewards, terms, dones, infos = env.step(actions)
        # `terms` ve `truncs` birleştirerek tüm ajanların tamamlanıp tamamlanmadığını kontrol et
        #dones = {agent: terms[agent] or truncs[agent] for agent in env.possible_agents}
    if rewards[0] > rewards[1]:  # Kırmızı takımın kazandığı oyunlar
        red_wins_after += 1

print(f"Red team wins after training: {red_wins_after}/{n_games}")

# Sonuçları görselleştir
labels = ['Before Training', 'After Training']
red_wins = [red_wins_before, red_wins_after]

plt.figure(figsize=(8, 6))
plt.bar(labels, red_wins, color=['blue', 'green'])
plt.ylabel('Red Team Wins')
plt.title('Red Team Wins Before and After Training')

# Grafiği kaydet
plt.savefig('red_team_wins_comparison.png')

# Grafiği göster
plt.show()
