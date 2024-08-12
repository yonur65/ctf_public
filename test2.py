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
num_games = 2  # Toplam oyun sayısı
timesteps_per_game = 10  # Her oyun için zaman adımı
win_rates = []
evaluation_interval = 100

# Modeli eğitin ve kazanma oranlarını kaydedin
for game in range(num_games):
    obs, _ = env.reset(
        map_size=20,
        policy_red=getattr(policy, 'Roomba')(),
        policy_blue=getattr(policy, 'Roomba')()
    )
    #if game % evaluation_interval == 0:
        # Modeli belirli bir adım sayısında eğit
       # model.learn(total_timesteps=timesteps_per_game * evaluation_interval)
    model.learn(total_timesteps=timesteps_per_game)
    rewards = []
    wins = 0
    
    for _ in range(timesteps_per_game):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, _, done, info = env.step(action)
       # model.remember(obs, action, reward, done)
        rewards.append(reward)
        if 'win' in info and info['win']:
            wins += 1
        if done:
            obs, _ = env.reset(
                map_size=20,
                policy_red=getattr(policy, 'Roomba')(),
                policy_blue=getattr(policy, 'Roomba')()
            )
    
    win_rate = sum(rewards) #wins / timesteps_per_game
    win_rates.append(win_rate)
    print(f"Game {game + 1}/{num_games}: Win Rate = {win_rate:.2f}")

# Modeli kaydedin
model.save("ppo_ctf_model")

# Kazanma oranı grafiğini oluşturun
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, num_games + 1), win_rates, marker='o', linestyle='-', color='b')
plt.title('Win Rate per Game')
plt.xlabel('Game Number')
plt.ylabel('Win Rate')
plt.grid(True)
plt.show()
plt.savefig('win_rate_plot.png')

# Ortamı kapatın
env.close()

















""" from gym import spaces
class Board(spaces.Space):
    
    def __init__(self, _shape=None, dtype=np.uint8):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)

        if _shape is None:
            self.shape = (20, 20, NUM_CHANNEL)
        else:
            assert _shape[2] == NUM_CHANNEL
            self.shape = tuple(_shape)
        super(Board, self).__init__(self.shape, self.dtype)

    def __repr__(self):
        return "Board" + str(self.shape)

    def sample(self):
        map_obj = [NUM_BLUE, NUM_BLUE_UAV, NUM_RED, NUM_RED_UAV, NUM_GRAY]
        state, _, _ = gen_random_map('map',
                self.shape[0], rand_zones=False, map_obj=map_obj)
        return state"""
