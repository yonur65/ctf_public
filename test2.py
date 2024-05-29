import gymnasium as gym
import gym_cap  # gym_cap environment
from stable_baselines3 import PPO, DQN, A2C
import matplotlib.pyplot as plt
import numpy as np
import gym_cap.heuristic as policy

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


class CompatibleEnv(gym.Wrapper):
    def reset(self, **kwargs):
        # Ensure seed is not passed if not handled by the environment
        kwargs.pop('seed', None)

        # Reset and manage all returned values appropriately
        output = self.env.reset(**kwargs)
        if isinstance(output, tuple):
            # Only return the observation (assuming it's the first item)
            return output[0]
        return output  # Return directly if not a tuple
# Wrap your environment
enva = CompatibleEnv(gym.make('cap-v0'))
# Ortamı oluşturun
env = gym.make('cap-v0')
envx = gym.make('CartPole-v1')

# PPO modelini oluşturun
model = PPO("MlpPolicy", env, verbose=1)

# Eğitim adımları ve kazanma oranlarını kaydetmek için değişkenler
num_games = 500  # Toplam oyun sayısı
timesteps_per_game = 1000  # Her oyun için zaman adımı
win_rates = []
#model.learn(total_timesteps=timesteps_per_game)

# Modeli eğitin ve kazanma oranlarını kaydedin
obs, _ = env.reset(
                    map_size=20,
                    #config_path='/demo/base_setting.ini',
                    policy_red=getattr(policy, 'Roomba')(),
                    policy_blue=getattr(policy, 'Zeros')() #policy.Fighter() # Defense Random Fighter Zeros Roomba Patrol Spiral Policy blue_policy
                )
#obs = envx.reset()
for game in range(num_games):
    model.learn(total_timesteps=timesteps_per_game)
    rewards = []
    # Kazanma oranını hesaplayın
    
    wins = 0
    for _ in range(timesteps_per_game):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, _, done, info = env.step(action)
        rewards.append(reward)
        if 'win' in info and info['win']:
            wins += 1
        if done:
            obs, _ = env.reset()
    
    obs, _ = env.reset()
    win_rate = sum(rewards) / timesteps_per_game  #wins
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

# Ortamı kapatın
env.close()
