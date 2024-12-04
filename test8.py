import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import psutil
import time

# Oyun ortamı sınıfı
class CaptureTheFlagEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    # Ortamı sıfırla
    def reset2(self):
        self.attacker_pos = [0, random.randint(0, self.grid_size - 1)]  # Saldırgan başlangıç pozisyonu
        self.defender_pos = [self.grid_size - 1, random.randint(0, self.grid_size - 1)]  # Savunmacı başlangıç pozisyonu
        self.flag_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]  # Bayrak pozisyonu
        return self.get_state()
    
        # Ortamı sıfırla
    def reset(self):
        self.attacker_pos = [0, random.randint(0, self.grid_size)]  # Saldırgan başlangıç pozisyonu
        self.defender_pos = [self.grid_size - 1, random.randint(0, self.grid_size)]  # Savunmacı başlangıç pozisyonu  self.grid_size - 1
        self.flag_pos = [random.randint(0, self.grid_size),random.randint(0, self.grid_size)]  # Bayrak pozisyonu random.randint(0, self.grid_size)
        return self.get_state()

    # Mevcut durumu al
    def get_state(self):
        return (tuple(self.attacker_pos), tuple(self.defender_pos), tuple(self.flag_pos))

    # Ajanları hareket ettir
    def step(self, attacker_action, defender_action, luck=True):
        prev_attacker_dist = self.get_distance(self.attacker_pos, self.flag_pos)
        prev_defender_dist = self.get_distance(self.defender_pos, self.attacker_pos)

        self.move(self.attacker_pos, attacker_action)
        self.move(self.defender_pos, defender_action)

        curr_attacker_dist = self.get_distance(self.attacker_pos, self.flag_pos)
        curr_defender_dist = self.get_distance(self.defender_pos, self.attacker_pos)

        reward = 0
        rreward = 0
        done = False

        # Saldırgan bayrağa ulaştı mı?
        if self.attacker_pos == self.flag_pos:
            reward = 20  # Ödül
            rreward = -5
            done = True  # Oyun bitti

        # Savunmacı saldırganı yakaladı mı?
        elif self.attacker_pos == self.defender_pos:
            reward = -5  # Ceza
            rreward = 20
            done = True  # Oyun bitti

        # Saldırganın bayrağa olan mesafesine göre ödül/ceza
        if curr_attacker_dist < prev_attacker_dist:
            rreward -= 1
            reward += 3  # Yaklaşıyor
        elif curr_attacker_dist > prev_attacker_dist:
            reward -= 2  # Uzaklaşıyor
        else:
            reward -= 1  # Mesafe aynı

        # Savunmacının saldırgana olan mesafesine göre ödül/ceza (tam tersi)
        if curr_defender_dist < prev_defender_dist:
            reward -= 1  # Savunmacı yaklaşıyor, saldırgan için ceza
            rreward += 4.5
        elif curr_defender_dist > prev_defender_dist:
            rreward -= 2
        else:
            rreward -= 1

        if luck:
            reward += random.randint(0, 4)
            rreward += random.randint(0, 4)

        return self.get_state(), reward, rreward, done

    # Ajanı hareket ettir
    def move(self, position, action):
        if action == 0 and position[0] > 0:  # Yukarı
            position[0] -= 1
        elif action == 1 and position[0] < self.grid_size - 1:  # Aşağı
            position[0] += 1
        elif action == 2 and position[1] > 0:  # Sol
            position[1] -= 1
        elif action == 3 and position[1] < self.grid_size - 1:  # Sağ
            position[1] += 1
        # 4 ise bekle, hiçbir şey yapma

    # İki nokta arasındaki Manhattan mesafesini hesapla
    def get_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Temel Ajan Sınıfı
class BaseAgent:
    def choose_action(self, state):
        raise NotImplementedError

    def learn(self, *args):
        pass

    def check_state_exist(self, state):
        pass

# Q-Öğrenme ajanı
class QLearningAgent(BaseAgent):
    def __init__(self, actions, alpha=0.7, gamma=0.99, epsilon=1):
        self.q_table = {}  # Q-değerleri tablosu
        self.actions = actions  # Eylem seti
        self.alpha = alpha  # Öğrenme hızı
        self.gamma = gamma  # İndirim faktörü
        self.epsilon = epsilon  # Keşif olasılığı

        self.initial_alpha = alpha
        self.alpha_min = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Epsilon'un her bölümde azalacağı oran

    # Eylem seç
    def choose_action(self, state):
        state = str(state)
        self.check_state_exist(state)
        if np.random.uniform(0, 1) < self.epsilon:
            # Rastgele eylem seç (keşif)
            action = np.random.choice(self.actions)
        else:
            # En iyi eylemi seç (sömürü)
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            max_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(max_actions)
        return action

    # Q-değerlerini güncelle
    def learn(self, s, a, r, s_):
        s, s_ = str(s), str(s_)
        self.check_state_exist(s)
        if s_ != 'terminal':
            self.check_state_exist(s_)
            q_target = r + self.gamma * np.max(self.q_table[s_])
        else:
            q_target = r
        q_predict = self.q_table[s][a]
        self.q_table[s][a] += self.alpha * (q_target - q_predict)

    # Durum Q-değerleri tablosunda yoksa ekle
    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
    
    def update_parameters(self, episode, total_episodes):
        # Alpha'yı güncelle: Bölüm ilerledikçe azalsın
        if self.alpha > self.alpha_min:
            self.alpha = self.initial_alpha * (1 - (episode / total_episodes))
        # Epsilon'u güncelle: Bölüm ilerledikçe azalır, epsilon_min sınırına kadar
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Q-Öğrenme ajanı
class QLearningAgent2(BaseAgent):
    def __init__(self, actions, alpha=0.2, gamma=0.9, epsilon=0.2):
        self.q_table = {}  # Q-değerleri tablosu
        self.actions = actions  # Eylem seti
        self.alpha = alpha  # Öğrenme hızı
        self.gamma = gamma  # İndirim faktörü
        self.epsilon = epsilon  # Keşif olasılığı

    # Eylem seç
    def choose_action(self, state):
        state = str(state)
        self.check_state_exist(state)
        if np.random.uniform(0, 1) < self.epsilon:
            # Rastgele eylem seç (keşif)
            action = np.random.choice(self.actions)
        else:
            # En iyi eylemi seç (sömürü)
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            max_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(max_actions)
        return action

    # Q-değerlerini güncelle
    def learn(self, s, a, r, s_):
        s, s_ = str(s), str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * np.max(self.q_table[s_])  # Q-learning
        else:
            q_target = r  # Terminal state
        self.q_table[s][a] += self.alpha * (q_target - q_predict)  # Güncelleme

    # Durum Q-değerleri tablosunda yoksa ekle
    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

    def update_parameters(self, episode, total_episodes):
        pass

# SARSA ajanı
class SARSAAgent(BaseAgent):
    def __init__(self, actions, alpha=0.2, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        state = str(state)
        self.check_state_exist(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            max_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(max_actions)
        return action

    def learn(self, s, a, r, s_, a_):
        s, s_ = str(s), str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[s_][a_]  # SARSA
        else:
            q_target = r
        self.q_table[s][a] += self.alpha * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

# Double Q-Öğrenme ajanı
class DoubleQLearningAgent(BaseAgent):
    def __init__(self, actions, alpha=0.2, gamma=0.9, epsilon=0.2):
        self.q_table1 = {}
        self.q_table2 = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        state = str(state)
        self.check_state_exist(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q_values = self.q_table1[state] + self.q_table2[state]
            max_q = np.max(q_values)
            max_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(max_actions)
        return action

    def learn(self, s, a, r, s_):
        s, s_ = str(s), str(s_)
        self.check_state_exist(s_)
        if s_ != 'terminal':
            action = np.argmax(self.q_table1[s_])
            q_target = r + self.gamma * self.q_table2[s_][action]
        else:
            q_target = r
        if np.random.uniform(0, 1) < 0.5:
            self.q_table1[s][a] += self.alpha * (q_target - self.q_table1[s][a])
        else:
            self.q_table2[s][a] += self.alpha * (q_target - self.q_table2[s][a])

    def check_state_exist(self, state):
        if state not in self.q_table1:
            self.q_table1[state] = np.zeros(len(self.actions))
            self.q_table2[state] = np.zeros(len(self.actions))

# Expected SARSA ajanı
class ExpectedSARSAAgent(BaseAgent):
    def __init__(self, actions, alpha=0.2, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        state = str(state)
        self.check_state_exist(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            max_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(max_actions)
        return action

    def learn(self, s, a, r, s_):
        s, s_ = str(s), str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            # Hesaplanan beklenen Q değeri
            policy = np.ones(len(self.actions)) * self.epsilon / len(self.actions)
            best_action = np.argmax(self.q_table[s_])
            policy[best_action] += (1.0 - self.epsilon)
            expected_q = np.dot(self.q_table[s_], policy)
            q_target = r + self.gamma * expected_q
        else:
            q_target = r
        self.q_table[s][a] += self.alpha * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

# Monte Carlo ajanı
class MonteCarloAgent(BaseAgent):
    def __init__(self, actions, gamma=0.9, epsilon=0.2):
        self.Q = defaultdict(lambda: np.zeros(len(actions)))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        state = str(state)
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            max_q = np.max(self.Q[state])
            max_actions = [i for i, q in enumerate(self.Q[state]) if q == max_q]
            return np.random.choice(max_actions)

    def learn(self, episodes):
        for episode in episodes:
            G = 0
            visited = set()
            for step in reversed(episode):
                state, action, reward = step
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    self.returns[state][action].append(G)
                    self.Q[state][action] = np.mean(self.returns[state][action])
                    visited.add((state, action))

# Dyna-Q ajanı
class DynaQAgent(BaseAgent):
    def __init__(self, actions, alpha=0.2, gamma=0.9, epsilon=0.2, planning_steps=5):
        self.q_table = {}
        self.model = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

    def choose_action(self, state):
        state = str(state)
        self.check_state_exist(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            max_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(max_actions)
        return action

    def learn(self, s, a, r, s_):
        s, s_ = str(s), str(s_)
        self.check_state_exist(s_)
        # Güncelleme
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * np.max(self.q_table[s_])
        else:
            q_target = r
        self.q_table[s][a] += self.alpha * (q_target - q_predict)
        # Modeli güncelle
        self.model[(s, a)] = (r, s_)
        # Planlama
        for _ in range(self.planning_steps):
            (state_p, action_p), (reward_p, next_state_p) = random.choice(list(self.model.items()))
            self.check_state_exist(next_state_p)
            if next_state_p != 'terminal':
                q_target_p = reward_p + self.gamma * np.max(self.q_table[next_state_p])
            else:
                q_target_p = reward_p
            self.q_table[state_p][action_p] += self.alpha * (q_target_p - self.q_table[state_p][action_p])

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

# Rastgele ajan (Önceden tanımlı)
class RandomAgent(BaseAgent):
    def __init__(self, actions):
        self.actions = actions

    def choose_action(self, state):
        return np.random.choice(self.actions)

    def learn(self, s, a, r, s_):
        pass  # Rastgele ajan öğrenmez


def get_system_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_mb": mem_info.rss / (1024 * 1024)  # RSS belleği MB cinsinden alır
    }

def split_state(state, fog=False):
    if fog:
        return state[0], state[1]
    return state, state

# Eğitim ve test fonksiyonu
def train_and_test():
    env = CaptureTheFlagEnv()
    actions = [0, 1, 2, 3, 4]  # Yukarı, Aşağı, Sol, Sağ, Bekle

    # Ajan modelleri
    agent_models = {
        'Q-Learning': {
            'attacker': QLearningAgent(actions),
            'defender': QLearningAgent(actions)
        },
    }
    agent_model2 = {
        'Q-Learning': {
            'attacker': QLearningAgent(actions),
            'defender': QLearningAgent(actions)
        },
        'SARSA': {
            'attacker': SARSAAgent(actions),
            'defender': SARSAAgent(actions)
        },
        'Double Q-Learning': {
            'attacker': DoubleQLearningAgent(actions),
            'defender': DoubleQLearningAgent(actions)
        },
        'Expected SARSA': {
            'attacker': ExpectedSARSAAgent(actions),
            'defender': ExpectedSARSAAgent(actions)
        },
        'Monte Carlo': {
            'attacker': MonteCarloAgent(actions),
            'defender': MonteCarloAgent(actions)
        },
        'Dyna-Q': {
            'attacker': DynaQAgent(actions),
            'defender': DynaQAgent(actions)
        }
    }
    # Rastgele ajanlar
    random_attacker = RandomAgent(actions)
    random_defender = RandomAgent(actions)

    total_episodes = 20000
    test_interval = 1000
    test_episodes = 500
    max_steps = 500 

    # Sonuçları saklamak için
    results = {model: {'trained_attacker': [], 'trained_defender': []} for model in agent_models.keys()}
    performance_metrics = {model: {"cpu": [], "memory": [], "time":[], "step":[],"total_cpu":0, "total_ram":0, "total_step":0, "total_time":0 } for model in agent_models.keys()}
    #total_step_count = 0
    # Eğitim ve periyodik test
    for episode in range(1, total_episodes + 1):
        print(f"Episode: {episode}" )
        mc_attacker_steps = []
        mc_defender_steps = []
        # Her model için ayrı eğitim
        for model_name, agent in agent_models.items():
            state = env.reset()
            attacker_agent = agent['attacker']
            defender_agent = agent['defender']
            done = False
            step_count = 0  # Adım sayacını başlat
            cpu_total=0
            ram_total=0
            start_time = time.time()
            while not done and step_count < max_steps:
                state_attacker, state_defender = split_state(state)
                step_count += 1
                # Ajan eylem seçimi
                attacker_action = attacker_agent.choose_action(state_attacker)
                defender_action = defender_agent.choose_action(state_defender)  # Aynı modelden defender

                next_state, reward, rreward, done = env.step(attacker_action, defender_action)

                if done:
                    next_state_str = 'terminal'
                else:
                    next_state_str = next_state

                # Ajanı eğit
                if model_name == 'SARSA':
                    # SARSA için sonraki eylemi de seçmek gerekiyor
                    next_attacker_action = attacker_agent.choose_action(next_state)
                    next_defender_action = defender_agent.choose_action(next_state)
                    attacker_agent.learn(state_attacker, attacker_action, reward, next_state, next_attacker_action)
                    defender_agent.learn(state_defender, attacker_action, rreward, next_state, next_defender_action)
                elif model_name == 'Monte Carlo':
                    # Monte Carlo için episodic yaklaşımlar gerek
                    # Burada basit bir yaklaşım kullanılabilir
                    #pass  # Detaylı implementasyon gerekli
                    mc_attacker_steps.append((state_attacker, attacker_action, reward))
                    # Collect steps for defender
                    mc_defender_steps.append((state_defender, defender_action, rreward))
                else:
                    attacker_agent.learn(state_attacker, attacker_action, reward, next_state_str)
                    defender_agent.learn(state_defender, attacker_action, rreward, next_state_str)

                        # Performans ölçümlerini al
                usage = get_system_usage()
                cpu_total +=usage["cpu_percent"]
                ram_total +=usage["memory_mb"]
                state = next_state
            #total_step_count +=step_count 
            if model_name == 'Q-Learning':
                attacker_agent.update_parameters(episode, total_episodes)
                defender_agent.update_parameters(episode, total_episodes)
            performance_metrics[model_name]["total_step"] += step_count
            performance_metrics[model_name]["total_time"] += time.time() - start_time
            performance_metrics[model_name]["total_cpu"] += cpu_total/step_count
            performance_metrics[model_name]["total_ram"] += ram_total/step_count
            performance_metrics[model_name]["cpu"].append(np.mean(performance_metrics[model_name]["total_cpu"]))
            performance_metrics[model_name]["memory"].append(np.mean(performance_metrics[model_name]["total_ram"]))
            performance_metrics[model_name]["time"].append(performance_metrics[model_name]["total_time"]/episode)
            performance_metrics[model_name]["step"].append(performance_metrics[model_name]["total_step"]/episode)

        # After the episode, update Monte Carlo agents
        if 'Monte Carlo' in agent_models:
            mc_attacker_agent = agent_models['Monte Carlo']['attacker']
            mc_defender_agent = agent_models['Monte Carlo']['defender']
            # Learn from the collected episode
            mc_attacker_agent.learn([mc_attacker_steps])
            mc_defender_agent.learn([mc_defender_steps])

        # Her test_interval bölümde bir test yap
        if episode % test_interval == 0:
            for model_name, agent in agent_models.items():
                # Eğitimli saldırgan vs rastgele savunmacı
                trained_attacker_rewards = []
                for _ in range(test_episodes):
                    state = env.reset()
                    total_reward = 0
                    step_count = 0 
                    while True and step_count < max_steps:
                        state_attacker, state_defender = split_state(state)
                        step_count += 1
                        attacker_action = agent['attacker'].choose_action(state_attacker)
                        defender_action = random_defender.choose_action(state_defender)

                        next_state, reward, rreward, done = env.step(attacker_action, defender_action)

                        state = next_state
                        total_reward += reward

                        if done:
                            trained_attacker_rewards.append(total_reward)
                            break

                # Rastgele saldırgan vs eğitimli savunmacı
                trained_defender_rewards = []
                for _ in range(test_episodes):
                    state = env.reset()
                    total_reward = 0
                    step_count = 0 
                    while True and step_count < max_steps:
                        state_attacker, state_defender = split_state(state)
                        step_count += 1
                        attacker_action = random_attacker.choose_action(state_attacker)
                        defender_action = agent['defender'].choose_action(state_defender)

                        next_state, reward, rreward, done = env.step(attacker_action, defender_action)

                        state = next_state
                        total_reward += rreward

                        if done:
                            trained_defender_rewards.append(total_reward)
                            break

                # Ortalama ödülleri kaydet
                trained_avg_reward = np.mean(trained_attacker_rewards)
                trained_defender_avg_reward = np.mean(trained_defender_rewards)

                results[model_name]['trained_attacker'].append(trained_avg_reward)
                results[model_name]['trained_defender'].append(trained_defender_avg_reward)

            print(f"Episode {episode} completed and tested.")

    # Sonuçları grafikle
    plt.figure(figsize=(14, 14))

    # Eğitimli saldırganlar
    plt.subplot(3, 2, 1)
    for model_name in agent_models.keys():
        plt.plot(range(test_interval, total_episodes + 1, test_interval),
                 results[model_name]['trained_attacker'],
                 label=f'{model_name} Attacker')
    plt.xlabel('Eğitim Bölümü Sayısı')
    plt.ylabel('Ortalama Ödül (Saldırgan)')
    plt.title('Eğitimli Saldırganların Performansı')
    plt.legend()

    # Eğitimli savunmacılar
    plt.subplot(3, 2, 2)
    for model_name in agent_models.keys():
        plt.plot(range(test_interval, total_episodes + 1, test_interval),
                 results[model_name]['trained_defender'],
                 label=f'{model_name} Defender')
    plt.xlabel('Eğitim Bölümü Sayısı')
    plt.ylabel('Ortalama Ödül (Savunmacı)')
    plt.title('Eğitimli Savunmacıların Performansı')
    plt.legend()

    # CPU kullanımı

    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(3, 2, 3)
    for model_name in performance_metrics.keys():
        plt.plot(performance_metrics[model_name]["cpu"], label=f'{model_name} CPU Kullanımı')
        #smoothed_cpu = moving_average(performance_metrics[model_name]["cpu"], window_size=50)
        #plt.plot(smoothed_cpu, label=f'{model_name}', alpha=0.7)
    plt.xlabel('Adım Sayısı')
    plt.ylabel('CPU Yüzdesi (%)')
    plt.title('Model Eğitiminde CPU Kullanımı')
    plt.legend()

    # RAM kullanımı
    plt.subplot(3, 2, 4)
    for model_name in performance_metrics.keys():
        plt.plot(performance_metrics[model_name]["memory"], label=f'{model_name}')
    plt.xlabel('Adım Sayısı')
    plt.ylabel('RAM Kullanımı (KB)')
    plt.title('Model Eğitiminde RAM Kullanımı')
    plt.legend()

    # RAM kullanımı
    plt.subplot(3, 2, 5)
    for model_name in performance_metrics.keys():
        plt.plot(performance_metrics[model_name]["time"], label=f'{model_name}')
    plt.xlabel('Adım Sayısı')
    plt.ylabel('Zaman Kullanımı (S)')
    plt.title('Model Eğitiminde Süre Kullanımı')
    plt.legend()

    # RAM kullanımı
    plt.subplot(3, 2, 6)
    for model_name in performance_metrics.keys():
        plt.plot(performance_metrics[model_name]["step"], label=f'{model_name}')
    plt.xlabel('Adım Sayısı')
    plt.ylabel('Adım kullanımı')
    plt.title('Model Eğitiminde Adım Kullanımı')
    plt.legend()

    plt.tight_layout()
    # Grafiği kaydet
    plt.savefig('test8_agent_performance_comparison_LT_FF_20K_v17.png')
    plt.show()
    print(performance_metrics)

if __name__ == "__main__":
    start_time = time.time() 
    train_and_test()
    print(time.time() - start_time)
