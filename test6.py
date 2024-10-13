import numpy as np
import matplotlib.pyplot as plt
import random


# Oyun ortamı sınıfı
class CaptureTheFlagEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    # Ortamı sıfırla
    def reset(self):
        self.attacker_pos = [0, random.randint(0, self.grid_size)]  # Saldırgan başlangıç pozisyonu
        self.defender_pos = [self.grid_size - 1, random.randint(0, self.grid_size)]  # Savunmacı başlangıç pozisyonu  self.grid_size - 1
        self.flag_pos = [random.randint(0, self.grid_size),random.randint(0, self.grid_size)]  # Bayrak pozisyonu random.randint(0, self.grid_size)
        return self.get_state()

    # Mevcut durumu al
    def get_state(self):
        return (tuple(self.attacker_pos), tuple(self.defender_pos), tuple(self.flag_pos) )

    # Ajanları hareket ettir
    def step(self, attacker_action, defender_action):
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
            rreward += 3
        elif curr_defender_dist > prev_defender_dist:
            #reward += 1  # Savunmacı uzaklaşıyor, saldırgan için ödül
            rreward -= 2
        else:
            #reward += 1  # Mesafe aynı, saldırgan için küçük ödül
            rreward -= 1

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

# Q-Öğrenme ajanı
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
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

# Rastgele ajan
class RandomAgent:
    def __init__(self, actions):
        self.actions = actions

    def choose_action(self, state):
        return np.random.choice(self.actions)

    def learn(self, s, a, r, s_):
        pass  # Rastgele ajan öğrenmez

# Eğitim ve test fonksiyonu
def train_and_test():
    env = CaptureTheFlagEnv()
    actions = [0, 1, 2, 3, 4]  # Yukarı, Aşağı, Sol, Sağ, Bekle

    # Eğitimli ajanlar
    attacker_agent = QLearningAgent(actions)
    defender_agent = QLearningAgent(actions)

    # Eğitilmemiş (rastgele) ajanlar RandomAgent
    random_attacker = QLearningAgent(actions)
    random_defender = QLearningAgent(actions)

    total_episodes = 200000
    test_interval = 5000
    test_episodes = 1000

    intervals = []
    trained_attacker_avg_rewards = []
    random_attacker_avg_rewards = []

    # Eğitim ve periyodik test
    for episode in range(1, total_episodes + 1):
        state = env.reset()
        total_reward = 0
        while True:
            attacker_action = attacker_agent.choose_action(state)
            defender_action = defender_agent.choose_action(state)

            next_state, reward, rreward, done = env.step(attacker_action, defender_action)

            if done:
                next_state_str = 'terminal'
            else:
                next_state_str = next_state

            # Ajanları eğit
            attacker_agent.learn(state, attacker_action, reward, next_state_str)
            defender_agent.learn(state, defender_action, rreward, next_state_str)

            state = next_state
            total_reward += reward

            if done:
                break

        # Her test_interval bölümde bir test yap
        if episode % test_interval == 0:
            # Eğitimli saldırgan vs rastgele savunmacı
            trained_attacker_rewards = []
            for _ in range(test_episodes):
                state = env.reset()
                total_reward = 0
                while True:
                    attacker_action = attacker_agent.choose_action(state)
                    defender_action = random_defender.choose_action(state)

                    next_state, reward, rreward, done = env.step(attacker_action, defender_action)

                    state = next_state
                    total_reward += reward

                    if done:
                        trained_attacker_rewards.append(total_reward)
                        break

            # Rastgele saldırgan vs eğitimli savunmacı
            random_attacker_rewards = []
            for _ in range(test_episodes):
                state = env.reset()
                total_reward = 0
                while True:
                    attacker_action = random_attacker.choose_action(state)
                    defender_action = defender_agent.choose_action(state)

                    next_state, reward, rreward, done = env.step(attacker_action, defender_action)

                    state = next_state
                    total_reward += rreward

                    if done:
                        random_attacker_rewards.append(total_reward)
                        break

            # Ortalama ödülleri kaydet
            trained_avg_reward = np.mean(trained_attacker_rewards)
            random_avg_reward = np.mean(random_attacker_rewards)

            intervals.append(episode)
            trained_attacker_avg_rewards.append(trained_avg_reward)
            random_attacker_avg_rewards.append(random_avg_reward)

    # Sonuçları grafikle
    plt.plot(intervals, trained_attacker_avg_rewards, label='Eğitimli Saldırganın Ortalama Ödülü')
    plt.plot(intervals, random_attacker_avg_rewards, label='Rastgele Saldırganın Ortalama Ödülü')
    plt.xlabel('Eğitim Bölümü Sayısı')
    plt.ylabel('Ortalama Ödül')
    plt.title('Ajanların Performans Artışı (Ortalama Ödül)')
    # Grafiği kaydet
    plt.savefig('manuel_code.png')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_test()
