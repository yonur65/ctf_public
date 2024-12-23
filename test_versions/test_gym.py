import gym
import gym_cap
import time

# cap-v0 ortamını oluştur
env = gym.make("cap-v0")

# Ortamı sıfırla ve başlangıç gözlemini al
obs = env.reset()

done = False

while not done:
    # Ekrana (terminale) mevcut durumu çizdir
    env.render()
    
    # Rastgele bir aksiyon seç
    action = env.action_space.sample()
    
    # Seçilen aksiyonu ortama uygula
    obs, reward, done, info = env.step(action)
    
    # Aksiyon ve ödül bilgilerini ekrana yazdır
    print(f"Aksiyon: {action}, Ödül: {reward}")
    
    # Adımlar arası kısa bir bekleme (görsel takibi kolaylaştırır)
    time.sleep(0.5)

# Ortamı kapat
env.close()
