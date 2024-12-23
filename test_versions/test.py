import time
import gym
import gym_cap
import gym_cap.heuristic as policy
import numpy as np
import torch
import configparser

from stable_baselines3 import PPO

def change_ini_settings(file_path, section, key, new_value):
    config = configparser.ConfigParser()
    config.read(file_path)
    original_value = config.get(section, key)
    config.set(section, key, new_value)

    with open(file_path, 'w') as config_file:
        config.write(config_file)
    return original_value




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("cap-v0")


output_file = open("results_1000_new.txt", "a")  # Metin dosyasını yazmak için dosya açılır
num_match = 1000
setting_file = 'setting.ini'

red_list=['Roomba'] #['Defense','Random','Zeros']
blue_list=['Defense','Patrol','Spiral','Random','Zeros']
values_to_change = {
    'elements': {
        'NUM_RED': '6',
        'NUM_RED_UAV': '3'
    },
    'settings': {
        'STOCH_TRANSITIONS': 'False',
        'STOCH_ATTACK': 'True'
    },
    'control': {
        'RED_ADV_BIAS': '2'
    },
    'memory': {
        'TEAM_MEMORY': 'fog',
        'INDIV_MEMORY': 'fog'
    },
    'communication': {
        'COM_GROUND': 'True',
        'COM_DISTANCE': '2'
    }
}

for section, key_value_dict in values_to_change.items():
    for key, new_value in key_value_dict.items():
        original_value = change_ini_settings(setting_file, section, key, new_value)

        output_file.write("time: " + time.ctime(time.time()) + " ( "+ key + " "+ original_value +" => "+ new_value +" )  \n")
        for bl in blue_list:
            for rl in red_list:
                observation = env.reset(
                                    map_size=20,
                                    config_path=setting_file,
                                    policy_red=getattr(policy, rl)(),
                                    policy_blue=getattr(policy, bl)() #policy.Fighter() # Defense Random Fighter Zeros Roomba Patrol Spiral Policy blue_policy
                                )
                action = env.action_space.sample()
                rscore = []
                start_time = time.time()

                for n in range(num_match):
                    done = False
                    rewards = []
                    while not done:
                        observation, reward, done, info = env.step(action)
                        rewards.append(reward)
                    env.reset()
                    rscore.append(sum(rewards))
                    duration = time.time() - start_time
                    #print("Time: %.2f s, Score: %.2f" % (duration, rscore[-1]))

                average_time = duration / num_match
                average_score = sum(rscore) / num_match
                print("red: "+rl+" blue: "+ bl+ " Average Time: %.2f s, Average Score: %.2f" % (duration/num_match, sum(rscore)/num_match))
                output_file.write("red: " + rl + " blue: " + bl + " Average Time: %.2f s, Average Score: %.2f\n" % (average_time, average_score))
        output_file.write("bitiş time: " + time.ctime(time.time()) + "\n")
        change_ini_settings(setting_file, section, key, original_value)


output_file.close()
env.close()