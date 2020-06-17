"""
CtF environment script and map generator

It is written to generate video rendering and script for analysis/debugging.

Original Author :
    Jacob (jacob-heglund)
Modifier/Editor :
    Seung Hyun (skim0119)
    Shamith (shamith2)
"""

import numpy as np
import csv

import os
import time
import argparse

from tqdm import tqdm

import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gym_cap
import gym_cap.heuristic as policy

import moviepy.editor as mp
from moviepy.video.fx.all import speedx

# Script Arguments
parser = argparse.ArgumentParser(description='Render CtF episodes')
parser.add_argument('--total', type=int, default=1000,
                    help='total number of episode rollout (default: 1000)')
parser.add_argument('--max-length', type=int, default=150,
                    help='maximum number of steps in each episode (default: 150)')
parser.add_argument('--num-success', type=int, default=10,
                    help='number of success (blue win) video to generate (default: 10)')
parser.add_argument('--num-failure', type=int, default=10,
                    help='number of failure (red win) video to generate (default: 10)')
parser.add_argument('--filter-min-length', type=int, default=40,
                    help='allowed minimum length of the episode (defualt: 40)')
parser.add_argument('--filter-max-length', type=int, default=120,
                    help='allowed maximum length of the episode (default: 120)')
parser.add_argument('--silence', action='store_true',
                    help='silence render')
args = parser.parse_args()

# Run Settings
total_run = args.total
max_episode_length = args.max_length
num_success = args.num_success
num_failure = args.num_failure
min_length = args.filter_min_length
max_length = args.filter_max_length

# Environment Preparation
env = gym.make("cap-v0").unwrapped 
observation = env.reset(map_size=20,
                        policy_blue=policy.Roomba(),
                        policy_red=policy.Random())
if args.silence:
    env.SILENCE_RENDER = True

# Export Settings
data_dir = 'render'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
raw_dir = 'raw_videos'
video_dir = os.path.join(data_dir, raw_dir)
if not os.path.exists(video_dir):
    os.mkdir(video_dir)

vid_success = []
vid_failure = []

def play_episode(episode=0):
    video_fn = 'episode_' + str(episode) + '.mp4'
    video_path = os.path.join(video_dir, video_fn)
    video_recorder = VideoRecorder(env, video_path)

    length = 0
    obs = env.reset()
    done = False
    while not done and length < max_episode_length:
        observation, reward, done, _ = env.step()

        video_recorder.capture_frame()

        # Optain waypoints
        waypoints = []
        for entity in env.get_team_blue.tolist() + env.get_team_red.tolist():
            waypoints.extend(entity.get_loc())
        length += 1

    # Closer
    video_recorder.close()
    vid = mp.VideoFileClip(video_path)

    # Check if episode has right length played
    if length <= min_length or length >= max_length:
        return

    # Post Processing
    if env.blue_win and len(vid_success) < num_success:
        vid_success.append(vid)
    elif env.red_win and len(vid_failure) < num_failure:
        vid_failure.append(vid)

def render_clip(frame, filename):
    """
    Render single clip with delayed frame
    """
    vid = speedx(frame, 0.1)
    video_path = os.path.join(data_dir, filename)          
    vid.write_videofile(video_path, verbose=False)
        

# Run until enough success episodes arefound
episode = 0
progbar = tqdm(total=total_run, unit='episode')
while (len(vid_success) < num_success or len(vid_failure) < num_failure) and episode < total_run:
    play_episode(episode)   
    episode += 1
    progbar.update(1)
progbar.close()

print(f"Requested {total_run} episode run, played {episode} run")
print(f"{len(vid_success)} success mode saved, and {len(vid_failure)} failure mode saved")
    
env.close()

# Export
for idx, video in enumerate(vid_success):
    render_clip(video, f'success_{idx}_video.mp4')
for idx, video in enumerate(vid_failure):
    render_clip(video, f'failure_{idx}_video.mp4')
