#!/bin/env python
import sys

import gym
import gym_sf

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

name = sys.argv[1]
max_step_episode = sys.argv[2]
exploration_fraction = 0.1
train_freq = 4

max_step = 500

render_mode = "rgb_array"
env = gym.make("four-room-v0", render_mode=render_mode, max_episode_steps=max_step_episode)

print('Start training')
tensorboard_log = '/data/p285087/four_room/stable_baselines3_DQN500_' + name + '/'
check_path = os.path.join(tensorboard_log, "logs/")
checkpoint_callback = CheckpointCallback(save_freq=100*max_step, save_path=check_path, name_prefix="rl_model")
eval_env = Monitor(gym.make("four-room-v0", render_mode="rgb_array", max_episode_steps=max_step))
eval_callback = EvalCallback(eval_env, best_model_save_path=check_path,
                             log_path=check_path, eval_freq=10*max_step,
                             deterministic=True, render=False)

# Initialize the model
model = DQN("MlpPolicy", env, tensorboard_log=tensorboard_log, exploration_fraction=exploration_fraction, train_freq=train_freq)
model.learn(total_timesteps=10_000_000, tb_log_name="DQN", callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)

