#!/bin/env python
import sys
import os
import platform
from datetime import datetime

import gym
import gym_sf

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from ddqn import DDQN


# model_name = "DDQN"
# max_step_episode = 500
# random_initial_position = False

max_step = 500
render_mode = "rgb_array"


model_name = sys.argv[1]
max_step_episode = int(sys.argv[2])
exploration_fraction = float(sys.argv[3])
train_freq = int(sys.argv[4])
random_initial_position = bool(sys.argv[5])
target_update_interval = int(sys.argv[6])


unique_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
env = gym.make("four-room-v0",
               render_mode=render_mode,
               max_episode_steps=max_step_episode,
               random_initial_position=random_initial_position)

print('Start training')
# Check in which system the code is running. Used to select the right path
if platform.system() == 'Linux':
    root = '/data/p285087/four_room/'
else:
    root = os.getcwd()
tensorboard_log = os.path.join(root, 'stable_baselines3', model_name, unique_id)

check_path = os.path.join(tensorboard_log, "logs/")

checkpoint_callback = CheckpointCallback(save_freq=100*max_step, save_path=check_path, name_prefix="rl_model")
eval_env = Monitor(env)
eval_callback = EvalCallback(eval_env, best_model_save_path=check_path,
                             log_path=check_path, eval_freq=10*max_step,
                             deterministic=True, render=False, verbose=0)

# Initialize the model
model = DDQN("MlpPolicy",
             env,
             tensorboard_log=tensorboard_log,
             exploration_fraction=exploration_fraction,
             train_freq=train_freq,
             target_update_interval=target_update_interval,
             verbose=0)

model.learn(total_timesteps=20_000_000,
            tb_log_name=model_name,
            callback=[checkpoint_callback, eval_callback],
            reset_num_timesteps=False,
            progress_bar=False)



