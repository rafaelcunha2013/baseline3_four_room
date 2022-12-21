import gym
import gym_sf

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import copy
import pickle
import sys


render_mode = "rgb_array"
env = gym.make("four-room-v0", render_mode=render_mode, max_episode_steps=500)

print('Start training')
tensorboard_log = '/data/p285087/four_room/stable_baselines3/'
path = tensorboard_log + 'PPO'
check_path = tensorboard_log + "logs/"
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=check_path, name_prefix="rl_model")
eval_env = Monitor(copy.deepcopy(env))
eval_callback = EvalCallback(eval_env, best_model_save_path=check_path,
                             log_path=check_path, eval_freq=100*1000,
                             deterministic=True, render=False)

model = PPO("MlpPolicy", env, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=200*1000, tb_log_name='PPO1', callback=[checkpoint_callback, eval_callback])
model.save(path)
print('First 200 gone')


env_test = gym.make("four-room-v0", render_mode="rgb_array_list", max_episode_steps=5000, video=True)

