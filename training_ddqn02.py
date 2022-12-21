import gym
import gym_sf

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import platform
from datetime import datetime
from ddqn import DDQN


model_name = "DDQN_optuna"
unique_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
max_step = 500
max_step_episode = 500
render_mode = "rgb_array"
env = gym.make("four-room-v0", render_mode=render_mode, max_episode_steps=max_step_episode)

print('Start training')
# Check in which system the code is running. Used to select the right path
if platform.system() == 'Linux':
    root = '/data/p285087/four_room/'
else:
    root = os.getcwd()
tensorboard_log = os.path.join(root, 'stable_baselines3', model_name, unique_id)

check_path = os.path.join(tensorboard_log, "logs/")

checkpoint_callback = CheckpointCallback(save_freq=100*max_step, save_path=check_path, name_prefix="rl_model")
eval_env = Monitor(gym.make("four-room-v0", render_mode="rgb_array", max_episode_steps=max_step))
eval_callback = EvalCallback(eval_env, best_model_save_path=check_path,
                             log_path=check_path, eval_freq=10*max_step,
                             deterministic=True, render=False, verbose=0)

# Initialize the model
model = DDQN("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=0)
model.learn(total_timesteps=10_000_000,
            tb_log_name=model_name,
            callback=[checkpoint_callback, eval_callback],
            reset_num_timesteps=False,
            progress_bar=False)



