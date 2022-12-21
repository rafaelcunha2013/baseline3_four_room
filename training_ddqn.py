import gym
import gym_sf

from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
from ddqn import DDQN


def my_test(path, env_test):
    terminated = False
    truncated = False
    next_state, _ = env_test.reset()

    best_model = DQN.load(path)

    for _ in range(500):
        action, _states = best_model.predict(next_state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env_test.step(action)

        if terminated or truncated:
            env_test.render()
            env_test.reset()
    env_test.close()


model_name = "DDQN"
max_step = 500
max_step_episode = 500
render_mode = "rgb_array"
env = gym.make("four-room-v0", render_mode=render_mode, max_episode_steps=max_step_episode)

print('Start training')
tensorboard_log = os.getcwd()
# tensorboard_log = '/data/p285087/four_room/stable_baselines3_DQN500/'
video_path = os.path.join(tensorboard_log, "video")
path = os.path.join(tensorboard_log, model_name)
check_path = os.path.join(tensorboard_log, "logs/")
checkpoint_callback = CheckpointCallback(save_freq=100*max_step, save_path=check_path, name_prefix="rl_model")
# eval_env = Monitor(gym.make("four-room-v0", render_mode="rgb_array", max_episode_steps=max_step, video=True))
eval_env = Monitor(gym.make("four-room-v0", render_mode="rgb_array", max_episode_steps=max_step))
eval_callback = EvalCallback(eval_env, best_model_save_path=check_path,
                             log_path=check_path, eval_freq=10*max_step,
                             deterministic=True, render=False, verbose=0)

# Initialize the model
# model = DQN.load(os.path.join(check_path, 'best_model'))
model = DDQN("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=0)
# env_test = gym.make("four-room-v0", render_mode="rgb_array_list", max_episode_steps=max_step, video=True, video_path=video_path)
env_test = gym.make("four-room-v0", render_mode="rgb_array_list", max_episode_steps=max_step, video=True)
i = 0
# for i in range(100):
while True:
    # model.learn(total_timesteps=1_000*max_step, tb_log_name=model_name, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)
    model.learn(total_timesteps=10_000_000, tb_log_name=model_name, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False, progress_bar=False)
    model.save(path)
    print(f"More {i * 1_000 * max_step} steps gone")

    my_test(os.path.join(check_path, 'best_model'), env_test)
    i += 1
