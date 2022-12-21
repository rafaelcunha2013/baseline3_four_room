import gym
import gym_sf

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os


def my_test(path, env_test):
    terminated = False
    truncated = False
    next_state, _ = env_test.reset()

    model = DQN.load(path)

    for _ in range(500):
        action, _states = model.predict(next_state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env_test.step(action)

        if terminated or truncated:
            env_test.render()
            env_test.reset()
    env_test.close()


max_step = 500
render_mode = "rgb_array"
env = gym.make("four-room-v0", render_mode=render_mode, max_episode_steps=max_step)

print('Start training')
tensorboard_log = '/data/p285087/four_room/stable_baselines3/'

path = os.path.join(tensorboard_log, 'PPO')
check_path = os.path.join(tensorboard_log, "logs/")

model = DQN.load(os.path.join(check_path, 'best_model'))
env_test = gym.make("four-room-v0", render_mode="rgb_array_list", max_episode_steps=max_step, video=True)

my_test(os.path.join(check_path, 'best_model'), env_test)


