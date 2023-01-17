import gym
import gym_sf

from stable_baselines3 import DQN
from ddqn import DDQN
import os

import time

def my_test(model, env_test):
    terminated = False
    truncated = False
    done = False
    next_state, _ = env_test.reset()

    while not done:
        action, _states = model.predict(next_state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env_test.step(action)

        if terminated or truncated:
            env_test.render()
            env_test.reset()
            done = True
    env_test.close()


max_step = 100
render_mode = "rgb_array"
env = gym.make("four-room-multiagent-v0",
               render_mode=render_mode,
               max_episode_steps=500,
               random_initial_position=False)

# path = os.path.join(os.getcwd(), "Peregrine", "DQN_03", "models/best_model.zip")
path = os.path.join(os.getcwd(), "stable_baselines3", "DDQN", "2023_01_16__19_37_45", "logs/best_model.zip")
print(path)
video_path = os.path.join(os.getcwd(), "stable_baselines3", "DDQN", "2023_01_16__19_37_45")
model = DDQN.load(path)
env_test = gym.make("four-room-multiagent-v0",
                    render_mode="rgb_array_list",
                    max_episode_steps=max_step,
                    video=True,
                    random_initial_position=False,
                    video_path=video_path)

for _ in range(3):
    my_test(model, env_test)
    time.sleep(1)


