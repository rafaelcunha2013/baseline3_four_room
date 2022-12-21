import gym
import gym_sf

from stable_baselines3 import DQN
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
env = gym.make("four-room-v0", render_mode=render_mode, max_episode_steps=max_step)

path = os.path.join(os.getcwd(), "Peregrine", "DQN_03", "models/best_model.zip")
model = DQN.load(path)
env_test = gym.make("four-room-v0", render_mode="rgb_array_list", max_episode_steps=max_step, video=True, random_initial_position=True)

for _ in range(3):
    my_test(model, env_test)
    time.sleep(1)


