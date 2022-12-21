import numpy as np
import gym
import gym_sf
from stable_baselines3 import DQN

render_mode = "rgb_array_list" # "rgb_array" "rgb_array_list"
env = gym.make("four-room-v0", render_mode=render_mode, max_episode_steps=5000, video=True)
# env = gym.make("four-room-v0", render_mode='human', new_step_api=True, max_episode_steps=5000)
terminated = False
truncated = False
next_state, _ = env.reset()

model = DQN.load("four_room02")

for _ in range(5000):
    action, _states = model.predict(next_state, deterministic=True)
    next_state, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        env.render()
        env.reset()
env.close()