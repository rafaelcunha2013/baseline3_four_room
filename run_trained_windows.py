import gym
import gym_sf

from stable_baselines3 import DQN
import os
import time


def run_episode(episode_model, env_test):
    terminated = False
    truncated = False
    next_state, _ = env_test.reset()

    while True:
        action, _states = episode_model.predict(next_state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env_test.step(action)

        if terminated or truncated:
            env_test.render()
            env_test.reset()
            break
    env_test.close()


# Initialize the model
model_name = "rl_model_49150000_steps"
max_step = 500
path = os.path.join(os.getcwd(), model_name)
model = DQN.load(path)

env = gym.make("four-room-v0", render_mode="rgb_array_list", max_episode_steps=max_step, video=True)
for _ in range(5):
    run_episode(model, env)
    time.sleep(1)

