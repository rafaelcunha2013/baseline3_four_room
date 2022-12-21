import gym
import gym_sf

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os


def my_test(path, env_test):
    terminated = False
    truncated = False
    next_state, _ = env_test.reset()

    model = PPO.load(path)

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
# tensorboard_log = '/data/p285087/four_room/stable_baselines3/'
tensorboard_log = os.getcwd()
path = os.path.join(tensorboard_log, 'PPO')
check_path = os.path.join(tensorboard_log, "logs/")
checkpoint_callback = CheckpointCallback(save_freq=100*max_step, save_path=check_path, name_prefix="rl_model")
eval_env = Monitor(gym.make("four-room-v0", render_mode="rgb_array_list", max_episode_steps=max_step, video=True))
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=check_path,
                             log_path=check_path,
                             eval_freq=10*max_step,
                             deterministic=True,
                             render=False)

model = PPO("MlpPolicy", env, tensorboard_log=tensorboard_log)
env_test = gym.make("four-room-v0", render_mode="rgb_array_list", max_episode_steps=max_step, video=True)

for i in range(10):
    model.learn(total_timesteps=1_000*max_step,
                tb_log_name='PPO',
                callback=[checkpoint_callback, eval_callback],
                reset_num_timesteps=False)
    model.save(path)
    print(f"More {i * 1_000 * max_step} steps gone")

    my_test(os.path.join(check_path, 'best_model'), env_test)

