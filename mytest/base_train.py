from setup_env import MyDoom
from time import sleep
import os
from stable_baselines3 import PPO
from utils import TrainAndLoggingCallback

#total_steps = 1e5
total_steps = 4e5


_env = MyDoom(render=False)
#state = _env.reset()

CHECKPOINT_DIR = './train/train_deadly_corridor_seed2'
LOG_DIR = './logs/log_deadly_corridor_seed2'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

#model = PPO('CnnPolicy', _env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
model = PPO('CnnPolicy', _env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)

model.learn(total_timesteps=total_steps, callback=callback)

