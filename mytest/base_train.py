from setup_env import MyDoom
from time import sleep
import os

from PPO import PPO
from utils import TrainAndLoggingCallback

#total_steps = 1e5
total_steps = 4e5


_env = MyDoom(render=False)
#state = _env.reset()

CHECKPOINT_DIR = './train/train_deadly_corridor_origin'
LOG_DIR = './logs/log_deadly_corridor_origin'

callback = TrainAndLoggingCallback(check_freq=10*10000, save_path=CHECKPOINT_DIR)

#model = PPO('CnnPolicy', _env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
Debug = True
if Debug:
    model = PPO('CnnPolicy', _env, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
    model.learn(total_timesteps=1e4)
else: 
    model = PPO('CnnPolicy', _env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
    model.learn(total_timesteps=total_steps, callback=callback)

