# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53/files reppo
from setup_env import MyDoom
from time import sleep
import os

from PPO import PPO
#from stable_baselines3 import PPO
from tools import TrainAndLoggingCallback

from PPO.common.monitor import Monitor
from PPO.common.vec_env import DummyVecEnv, VecFrameStack
#from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

#total_steps = 1e5
total_steps = 4e5


_stage = "deadly_corridor"
_test = "frameStack"
CHECKPOINT_DIR = f'./train/train_{_stage}_{_test}'
LOG_DIR = f'./logs/log_{_stage}_{_test}'

_env = MyDoom(render=False)
_env = Monitor(_env, LOG_DIR)
_env = DummyVecEnv([lambda: _env])
_env = VecFrameStack(_env, 4, channels_order='last')

#state = _env.reset()

callback = TrainAndLoggingCallback(check_freq=20*10000, save_path=CHECKPOINT_DIR)

#model = PPO('CnnPolicy', _env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
Debug = True
if Debug:
    model = PPO('CnnLSTMPolicy', _env, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
    model.learn(total_timesteps=1e4)
else: 
    model = PPO('CnnPolicy', _env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
    model.learn(total_timesteps=total_steps, callback=callback)

