from setup_env import MyDoom
from time import sleep
import os
from stable_baselines3 import PPO
from utils import TrainAndLoggingCallback


total_steps = 4e4

#model = PPO.load('./train/train_deadly_corridor_seed2/best_model_260000')
model = PPO.load('./train/train_deadly_corridor5/best_model_40000')

for i in range(4):
    _i = i+2
    CHECKPOINT_DIR = f'./train/train_deadly_corridor{_i}'
    #LOG_DIR = f'./logs/log_deadly_corridor-{_i}'
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    env = MyDoom(config=f'../scenarios/deadly_corridor_s{_i}.cfg')
    model.set_env(env)
    model.learn(total_timesteps=total_steps, callback=callback)
