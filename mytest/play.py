# Import eval policy to test agent
from stable_baselines3.common.evaluation import evaluate_policy
from setup_env import MyDoom
from time import sleep
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

# Reload model from disc
model = PPO.load('./train/train_basic_n_step2048/best_model_100000')

# Create rendered environment
env = MyDoom(render=True)

# Evaluate mean reward for 10 games
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

for episode in range(100): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # time.sleep(0.20)
        total_reward += reward
        sleep(0.2)
    print('Total Reward for episode {} is {}'.format(total_reward, episode))
    sleep(2)
