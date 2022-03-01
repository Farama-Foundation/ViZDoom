from setup_env import MyDoom
from time import sleep
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO


total_steps = 1e5

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

_env = MyDoom(render=False)
#state = _env.reset()

CHECKPOINT_DIR = './train/train_defend_nstep4096'
LOG_DIR = './logs/log_defend_nstep4096'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

model = PPO('CnnPolicy', _env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)

model.learn(total_timesteps=total_steps, callback=callback)

