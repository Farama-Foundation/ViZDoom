import os
from random import choice
from time import sleep
from typing import Any, Dict, Tuple
import vizdoom as vzd
import numpy as np

import cv2
import gym
from gym import Env
from gym.spaces import Discrete, Box


WIDTH = 160
HEIGHT = 100
TICRATE = 350
class MyDoom(Env):
    def __init__(self, render=False):
        self.game = vzd.DoomGame()
        self.game.load_config('../scenarios/deadly_corridor.cfg')

        if render == False:           
            self.game.set_window_visible(False)
        else:            
            #self.game.set_mode(vzd.Mode.PLAYER)
            #self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
            #self.game.set_ticrate(TICRATE)
            self.game.set_window_visible(True)
        
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8) 
        self.action_space = Discrete(7)
    def step(self, action):
        actions = np.identity(7, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)
        done =  self.game.is_episode_finished()

        if self.game.get_state():
            _stat = self.game.get_state()
            state = _stat.screen_buffer
            state = self.resize(state)
            #info = { "ammo": _stat.game_variables[0], "health": _stat.game_variables[1] }
            info = { "health": _stat.game_variables[0] }
        else:
            state = np.zeros(self.observation_space.shape)
            #info = { "ammo": 0, "health": 0 }
            info = { "health": 0 }
        return state, reward, done, info
    def resize(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (HEIGHT, WIDTH, 1))
        return state
    def reset(self):
        self.game.new_episode()
        _stat = self.game.get_state()
        return self.resize(_stat.screen_buffer)
    def close(self) -> None:
        return self.game.close()
