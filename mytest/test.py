import os
from random import choice
from time import sleep
import vizdoom as vzd
import numpy as np

import cv2
import gym
from gym import Env
from gym.spaces import Discrete, Box

if __name__ == "__main__":
    """
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()
    game.load_config('../scenarios/basic.cfg')
    game.init()
    actions = np.identity(3, dtype=np.uint8)
    for ep in range(10):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            info = state.game_variables
            reward = game.make_action(choice(actions))
            print('reward:', reward, img.shape) 
            sleep(0.02)
        print('Result:', game.get_total_reward())
        sleep(2)
    game.close()
    """
    print(Discrete(3).sample())
    print(Box(low=0, high=10, shape=(10, 10), dtype=np.uint8).sample())
    #print(Box(low=0, high=255, shape=(320, 240), dtype=np.uint8).sample())