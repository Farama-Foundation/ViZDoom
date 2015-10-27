#!/usr/bin/python

import numpy as np
from games import ShootingDotGame
import random

width = 3

game = ShootingDotGame(width = width,height = 1 , max_moves = 50, miss_penalty = 0, living_reward = -1, hit_reward = 150, random_background = False)

print game.compute_qvalues()