#!/usr/bin/python

from vizia import DoomGame
from time import sleep
from vizia import Button
game = DoomGame()
game.load_config("sample_config.dcfg")
game.init()
sleep(1)

#game.close()
#exit(0)