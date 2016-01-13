#!/usr/bin/python

from vizia import DoomGame

game = DoomGame()
game.load_config("sample_config.dcfg")
game.init()


game.close()