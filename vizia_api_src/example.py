#!/usr/bin/python

from api import ViziaMain
import random
from time import sleep

game = ViziaMain()

game.setScreenResolution(320,240)
game.setDoomGamePath("zdoom")
game.setDoomIwadPath("doom2.wad")
game.setDoomFilePath("s1_b.wad")
game.setEpisodeTimeoutInDoomTics(250)

game.setRenderHud(True)
game.setRenderCrosshair(True)
game.setRenderWeapon(True)
game.setRenderDecals(False)

game.addAvailableAction("MOVELEFT")
game.addAvailableAction("MOVERIGHT")
game.addAvailableAction("ATTACK")

game.init()


actions = [[True,False,False],[False,True,False],[False,False,True]]

iters = 1000
sleep_time = 0.01
for i in range(iters):
	game.makeAction(actions[0])
	sleep(sleep_time)

game.close()

 
    
