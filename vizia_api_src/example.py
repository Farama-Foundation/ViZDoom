#!/usr/bin/python

from vizia import ViziaGame
import random
from time import sleep

game = ViziaGame()

game.setScreenResolution(320,240)
game.setDoomGamePath("zdoom")
game.setDoomIwadPath("doom2.wad")
game.setDoomFilePath("s1_b.wad")
game.setEpisodeTimeoutInDoomTics(150)

game.setRenderHud(True)
game.setRenderCrosshair(True)
game.setRenderWeapon(True)
game.setRenderDecals(False)

game.addAvailableAction("MOVELEFT")
game.addAvailableAction("MOVERIGHT")
game.addAvailableAction("ATTACK")

game.addStateAvailableVar("HP");
game.addStateAvailableVar("AMMO_ROCKET");

game.init()


actions = [[True,False,False],[False,True,False],[False,False,True]]

iters = 1000
sleep_time = 0.01
for i in range(iters):
	if i%150==0:
		game.newEpisode()
	game.makeAction(random.choice(actions))
	sleep(sleep_time)

game.close()

 
    
