#!/usr/bin/python

from vizia import ViziaGame
import random
from time import sleep
from time import time

game = ViziaGame()

game.setScreenResolution(320,240)
game.setDoomGamePath("zdoom")
game.setDoomIwadPath("doom2.wad")
game.setDoomFilePath("s1_b.wad")
game.setEpisodeTimeoutInDoomTics(200)

game.setRenderHud(False)
game.setRenderCrosshair(True)
game.setRenderWeapon(True)
game.setRenderDecals(False)
game.setRenderParticles(False);

game.addAvailableAction("MOVELEFT")
game.addAvailableAction("MOVERIGHT")
game.addAvailableAction("ATTACK")

game.addStateAvailableVar("HP");
game.addStateAvailableVar("AMMO1");

game.init()


actions = [[True,False,False],[False,True,False],[False,False,True]]

iters = 10000
sleep_time = 0.01
game.newEpisode()
start = time()
for i in range(iters):
	if game.isEpisodeFinished():
		game.newEpisode()
	game.makeAction(random.choice(actions))
	#sleep(sleep_time)
end=time()

t = end-start
print "time:",round(t,3)
print "fps: ",round(iters/t,2)
game.close()

 
    
