#!/usr/bin/python

from vizia import ViziaGame
import random
from time import sleep
from time import time

game = ViziaGame()

game.setScreenResolution(320,0)

game.setDoomGamePath("zdoom")
game.setDoomIwadPath("doom2.wad")
game.setDoomFilePath("s1_b.wad")
game.setEpisodeTimeoutInDoomTics(200)

game.setRenderHud(True)
game.setRenderCrosshair(True)
game.setRenderWeapon(True)
game.setRenderDecals(False)
game.setRenderParticles(False);

game.addAvailableKey("MOVELEFT")
game.addAvailableKey("MOVERIGHT")
game.addAvailableKey("ATTACK")


game.addStateAvailableVar("HP");
game.addStateAvailableVar("AMMO1");

game.init()


actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 1000
sleep_time = 0.01
game.newEpisode()
start = time()
for i in range(iters):
	if game.isEpisodeFinished():
		game.newEpisode()
	game.makeAction(random.choice(actions))
	print game.getState()
	print game.getActionFormat()
	print game.getStateFormat()
	sleep(0.01)
end=time()

t = end-start
print "time:",round(t,3)
print "fps: ",round(iters/t,2)
game.close()

 
    
