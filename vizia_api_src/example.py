#!/usr/bin/python
from vizia import ViziaGame
from vizia import Button
from vizia import GameVar
import random
from time import sleep
from time import time
import cv2
import numpy as np
game = ViziaGame()

#available resolutions: 40x30, 60x45, 80x60, 100x75, 120x90, 160x120, 200x150, 320x240, 640x480
game.setScreenResolution(320,0)

game.setDoomGamePath("zdoom")
game.setDoomIwadPath("doom2.wad")
game.setDoomFilePath("s1_b.wad")
game.setEpisodeTimeoutInDoomTics(200)

game.setRenderHud(False)
game.setRenderCrosshair(False)
game.setRenderWeapon(True)
game.setRenderDecals(False)
game.setRenderParticles(False);

game.addAvailableButton(Button.MOVELEFT)#moveleft
game.addAvailableButton(Button.MOVERIGHT)#moveright
game.addAvailableButton(Button.ATTACK)#attack

game.addStateAvailableVar(GameVar.AMMO1);

##game.init()
exit(0)

actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 10000
sleep_time = 0.01
game.newEpisode()
start = time()
for i in range(iters):
	if game.isEpisodeFinished():
		game.newEpisode()
	r = game.makeAction(random.choice(actions))
	s = game.getState()
	#if r!=0:
		#print "reward:",r
	#sleep(0.01)
end=time()

t = end-start
print "time:",round(t,3)
print "fps: ",round(iters/t,2)
game.close()

 
    
