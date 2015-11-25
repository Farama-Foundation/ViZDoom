#!/usr/bin/python
from vizia import ViziaGame
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

game.addAvailableKey("MOVELEFT")
game.addAvailableKey("MOVERIGHT")
game.addAvailableKey("ATTACK")

#game.addStateAvailableVar("HEALTH");
#game.addStateAvailableVar("AMMO1");

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
	s = game.getState()
	if not game.isEpisodeFinished():
		img = np.float32(s[1])/256.0
		img = np.swapaxes(img,0,2)
		img = np.swapaxes(img,0,1)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		cv2.namedWindow('doom', cv2.WINDOW_NORMAL)
		cv2.imshow('doom',img)
		cv2.waitKey(1)
		#cv2.destroyAllWindows()

	#sleep(0.01)
end=time()

t = end-start
print "time:",round(t,3)
print "fps: ",round(iters/t,2)
game.close()

 
    
