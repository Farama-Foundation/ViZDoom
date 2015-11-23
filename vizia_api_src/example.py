#!/usr/bin/python

from api import ViziaMain


game = ViziaMain()

game.setScreenResolution(320,240)
game.setDoomGamePath("zdoom")
game.setDoomIwadPath("doom2.wad")
game.setDoomFilePath("s1_b.wad")
game.setEpisodeTimeoutInDoomTics(200)

game.setRenderHud(False)
game.setRenderCrosshair(True)
game.setRenderWeapon(True)
game.setRenderDecals(False)

game.addAvailableAction("MOVELEFT")
game.addAvailableAction("MOVERIGHT")
game.addAvailableAction("ATTACK")
game.init()



 
    
