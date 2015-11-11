#!/usr/bin/python

import api
import numpy as np

x = 5
y = 3
random_background = True
api.init(x,y,random_background,50,-0.02,0.05,1,np.inf)
print api.is_finished()


api.new_episode()

actions = [[True,False,False],[False,True,False],[False,False,True]]
api.make_action(actions[1]);
