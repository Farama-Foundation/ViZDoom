To run the examples, [freedoom2.wad]( https://freedoom.github.io/download.html) should be placed in the  ``scenarios`` subdirectory.
Furthermore, you need to have ``vizdoom.so/vizdoom.pyd`` present (or symlinked) in the ``examples/python`` directory (on Linux it should be done automatically by the building process).

##Troubleshooting
 * `SystemError: dynamic module not initialized properly` may mean that you are trying the run an example using python3 instead of python2.

---
##Examples

###[basic.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/basic.py)
Demonstrates how to use the most basic features of the environment. It configures the engine, and makes the agent perform random actions. It also prints the current state and the reward earned with every action.

###[cig.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/cig.py), [cig_host.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/cig_host.py)
Demonstrates how to configure and play multiplayer game for CIG 2016 competition.

###[cig_bots.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/cig_bots.py)
Demonstrates how to play with bots to simulate multiplayer game. Helpful for developing AI agent for CIG 2016 competition.

###[delta_buttons.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/delta_buttons.py)
Shows how delta buttons work (they may take values other than 0 and 1 and can be used for precise movement).

###[format.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/format.py)
Presents different formats of the screen buffer. [OpenCV](http://opencv.org/) is used to display the images.

###[fps.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/fps.py)
Tests the performance of the environment in frames per second. It should give you some idea how fast the framework works on your hardware.

###[shaping.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/shaping.py)
Demonstrates how to make use of the game variables to implement [shaping](https://en.wikipedia.org/wiki/Shaping_(psychology)) using health_guided.wad scenario.

###[scenarios.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/scenarios.py)
Presents different scenarios that come with ViZDoom environment.

###[seed.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/seed.py)
Shows how to run deterministic episodes by setting the seed. After setting the seed every episode will look the same (if the agent behaves deterministically).

###[spectator.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/spectator.py)
Shows how to use the *SPECTATOR* mode in which YOU play Doom and AI is the spectator (intended for apprenticeship learning).
