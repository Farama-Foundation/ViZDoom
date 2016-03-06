To run the examples [freedoom2.wad]( https://freedoom.github.io/download.html) file is needed and should be placed in ../scenarios subdirectory.
Furthermore you need to have **vizdoom.so** library and **vizdoom** present (or symlinked) in this directory.

```bash
ln -s ../../bin/vizdoom .
ln -s ../../bin/python/vizdoom.so .
```
---
##Examples

###[basic.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/basic.py)
This script presents how to use the most basic features of the environment. It configures the engine, and makes the agent perform random actions. It also prints current state and reward earned with every action.

###[scenarios.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/scenarios.py)
This script shows different scenarios that come with ViZDoom environment

###[spectator.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/spectator.py)
This example shows how to use *SPECTATOR* mode in which YOU play and AI is the spectator (intended for apprenticeship learning).

###[format.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/format.py)
This script presents different formats of the screen buffer. [OpenCV](http://opencv.org/) is used here to display images, install it or remove any references to cv2.

###[fps.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/fps.py)
This script tests performance of the environment in frames per second. It should give you some idea how fast the framework works on your hardware.

###[shaping.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/shaping.py)
This script presents how to make use of game variables to implement shaping using health_guided.wad scenario.

###[seed.py](https://github.com/Marqt/ViZDoom/blob/master/examples/python/seed.py)
This script presents how to run deterministic episodes by setting the seed. After setting the seed every episode will look the same (if agent will behave deterministically).
