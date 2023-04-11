# C++ Examples

## Building
To build the examples first you need to build the ViZDoom library, then you can run the following commands in this directory (`examples/c++`):

```
cmake . && make
```

Examples will be placed in `<vizdoom_dir>/bin/examples`.

---
## The list of examples

### [Basic.cpp](https://github.com/Marqt/ViZDoom/blob/master/examples/c++/Basic.cpp)
Demonstrates how to use the most basic features of the environment. It configures the engine, and makes the agent perform random actions. It also prints the current state and the reward earned with every action.

### [CIG.cpp](https://github.com/Marqt/ViZDoom/blob/master/examples/c++/CIG.cpp), [CIGHost.cpp](https://github.com/Marqt/ViZDoom/blob/master/examples/c++/CIGHost.cpp)
Demonstrates how to configure and play multiplayer game for CIG competition.

### [CIGBots.cpp](https://github.com/Marqt/ViZDoom/blob/master/examples/c++/CIGBots.cpp)
Demonstrates how to play with bots to simulate multiplayer game. Helpful for developing AI agent for CIG competition.

### [DeltaButtons.cpp](https://github.com/Marqt/ViZDoom/blob/master/examples/c++/DeltaButtons.cpp)
Shows how delta buttons work (they may take values other than 0 and 1 and can be used for precise movement).

### [FPSTest.cpp](https://github.com/mwydmuch/ViZDoom/blob/master/examples/c++/FPSTest.cpp)
Tests the performance of the environment in frames per second. It should give you some idea how fast ViZDoom works on your hardware.

### [Seed.cpp](https://github.com/Marqt/ViZDoom/blob/master/examples/c++/Seed.cpp)
Shows how to run deterministic episodes by setting the seed. After setting the seed every episode will look the same (if the agent behaves deterministically).

### [Shaping.cpp](https://github.com/Marqt/ViZDoom/blob/master/examples/c++/Shaping.cpp)
Demonstrates how to make use of the game variables to implement [shaping](https://en.wikipedia.org/wiki/Shaping_(psychology)) using health_guided.wad scenario.

### [Spectator.cpp](https://github.com/Marqt/ViZDoom/blob/master/examples/c++/Spectator.cpp)
Shows how to use the *SPECTATOR* mode in which YOU play Doom and AI is the spectator (intended for apprenticeship learning).
