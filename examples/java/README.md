To run the examples, [freedoom2.wad]( https://freedoom.github.io/download.html) should be placed in the  ``../scenarios`` subdirectory.
This step should be done automatically by the building process.

Windows: All needed classes can be found in ``../../bin/java/vizjavaclasses.jar``
Linux: modify and run ./run_JExample.sh
---
##Examples

###[Basic.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Basic.java)
Demonstrates how to use the most basic features of the environment. It configures the engine, and makes the agent perform random actions. It also prints the current state and the reward earned with every action.

###[Spectator.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Spectator.java)
Shows how to use the *SPECTATOR* mode in which YOU play Doom and AI is the spectator (intended for apprenticeship learning).

###[Shaping.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Shaping.java)
Demonstrates how to make use of the game variables to implement [shaping](https://en.wikipedia.org/wiki/Shaping_(psychology)) using health_guided.wad scenario.

###[Seed.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Seed.java)
Shows how to run deterministic episodes by setting the seed. After setting the seed every episode will look the same (if the agent behaves deterministically).
