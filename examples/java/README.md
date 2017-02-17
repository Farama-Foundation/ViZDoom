Use CMake to generate Makefile or project or build it using:
```bash
javac -classpath "../../bin/java/vizdoom.jar" Example.java
jar cf ../../bin/examples/example.jar Example.class
```

Examples will be build in ``vizdoom_root_dir/bin/examples``.
To run example use:
```bash
java -Djava.library.path="../java" -classpath "../java/vizdoom.jar:example.jar" Example
```


To run the examples, [freedoom2.wad]( https://freedoom.github.io/download.html) should be placed in the  ``scenarios`` subdirectory.

---
##Examples

###[Basic.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Basic.java)
Demonstrates how to use the most basic features of the environment. It configures the engine, and makes the agent perform random actions. It also prints the current state and the reward earned with every action.

###[CIG.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/CIG.java), [CIGHost.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/CIGHost.java)
Demonstrates how to configure and play multiplayer game for CIG 2016 competition.

###[CIGBots.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/CIGBots.java)
Demonstrates how to play with bots to simulate multiplayer game. Helpful for developing AI agent for CIG 2016 competition.

###[DeltaButtons.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/DeltaButtons.java)
Shows how delta buttons work (they may take values other than 0 and 1 and can be used for precise movement).

###[Multiplayer.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Multiplayer.java), [MultiplayerHost.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/MultiplayerHost.java)
Demonstrates how to configure and play multiplayer game.

###[Seed.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Seed.java)
Shows how to run deterministic episodes by setting the seed. After setting the seed every episode will look the same (if the agent behaves deterministically).

###[Shaping.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Shaping.java)
Demonstrates how to make use of the game variables to implement [shaping](https://en.wikipedia.org/wiki/Shaping_(psychology)) using health_guided.wad scenario.

###[Spectator.java](https://github.com/Marqt/ViZDoom/blob/master/examples/java/Spectator.java)
Shows how to use the *SPECTATOR* mode in which YOU play Doom and AI is the spectator (intended for apprenticeship learning).
