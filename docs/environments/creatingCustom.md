# Creating a custom environment

ViZDoom allows using of custom scenarios/environments that can be easily prepared using modern Doom map editors like [SLADE](http://slade.mancubus.net/index.php?page=downloads) (available for Linux, MacOS, and Windows) or [DoomBuilder](http://www.doombuilder.com/index.php?p=downloads) (a bit better editor, but only available for Windows), that we recommend using. These editors allow designing a map used by the environment as well as programming the custom logic and rewards using ACS scripting language. In addition to a map created in one of the editors, that are saved to a .wad file, ViZDoom uses .cfg config files that store additional information about the environment. Such .wad and .cfg together define a custom environment that can be used with ViZDoom. The following will guide you through the process of creating a custom environment.


## Limitations and possibilities

Before we start explaining the process of creating custom environments, one question you might ask is what kind of environments can be created using the old Doom engine. The following list summarizes the most important limitations and possibilities for creating the environments for ViZDoom:

- **3D is limited:** ViZDoom engine does not support full 3D maps. As in the original Doom, the map is, in fact, a 2D map with additional information about floor and ceiling height. This means that some 3D structures, like bridges or multi-floor buildings, are impossible in ViZDoom. However, ViZDoom supports 3D movement like jumping, crouching, swimming, or flying, which were not possible in original Doom.
- **Map editors are easy to use:** Because of 3D limitations, the Doom-level editors (like mentioned SLADE or DoomBuilder) are actually much simpler than editors for later full 3D engines since they are based on drawing a map from a top-down view. Because of that, they are much easier to use, and everyone is able to create new maps right away.
- **Scripting is powerful:** ViZDoom environments are not limited to particular tasks, as ViZDoom supports ACS scripting language, which was created for later revisions of the Doom engine.
It has a simple C-like syntax and is very powerful. It allows to create of custom game rules and rewards. It has a large number of functions that allow to modify/extend the game logic in many ways. ZDoom ACS documentation (https://zdoom.org/wiki/ACS) is generally well-written and complete, making it easy to find the right function for the task.
Due to the engine's architecture, the only area that ACS is a bit lacking is the possibility of modifying map geometry. Simple modifications, like changing the height of some part of the level to create elevators or doors, are possible, but there are not many more options. Using those, it is possible, for example, to create a randomized maze, but something more complex might be tricky or impossible.
- **Basic functionality provided by the library:** To simplify the creation of environments, some simple functionalities are also embedded into the library. This way, they don't need to be implemented in ACS every single time but can be configured in a config file. These include:
  - possibility to define actions space
  - living rewards and death rewards
  - limited time/truncation
- **Lack of advanced physics:** ViZDoom engine is obviously based on old technology, and it's limited. It does not support advanced physics, so environments where the aim is to move objects, build structures, etc., are not possible. 
- **Support for multiplayer**: ViZDoom supports multiplayer for up to 16 players. Beyond standard multiplayer mods. ACS can be used to create custom multiplayer scenarios, which can be cooperative or adversarial.


## Createing a custom map






The following tutorial will show how to create a simple environment with a custom map and a custom scenario.



To create a custom scenario (iwad file), you need to use a dedicated editor. Doom Builder and Slade are the software tools we recommend for this task.

Scenarios (iwad files) contain maps and ACS scripts. For starters, it is a good idea to analyze the sample scenarios, which come with ViZDoom (remember that these are binary files).


KEEP THIS IN MIND

ACS and software for creating wads is quite simple and relatively user friendly but sometimes they act unexpectedly without notifying you so here are some thoughts that can potentailly help you and save hours of wondering:

    1.0 and 1 is not the same, the first one is the fixed point number stored in int and the second one is an ordinary int. Watch out what is expected by the functions you use cause using the wrong format can result in rubbish.
    Use UDMF format for maps and ZDBPS which is node (whatever that is) builder for Zdoom.

Reward

In order to use the rewarding mechanism you need to employ the global variable 0:

global int 0:reward;
...
script 1(void)
{
    ...
    reward += 100.0;
}
...

ViZDoom treats the reward as a fixed point numeral, so you need to use decimal points in ACS scripts. For example 1 is treated as an ordinary integer and 1.0 is a fixed point number. Using ordinary integer values will, most probably, result in unexpected behaviour.



