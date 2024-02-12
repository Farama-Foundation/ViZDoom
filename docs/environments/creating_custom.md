# Creating a custom environment

ViZDoom allows the use of custom scenarios/environments that can be easily prepared using modern Doom map editors like [SLADE](http://slade.mancubus.net/index.php?page=downloads) (available for Linux, MacOS, and Windows) or [DoomBuilder](http://www.doombuilder.com/index.php?p=downloads) (a bit better editor, but only available for Windows), that we recommend using. These editors allow you to create a map used by the environment and program custom logic and rewards using ACS scripting language. In addition to a map+script created in one of the editors that is saved to a .wad file, ViZDoom uses .cfg config files that store additional information about the environment. Such .wad and .cfg together define a custom environment that can be used with ViZDoom. The following will guide you through the process of creating a custom environment.


## Limitations and possibilities

Before we start explaining the process of creating custom environments, one question you might ask is what kind of environments can be created using the old Doom engine. The following list summarizes the most important limitations and possibilities for creating the environments for ViZDoom:

- **3D is limited:** ViZDoom engine does not support full 3D maps. As in the original Doom, the map is, in fact, a 2D map with additional information about floor and ceiling height. This means that some 3D structures, like bridges or multi-floor buildings, are impossible in ViZDoom. However, ViZDoom supports 3D movement like jumping, crouching, swimming, or flying, which were not possible in original Doom.
- **Map editors are easy to use:** Because of 3D limitations, the Doom-level editors (like mentioned SLADE or DoomBuilder) are actually much simpler than editors for later full 3D engines since they are based on drawing a map from a top-down view. Because of that, they are much easier to use, and everyone is able to create new maps right away.
- **Scripting is powerful:** ViZDoom environments are not limited to particular tasks, as ViZDoom supports ACS scripting language, which was created for later revisions of the Doom engine.
It has a simple C-like syntax and is very powerful. It allows you to create custom game rules and rewards. It has a large number of functions that allow the modification/extension of the game logic in many ways. ZDoom ACS documentation (https://zdoom.org/wiki/ACS) is generally well-written and complete, making it easy to find the right function for the task.
Due to the engine's architecture, the only area that ACS is a bit lacking is the possibility of modifying map geometry. Simple modifications are possible (like changing the height of some part of the level to create elevators or doors), but there are not many more options. Using those, it is possible, for example, to create a randomized maze, but something more complex might be tricky or impossible.
- **Basic functionality provided by the library:** To simplify the creation of environments, some simple functionalities are also embedded into the library. This way, they don't need to be implemented in ACS every single time but can be configured in a config file. These include:
  - possibility to define actions space
  - possibility to define what is included in the observation (types of buffers, additional variables, etc.)
  - living rewards and death rewards
  - limited time/truncation
- **Lack of advanced physics:** ViZDoom engine is obviously based on old technology, and it's limited. It does not support advanced physics, so environments where the aim is to move objects, build structures, etc., are not possible.
- **Support for multiplayer**: ViZDoom supports multiplayer for up to 16 players. Beyond standard multiplayer mods. ACS can be used to create custom multiplayer scenarios, which can be cooperative or adversarial.


## Step 1: Creating a custom map

To create a custom scenario (.wad file), you need to use a dedicated editor. [SLADE](http://slade.mancubus.net/index.php?page=downloads) (available for Linux, MacOS, and Windows) or [DoomBuilder](http://www.doombuilder.com/index.php?p=downloads) (a bit better editor, but only available for Windows), are software that we recommend using for this task.

When creating a new map, select UDMF format for maps. If asked for a node builder, you can select none, as ViZDoom has it built in. You should not have any problems with creating a map using the editor, it is simple, and you can find a lot of tutorials on the internet.

You can add some custom ACS scripts to your map. This ACS script allows the implementation of a rewarding mechanism.
To do that, you need to employ the global variable 0 like this:

```{code-block} C
global int 0:reward;
...
script 1(void)
{
    ...
    reward += 100.0;
}
...
```

The global variable 0 will be used by ViZDoom to get the reward value.

Please note that in ACS, `1.0` and `1` are not the same. The first one is the fixed point number stored in int, and the second one is an ordinary int. Please be aware of that difference. ViZDoom treats the reward as a fixed point numeral, so you always need to use decimal points in ACS scripts.
Unfortunately, ACS does not support real floating point numbers.


## Step 2: Creating a custom config file

After creating a map, it is a good idea to create an accompanying config file, that allows to easily define action space, available information in a state/observation, additional rewards, etc. The config file is a simple text file in an *.ini-like format that can be created using any text editor. The config files are documented under [api/configurationFiles.md](api/configurationFiles.md).

The following is an example of a config file that can be used with the map created in the previous step:

```{code-block} ini
doom_scenario_path = mywad.wad
doom_map = map01        # map in the wad file that will be used (wad can contain more than one map)

living_reward = -1      # add -1 reward for each tic (action)
episode_start_time = 14 # make episodes start after 14 tics (after unholstering the gun)
episode_timeout = 300   # make episodes finish after 300 actions (tics)

available_buttons = {   # limit action space to only three buttons
    MOVE_LEFT
    MOVE_RIGHT
    ATTACK
}

available_game_variables = { # make information about ammo available in the state
    AMMO2
    AMMO3
}

depth_buffer = true     # add depth buffer to the state
```


## Step 3: Loading/using a custom environment/scenario

The easiest way to use a custom scenario in the original ViZDoom API is to load the config file using a dedicated method:

```{code-block} python
game = vzd.DoomGame()
game.load_config("<path to .cfg file>")
```

It can also be registered as a Gymnasium environment using the following method:

```{code-block} python
from gymnasium.envs.registration import register

register(
    id="<name of your environment>",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"scenario_file": "<path to .cfg file>"},
)
```

And then used as any other Gymnasium environment:

```{code-block} python
env = gymnasium.make("<name of your environment>")
```
