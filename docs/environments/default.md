# Default scenarios/environments

Below we describe a default singleplayer scenarios (original ViZDoom nomenclature)/environments (Gymnasium/Open AI Gym nomenclature) included with ViZDoom. The PRs with new scenarios are welcome!


## How to use default scenarios

When using original ViZDoom API, each scenario can be loaded [`load_config`](../api/python/doom_game.md#vizdoom.DoomGame.load_config) (Python)/[`loadConfig`](../api/cpp/doom_game.md#loadconfig) (C++) method:

Python example:
```{code-block} python
import os
import vizdoom as vzd
game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg")) # or any other scenario file
```

When using Gymnasium API the scenario can be loaded by passing the scenario id to `make` method like-this:

```{code-block} python
import gymnasium
from vizdoom import gymnasium_wrapper # This import will register all the environments

env = gymnasium.make("VizdoomBasic-v0") # or any other environment id
```



## Note on .wad, .cfg files, and rewards

A scenario usually consist of two files - .wad and .cfg ([see scenarios directory](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios)). The .wad file contains the map and script, and the .cfg file contains additional settings. The maps contained in .wad files (Doom's engine format for storing maps and assets) usually do not implement action constraints, the death penalty, and living rewards (however it is possible). To make it easier, this can be specified in ViZDoom .cfg files as well as other options like access to additional information. These can also be overridden in the code when using the original ViZDoom API. Every mention of any settings that are not included in .wad files is specified with "(config)" in the descriptions below. ViZDoom does not support setting certain rewards (such as killing opponents) in .cfg files. These must be programmed in the .wad files instead.


## BASIC
The purpose of the scenario is just to check if using this
framework to train some AI in a 3D environment is feasible.

The map is a rectangle with gray walls, ceiling, and floor.
The player is spawned along the longer wall in the center.
A red, circular monster is spawned randomly somewhere along
the opposite wall. A player can only (config) go left/right
and shoot. 1 hit is enough to kill the monster. The episode
finishes when the monster is killed or on timeout.

**REWARDS:**
* +106 for killing the monster
* -5 for every shot
* +1 for every tic the agent is alive

The episode ends after killing the monster or on timeout.

**CONFIGURATION:**
* 3 available buttons: move left/right, shoot (attack)
* 1 available game variable: player's ammo
* timeout = 300 tics


**Gymnasium/Gym id: `"VizdoomBasic-v0"`**

**Configuration file: [basic.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/basic.cfg)**


## BASIC AUDIO
The environment is similar to the BASIC scenario, 
but the monster is invisible instead it emits sounds.
The purpose of this scenario is to check if the agent can
learn to use audio information to find and kill the monster.

**REWARDS:**
* +106 for killing the monster
* -5 for every shot
* +1 for every tic the agent is alive

The episode ends after killing the monster or on timeout.

**CONFIGURATION:**
* 3 available buttons: move left/right, shoot (attack)
* 1 available game variable: player's ammo
* timeout = 300 tics


**Gymnasium/Gym id: `"VizdoomBasicAudio-v0"`**

**Configuration file: [basic_audio.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/basic_audio.cfg)**


## DEADLY CORRIDOR
The purpose of this scenario is to teach the agent to navigate towards
his fundamental goal (the vest) and make sure he survives at the
same time.

The map is a corridor with shooting monsters on both sides (6 monsters
in total). A green vest is placed at the opposite end of the corridor.
The reward is proportional (negative or positive) to the change in the
distance between the player and the vest. If the player ignores monsters
on the sides and runs straight for the vest, he will be killed somewhere
along the way. To ensure this behavior difficulty level (`doom_skill`) = 5 (config) is
needed.

**REWARDS:**
* +dX for getting closer to the vest.
* -dX for getting further from the vest.
* -100 for death

**CONFIGURATION:**
* 7 available buttons: move forward/backwward/left/right, turn left/right, shoot (attack)
* 1 available game variable: player's health
* timeout = 2100
* difficulty level (`doom_skill`) = 5

**Gymnasium/Gym id: `"VizdoomCorridor-v0"`**

**Configuration file: [deadly_corridor.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/deadly_corridor.cfg)**


## DEATHMATCH
In this scenario, the agent is spawned in the random place of the arena filled with resources.
A random monster is spawned every few seconds that will try to kill the player.
The reward for killing a monster depends on its difficulty.
The aim of the agent is to kill as many monsters as possible
before the time runs out or it's killed by monsters.

**REWARDS:**
* Different rewards are given for killing different monsters

**CONFIGURATION:**
* 16 available binary buttons: move forward/backwward/left/right, turn left/right, strafe, sprint (speed), shoot (attack), select weapon 1-6/next/previous
* 3 available delta buttons: look up/down, turn left/right, move left/right
* 5 available game variables: player's health, armor, selected weapon and ammo, killcount
* timeout = 4200
* difficulty level (`doom_skill`) = 3

**Gymnasium/Gym id: `"VizdoomDeathmatch-v0"`**

**Configuration file: [scenarios/deathmatch.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/deathmatch.cfg)**


## DEFEND THE CENTER
The purpose of this scenario is to teach the agent that killing the
monsters is GOOD and when monsters kill you is BAD. In addition,
wasting ammunition is not very good either. Agent is rewarded only
for killing monsters so he has to figure out the rest for himself.

The map is a large circle. A player is spawned in the exact center.
5 melee-only, monsters are spawned along the wall. Monsters are
killed after a single shot. After dying, each monster is respawned
after some time. The episode ends when the player dies (it's inevitable
because of limited ammo).

**REWARDS:**
* +1 for killing a monster
* -1 for death

**CONFIGURATION:**
* 3 available buttons: turn left, turn right, shoot (attack)
* 2 available game variables: player's health and ammo
* timeout = 2100
* difficulty level (`doom_skill`) = 3

**Gymnasium/Gym id: `"VizdoomDefendCenter-v0"`**

**Configuration file: [defend_the_center.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/defend_the_center.cfg)**


## DEFEND THE LINE
The purpose of this scenario is to teach an agent that killing the
monsters is GOOD and when monsters kill you is BAD. In addition,
wasting ammunition is not very good either. The agent is rewarded only
for killing monsters, so it has to figure out the rest for itself.

The map is a rectangle. A player is spawned along the longer wall in the
center. 3 melee-only and 3 shooting monsters are spawned along the
opposite wall. Monsters are killed after a single shot, at first.
After dying, each monster is respawned after some time and can endure
more damage. The episode ends when the player dies (it's inevitable
because of limited ammo).

**REWARDS:**
* +1 for killing a monster
* -1 for death

**CONFIGURATION:**
* 3 available buttons: turn left, turn right, shoot (attack)
* 2 available game variables: player's health and ammo
* difficulty level (`doom_skill`) = 3

**Gymnasium/Gym id: `"VizdoomDefendLine-v0"`**

**Configuration file: [defend_the_line.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/defend_the_line.cfg)**


## HEALTH GATHERING (AND HEALTH GATHERING SUPREME)
The purpose of this scenario is to teach the agent how to survive
without knowing what makes him survive. An agent knows only that life
is precious, and death is bad, so he must learn what prolongs his
existence and that his health is connected with it.

The map is a rectangle with a green, acidic floor, which hurts the player
periodically. Initially, there are some medkits spread uniformly
over the map. A new medkit falls from the skies every now and then.
Medkits heal some portions of the player's health - to survive agent
needs to pick them up. The episode finishes after the player's death or
on timeout.

There is more advance version of this scenario called HEALTH GATHERING SUPREME,
that makes map layout more complex.

**REWARDS:**
* +1 for every tic the agent is alive
* -100 for death

**CONFIGURATION:**
* 3 available buttons: turn left/right, move forward
* 1 available game variable: player's health

**Gymnasium/Gym id: `"VizdoomHealthGathering-v0"`/`"VizdoomHealthGatheringSupreme-v0"`**

**Configuration file: [health_gathering.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/health_gathering.cfg)/[health_gathering_supreme.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/health_gathering_supreme.cfg)**


## MY WAY HOME
The purpose of this scenario is to teach the agent how to navigate
in a labyrinth-like surrounding and reach his ultimate goal
(and learn what it actually is).

The map is a series of rooms with interconnection and 1 corridor
with a dead end. Each room has a different color. There is a
green vest in one of the rooms (the same room every time).
The player is spawned in a randomly chosen room facing a random
direction. The episode ends when the vest is reached or on timeout/

**REWARDS:**
* +1 for reaching the vest
* -0.0001 for every tic the agent is alive

**CONFIGURATION:**
* 3 available buttons: turn left/right, move forward
* timeout = 2100

**Gymnasium/Gym id: `"VizdoomMyWayHome-v0"`**

**Configuration file: [my_way_home.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/my_way_home.cfg)**


## PREDICT POSITION
The purpose of the scenario is to teach an agent to synchronize
missile weapon shot (involving a significant delay between
shooting and hitting) with target movements. Agent should be
able to shoot so that the missile and the monster meet each other.

The map is a rectangular room. A player is spawned along the longer
wall in the center. A monster is spawned randomly somewhere
along the opposite wall and walks between left and right corners
along the wall. The player is equipped with a rocket launcher and
a single rocket. The episode ends when the missile hits a wall/the monster
or on timeout.

**REWARDS:**
* +1 for killing the monster
* -0.0001 for every tic the agent is alive

**CONFIGURATION:**
* 3 available buttons: turn left/right, shoot (attack)
* timeout = 300

**Gymnasium/Gym id: `"VizdoomPredictPosition-v0"`**

**Configuration file: [predict_position.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/predict_position.cfg)**


## TAKE COVER
The purpose of this scenario is to teach an agent to link incoming
missiles with his estimated lifespan. An agent should learn that
being hit means health decrease, and this in turn will lead to
death which is undesirable. In effect, the agent should avoid
missiles.

The map is a rectangle. A player is spawned along the longer wall,
in the center. A couple of shooting monsters are spawned
randomly somewhere along the opposite wall and try to kill
the player with fireballs. The player can only (config) move
left/right. More monsters appear with time. The episode ends when
the player dies.

**REWARDS:**
* +1 for every tic the agent is alive

**CONFIGURATION:**
* 2 available buttons: move left/right
* 1 available game variable: player's health
* difficulty level (`doom_skill`) = 4

**Gymnasium/Gym id: `"VizdoomTakeCover-v0"`**

**Configuration file: [take_cover.cfg](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios/take_cover.cfg)**
