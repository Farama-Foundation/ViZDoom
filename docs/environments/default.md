# Default scenarios/environments

Below we describe all default scenarios (original ViZDoom nomenclature)/environments (Gymnasium/Open AI Gym nomenclature) included with ViZDoom. The PRs with new scenarios are welcome!


## Note on .wad, .cfg files, and rewards

The scenarios consist of two files - .wad and .cfg ([see scenarios directory](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios)). The .wad file contains the map and script, and the .cfg file contains additional settings. The maps contained in .wad files (Doom's engine format for storing maps and assets) usually do not implement action constraints, the death penalty, and living rewards (however it is possible). To make it easier, this can be specified in ViZDoom .cfg files as well as other options like access to additional information. These can also be overridden in the code when using the original ViZDoom API. Every mention of any settings that are not included in .wad files is specified with "(config)" in the descriptions below. ViZDoom does not support setting certain rewards (such as killing opponents) in .cfg files. These must be programmed in the .wad files instead.


## BASIC
The purpose of the scenario is just to check if using this
framework to train some AI in a 3D environment is feasible.

The map is a rectangle with gray walls, ceiling, and floor.
The player is spawned along the longer wall in the center.
A red, circular monster is spawned randomly somewhere along
the opposite wall. A player can only (config) go left/right
and shoot. 1 hit is enough to kill the monster. The episode
finishes when the monster is killed or on timeout.

__REWARDS:__

+101 for killing the monster
-5 for missing
The episode ends after killing the monster or on timeout.

Further configuration:
* living reward = -1,
* 3 available buttons: move left, move right, shoot (attack)
* timeout = 300

## DEADLY CORRIDOR
The purpose of this scenario is to teach the agent to navigate towards
his fundamental goal (the vest) and make sure he survives at the
same time.

The map is a corridor with shooting monsters on both sides (6 monsters
in total). A green vest is placed at the opposite end of the corridor.
The reward is proportional (negative or positive) to the change in the
distance between the player and the vest. If the player ignores monsters
on the sides and runs straight for the vest, he will be killed somewhere
along the way. To ensure this behavior doom_skill = 5 (config) is
needed.

__REWARDS:__

+dX for getting closer to the vest.
-dX for getting further from the vest.

Further configuration:
* 5 available buttons: turn left, turn right, move left, move right, shoot (attack)
* timeout = 4200
* death penalty = 100
* doom_skill = 5


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

__REWARDS:__
+1 for killing a monster

Further configuration:
* 3 available buttons: turn left, turn right, shoot (attack)
* death penalty = 1

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

__REWARDS:__
+1 for killing a monster

Further configuration:
* 3 available buttons: turn left, turn right, shoot (attack)
* death penalty = 1

## HEALTH GATHERING
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


Further configuration:
* living_reward = 1
* 3 available buttons: turn left, turn right, move forward
* 1 available game variable: HEALTH
* death penalty = 100

## MY WAY HOME
The purpose of this scenario is to teach the agent how to navigate
in a labyrinth-like surrounding and reach his ultimate goal
(and learn what it actually is).

The map is a series of rooms with interconnection and 1 corridor
with a dead end. Each room has a different color. There is a
green vest in one of the rooms (the same room every time).
The player is spawned in a randomly chosen room facing a random
direction. The episode ends when the vest is reached or on timeout/

__REWARDS:__
+1 for reaching the vest

Further configuration:
* 3 available buttons: turn left, turn right, move forward
* living reward = -0.0001
* timeout = 2100

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

__REWARDS:__
+1 for killing the monster

Further configuration:
* living reward = -0.0001,
* 3 available buttons: turn left, turn right, shoot (attack)
* timeout = 300

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

__REWARDS:__
+1 for each tic of life

Further configuration:
* living reward = 1.0,
* 2 available buttons: move left, move right
