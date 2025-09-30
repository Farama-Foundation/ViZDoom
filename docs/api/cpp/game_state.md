# GameState

GameState is the main object returned by [`DoomGame::getState`](./doom_game.md#getstate) method.
The declarations of all the enums can be found in the `include/ViZDoomTypes.h` header file.


## `GameState`
(`C++ type / Python type` **name**)

- `unsigned int / int` **number**
- `unsigned int / int` **tic**
- `std::vector<float> / numpy.double[] | None` **gameVariables / game_variables**
- `ImageBufferPtr / numpy.uint8[] | None`  **screenBuffer / screen_buffer**
- `ImageBufferPtr / numpy.uint8[] | None`  **depthBuffer / depth_buffer**
- `ImageBufferPtr / numpy.uint8[] | None`  **labelsBuffer / labels_buffer**
- `ImageBufferPtr / numpy.uint8[] | None`  **automapBuffer / automap_buffer**
- `AudioBufferPtr / numpy.int16[] | None` **audioBuffer / audio_buffer**
- `std::string / str | None` **NotificationsBuffer / notifications_buffer**
- `std::vector<Label> / list | None`  **labels / labels**
- `std::vector<Objects> / list | None`  **objects / objects**
- `std::vector<Sectors> / list | None`  **sectors / sectors**

**number** - number of the state in the episode.
**tic** - ingame time, 1 tic is 1/35 of second in the game world. Note: added in 1.1.1.
**labels, objects, sectors** - note: added in 1.1.8.
**audioBuffer / audio_buffer** - note: added in 1.1.9.
**NotificationsBuffer / notifications_buffer** - note: added in 1.3.0.

See also:
- [`DoomGame::getState`](./doom_game.md#getstate),
- [examples/python/basic.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py).
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py).



### Types used only in C++

- `Buffer (std::vector<uint8_t>)`
- `ImageBufferPtr (std::shared_ptr<Buffer>)`
- `GameStatePtr (std::shared_ptr<GameState>)`


## Structures


### `Label`
(`C++ type / Python type` **name**)

- `unsigned int / int` **objectId / object_id**
- `std::string / str` **objectName / object_name**
- `std::string / str` **objectCategory / object_category**
- `uint8_t / int` **value**
- `unsigned int / int` **x**
- `unsigned int / int` **y**
- `unsigned int / int` **width**
- `unsigned int / int` **height**
- `double / float` **objectPositionX / object_position_x**
- `double / float` **objectPositionY / object_position_y**
- `double / float` **objectPositionZ / object_position_z**
- `double / float` **objectAngle / object_angle**
- `double / float` **objectPitch / object_pitch**
- `double / float` **objectRoll / object_roll**
- `double / float` **objectVelocityX / object_velocity_x**
- `double / float` **objectVelocityY / object_velocity_y**
- `double / float` **objectVelocityZ / object_velocity_z**


Description of the object in the labels buffer.

**objectId / object_id** - unique object ID, if both Labels and Objects information is enabled, this will be the same as **id** in corresponding **Object**.

**objectName / object_name** - ingame object name, many different objects can have the same name (e.g. Medikit, Clip, Zombie).

**objectCategory / object_category** - category of the object (e.g. "Monster", "Weapon", "Player"). Category "Self" is assigned to current player.

**value** - value that represents this particular object in **labelsBuffer**.

**x**, **y**, **width**, **height** - describes bounding box of this particular object in **labelsBuffer**. Note: added in 1.1.5.


See also:
- [`DoomGame::setLabelsBufferEnabled`](./doom_game.md#setlabelsbufferenabled),
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/labels_buffer.py).


---
### `Object`
(`C++ type / Python type` **name**)

- `unsigned int / int` **id**
- `std::string / str` **name**
- `double / float` **positionX / position_x**
- `double / float` **positionY / position_y**
- `double / float` **positionZ / position_z**
- `double / float` **angle**
- `double / float` **pitch**
- `double / float` **roll**
- `double / float` **velocityX / velocity_x**
- `double / float` **velocityY / velocity_y**
- `double / float` **velocityZ / velocity_z**

Description of the object present in the game world.

**id** - unique object ID.

**name** - ingame object name, many different objects can have the same name (e.g. Medikit, Clip, Zombie).

See also:
- [`DoomGame::setObjectsInfoEnabled`](./doom_game.md#setsectorsinfoenabled),
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py).

Note: added in 1.1.8.


---
### `Line`
(`C++ type / Python type` **name**)

- `double / float` **x1**
- `double / float` **y1**
- `double / float` **x2**
- `double / float` **y2**
- `bool / bool` **isBlocking / is_blocking**

Description of the line that is part of a sector definition.

**x1**, **y1** - position of the line's first vertex.

**x2**, **y2** - position of the line's second vertex.

**isBlocking / is_blocking** - is true, if line is a wall that can't be passed.

See also:
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py).

Note: added in 1.1.8.


---
### `Sector`
(`C++ type / Python type` **name**)

- `double / float` **floorHeight / floor_height**
- `double / float` **ceilingHeight / ceiling_height**
- `std::vector<Label> / list` **lines**

Description of the sector, part of the map with the same floor and ceiling height.

**floorHeight / floor_height** - height of the sector's floor.

**ceilingHeight / ceiling_height** - height of the sector's ceiling.

**lines** - contains list of line segments, that forms sector.

See also:
- [`DoomGame::setSectorsInfoEnabled`](./doom_game.md#setsectorsinfoenabled),
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py).

Note: added in 1.1.8.


---
## `ServerState`
(`C++ type / Python type` **name**)

- `unsigned int / int` **tic**
- `unsigned int / int` **playerCount / player_count**
- `bool[] / list` **playersInGame / players_in_game**
- `int[] / list` **playersFrags / players_frags**
- `std::string[] / list` **playersNames / players_names**
- `bool[] / list` **playersAfk / players_afk**
- `unsigned int[] / list` **playersLastActionTic / players_last_action_tic **
- `unsigned int[] / list` **playersLastKillTic / players_last_kill_tic **

ServerState is the main object returned by [`DoomGame::getServerState`](./doom_game.md#getserverstate) method, and it purpose is to get more information about the state of the multi-player game.

See also:
- [`DoomGame::getServerState`](./doom_game.md#getserverstate),

Note: added in 1.1.6.
