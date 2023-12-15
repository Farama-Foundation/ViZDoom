# Enums

ViZDoom is using few types of enums as parameters for its functions. Below you can find the description of all of them.
The declarations of all the enums can be found in the `include/ViZDoomTypes.h` header file.

## `Mode`

Enum type that defines all supported modes.

- **PLAYER** - synchronous player mode
- **SPECTATOR** - synchronous spectator mode
- **ASYNC_PLAYER** - asynchronous player mode
- **ASYNC_SPECTATOR** - asynchronous spectator mode

In **PLAYER** and **ASYNC_PLAYER** modes, the agent controls ingame character.

In **SPECTATOR** and **ASYNC_SPECTATOR** modes, ingame character should be controlled by the human and the agent gets information about the human action.

In **PLAYER** and **SPECTATOR** modes, the game waits for agent action or permission to continue.

In **ASYNC** modes the game progress with constant speed (default 35 tics per second, this can be set) without waiting for the agent actions.

All modes can be used in singleplayer and multiplayer.

See also:
- [`DoomGame::getMode`](./doomGame.md#getmode),
- [`DoomGame::setMode`](./doomGame.md#setmode),
- [`DoomGame::getTicrate`](./doomGame.md#getticrate),
- [`DoomGame::setTicrate`](./doomGame.md#setticrate).


---
## `ScreenFormat`

Enum type that defines all supported **screenBuffer** and **automapBuffer** formats.

- **CRCGCB**    - 3 channels of 8-bit values in RGB order
- **RGB24**     - channel of RGB values stored in 24 bits, where R value is stored in the oldest 8 bits
- **RGBA32**    - channel of RGBA values stored in 32 bits, where R value is stored in the oldest 8 bits
- **ARGB32**    - channel of ARGB values stored in 32 bits, where A value is stored in the oldest 8 bits
- **CBCGCR**    - 3 channels of 8-bit values in BGR order
- **BGR24**     - channel of BGR values stored in 24 bits, where B value is stored in the oldest 8 bits
- **BGRA32**    - channel of BGRA values stored in 32 bits, where B value is stored in the oldest 8 bits
- **ABGR32**    - channel of ABGR values stored in 32 bits, where A value is stored in the oldest 8 bits
- **GRAY8**     - 8-bit gray channel
- **DOOM_256_COLORS8** - 8-bit channel with Doom palette values


In **CRCGCB** and **CBCGCR** format **screenBuffer** and **automapBuffer** store all red 8-bit values then all green values and then all blue values, each channel is considered separately. As matrices they have [3, y, x] shape.

In **RGB24** and **BGR24** format **screenBuffer** and **automapBuffer** store 24 bit RGB triples. As matrices they have [y, x, 3] shape.

In **RGBA32**, **ARGB32**, **BGRA32** and **ABGR32** format **screenBuffer** and **automapBuffer** store 32 bit sets of RBG + alpha values. As matrices they have [y, x, 4] shape.

In **GRAY8** and **DOOM_256_COLORS8** format **screenBuffer** and **automapBuffer** store single 8 bit values. As matrices they have [y, x] shape.

**depthBuffer** and **lablesBuffer** always store single 8-bit values, so they always have [y, x] shape.

See also:
- [`DoomGame::getScreenFormat`](./doomGame.md#getscreenformat),
- [`DoomGame::setScreenFormat`](./doomGame.md#setscreenformat),
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py).


---
## `ScreenResolution`

Enum type that defines all supported resolutions - shapes of **screenBuffer**, **depthBuffer**, **labelsBuffer** and **automapBuffer** in **State**.

- **RES_160X120** (4:3)
- **RES_200X125** (16:10)
- **RES_200X150** (4:3)
- **RES_256X144** (16:9)
- **RES_256X160** (16:10)
- **RES_256X192** (4:3)
- **RES_320X180** (16:9)
- **RES_320X200** (16:10)
- **RES_320X240** (4:3)
- **RES_320X256** (5:4)
- **RES_400X225** (16:9)
- **RES_400X250** (16:10)
- **RES_400X300** (4:3)
- **RES_512X288** (16:9)
- **RES_512X320** (16:10)
- **RES_512X384** (4:3)
- **RES_640X360** (16:9)
- **RES_640X400** (16:10)
- **RES_640X480** (4:3)
- **RES_800X450** (16:9)
- **RES_800X500** (16:10)
- **RES_800X600** (4:3)
- **RES_1024X576** (16:9)
- **RES_1024X640** (16:10)
- **RES_1024X768** (4:3)
- **RES_1280X720** (16:9)
- **RES_1280X800** (16:10)
- **RES_1280X960** (4:3)
- **RES_1280X1024** (5:4)
- **RES_1400X787** (16:9)
- **RES_1400X875** (16:10)
- **RES_1400X1050** (4:3)
- **RES_1600X900** (16:9)
- **RES_1600X1000** (16:10)
- **RES_1600X1200** (4:3)
- **RES_1920X1080** (16:9)

See also:
- [`DoomGame::setScreenResolution`](./doomGame.md#setscreenresolution),
- [`DoomGame::getScreenWidth`](./doomGame.md#getscreenwidth),
- [`DoomGame::getScreenHeight`](./doomGame.md#getscreenheight).


---
## `AutomapMode`

Enum type that defines all **automapBuffer** modes.

- **NORMAL**    - Only level architecture the player has seen is shown.
- **WHOLE**     - All architecture is shown, regardless of whether or not the player has seen it.
- **OBJECTS**   - In addition to the previous, shows all things in the map as arrows pointing in the direction they are facing.
- **OBJECTS_WITH_SIZE** - In addition to the previous, all things are wrapped in a box showing their size.

See also:
- [`DoomGame::setAutomapMode`](./doomGame.md#setautomapmode),
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py).


---
## `GameVariable`

Enum type that defines all variables that can be obtained from the game.

### Defined variables
- **KILLCOUNT**             - Counts the number of monsters killed during the current episode. ~Killing other players/bots do not count towards this.~ From 1.1.5 killing other players/bots counts towards this.
- **ITEMCOUNT**             - Counts the number of picked up items during the current episode.
- **SECRETCOUNT**           - Counts the number of secret location/objects discovered during the current episode.
- **FRAGCOUNT**             - Counts the number of players/bots killed, minus the number of committed suicides. Useful only in multiplayer mode.
- **DEATHCOUNT**            - Counts the number of players deaths during the current episode. Useful only in multiplayer mode.
- **HITCOUNT**              - Counts number of hit monsters/players/bots during the current episode. Note: added in 1.1.5.
- **HITS_TAKEN**            - Counts number of hits taken by the player during the current episode. Note: added in 1.1.5.
- **DAMAGECOUNT**           - Counts number of damage dealt to monsters/players/bots during the current episode. Note: added in 1.1.5.
- **DAMAGE_TAKEN**          - Counts number of damage taken by the player during the current episode. Note: added in 1.1.5.
- **HEALTH**                - Can be higher then 100!
- **ARMOR**                 - Can be higher then 100!
- **DEAD**                  - True if the player is dead.
- **ON_GROUND**             - True if the player is on the ground (not in the air).
- **ATTACK_READY**          - True if the attack can be performed.
- **ALTATTACK_READY**       - True if the altattack can be performed.
- **SELECTED_WEAPON**       - Selected weapon's number.
- **SELECTED_WEAPON_AMMO**  - Ammo for selected weapon.
- **AMMO0** - **AMMO9**     - Number of ammo for weapon in N slot.
- **WEAPON0** - **WEAPON9** - Number of weapons in N slot.
- **POSITION_X**            - Position of the player, not available if `viz_nocheat` is enabled.
- **POSITION_Y**
- **POSITION_Z**
- **ANGLE**                 - Orientation of the player, not available if `viz_nocheat` is enabled.
- **PITCH**
- **ROLL**
- **VIEW_HEIGHT**           - View high of the player, not available if `viz_nocheat` is enabled. Position of the camera in Z axis is equal to **POSITION_Z** + **VIEW_HEIGHT**. Note: added in 1.1.7.
- **VELOCITY_X**            - Velocity of the player, not available if `viz_nocheat` is enabled.
- **VELOCITY_Y**
- **VELOCITY_Z**
- **CAMERA_POSITION_X**     - Position of the camera, not available if `viz_nocheat` is enabled. Note: added in 1.1.7.
- **CAMERA_POSITION_Y**
- **CAMERA_POSITION_Z**
- **CAMERA_ANGLE**          - Orientation of the camera, not available if `viz_nocheat` is enabled. Note: added in 1.1.7.
- **CAMERA_PITCH**
- **CAMERA_ROLL**
- **CAMERA_FOV**            - Field of view in degrees, not available if `viz_nocheat` is enabled. Note: added in 1.1.7.
- **PLAYER_NUMBER**         - Player's number in multiplayer game.
- **PLAYER_COUNT**          - Number of players in multiplayer game.
- **PLAYER1_FRAGCOUNT** - **PLAYER16_FRAGCOUNT** - Number of player's frags (number of kills - suicides in multiplayer deathmatch).


### User (ACS) variables
- **USER1** - **USER60**

ACS global int variables can be accessed as USER GameVariables.
global int 0 is reserved for reward and is always treated as Doom's fixed point numeral.
Other from 1 to 60 (global int 1-60) can be accessed as `USER1` - `USER60` GameVariables.
If you assign fixed point numeral to `USER1` - `USER60` GameVariables,
you can convert them to floating point by using [`doomFixedToDouble`](utils.md#doomfixedtodouble) function.

See also:
- [ZDoom Wiki: ACS](http://zdoom.org/wiki/ACS),
- [`DoomGame::getAvailableGameVariables`](./doomGame.md#getavailablegamevariables),
- [`DoomGame::setAvailableGameVariables`](./doomGame.md#setavailablegamevariables),
- [`DoomGame::addAvailableGameVariable`](./doomGame.md#addavailablegamevariable),
- [`DoomGame::getGameVariable`](./doomGame.md#getgamevariable),
- [`doomFixedToDouble`](utils.md#doomfixedtodouble),
- [examples/python/basic.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/shaping.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/shaping.py).


---
## `Button`

Enum type that defines all buttons that can be "pressed" by the agent.

### Binary buttons

Binary buttons have only 2 states "not pressed" if value 0 and "pressed" if value other then 0.

- **ATTACK**
- **USE**
- **JUMP**
- **CROUCH**
- **TURN180**
- **ALTATTACK**
- **RELOAD**
- **ZOOM**
- **SPEED**
- **STRAFE**
- **MOVE_RIGHT**
- **MOVE_LEFT**
- **MOVE_BACKWARD**
- **MOVE_FORWARD**
- **TURN_RIGHT**
- **TURN_LEFT**
- **LOOK_UP**
- **LOOK_DOWN**
- **MOVE_UP**
- **MOVE_DOWN**
- **LAND**
- **SELECT_WEAPON1**
- **SELECT_WEAPON2**
- **SELECT_WEAPON3**
- **SELECT_WEAPON4**
- **SELECT_WEAPON5**
- **SELECT_WEAPON6**
- **SELECT_WEAPON7**
- **SELECT_WEAPON8**
- **SELECT_WEAPON9**
- **SELECT_WEAPON0**
- **SELECT_NEXT_WEAPON**
- **SELECT_PREV_WEAPON**
- **DROP_SELECTED_WEAPON**
- **ACTIVATE_SELECTED_ITEM**
- **SELECT_NEXT_ITEM**
- **SELECT_PREV_ITEM**
- **DROP_SELECTED_ITEM**


### Delta buttons

Buttons whose value defines the speed of movement.
A positive value indicates movement in the first specified direction and a negative value in the second direction.
For example: value 10 for `MOVE_LEFT_RIGHT_DELTA` means slow movement to the right and -100 means fast movement to the left.

- **LOOK_UP_DOWN_DELTA**
- **TURN_LEFT_RIGHT_DELTA**
- **MOVE_FORWARD_BACKWARD_DELTA**
- **MOVE_LEFT_RIGHT_DELTA**
- **MOVE_UP_DOWN_DELTA**

In case of **TURN_LEFT_RIGHT_DELTA** and **LOOK_UP_DOWN_DELTA** values correspond to degrees.
In case of **MOVE_FORWARD_BACKWARD_DELTA**, **MOVE_LEFT_RIGHT_DELTA**, **MOVE_UP_DOWN_DELTA** values correspond to Doom Map unit (see Doom Wiki if you want to know how it translates into real life units).

See also:
- [Doom Wiki: Map unit](https://doomwiki.org/wiki/Map_unit),
- [`DoomGame::getAvailableButtons`](./doomGame.md#getavailablebuttons),
- [`DoomGame::setAvailableButtons`](./doomGame.md#setavailablebuttons),
- [`DoomGame::addAvailableButton`](./doomGame.md#addavailablebutton),
- [`DoomGame::setButtonMaxValue`](./doomGame.md#setbuttonmaxvalue),
- [`DoomGame::getButtonMaxValue`](./doomGame.md#getbuttonmaxvalue),
- [examples/python/basic.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/delta_buttons.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/delta_buttons.py),
- [GitHub issue: Angle changes by executing certain commands](https://github.com/Farama-Foundation/ViZDoom/issues/182).


## `SamplingRate`

Enum type that defines all supported sampling rates for **audioBuffer** in **State**.

- **SR_11025**
- **SR_22050**
- **SR_44100**

See also:
- [`DoomGame::setAudioSamplingRate`](./doomGame.md#setaudiosamplingrate).

Note: added in 1.1.9.
