# Types

* [Label](#label)
* [GameState](#gamestate)
* [Enums](#enums)
	* [Mode](#mode)
	* [ScreenFormat](#screenformat)
 	* [ScreenResolution](#screenresolution)
 	* [AutomapMode](#automapmode)
 	* [GameVariable](#gamevariable)
 	* [Button](#button)
 		* [binary buttons](#binarybuttons)
 		* [delta buttons](#deltabuttons)


## C++ only

- `Buffer (std::vector<uint8_t>)`
- `BufferPtr (std::shared_ptr<Buffer>)`
- `GameStatePtr (std::shared_ptr<GameState>)`


## Structures

---
### <a name="label"></a> `Label`
(`C++ type / Lua type / Java type / Python type` **name**)

- `unsigned int / number / unsigned int / int` **objectId / object_id**
- `std::string / string / String / str` **objectName / object_name**
- `uint8_t / number / byte / int` **value**
- `double / number / double / float` **objectPositionX / object_position_x**
- `double / number / double / float` **objectPositionX / object_position_y**
- `double / number / double / float` **objectPositionX / object_position_z**

**objectId / object_id** - unique object instance ID - assigned when object is seen for the first time 
(so object with lower id was seen before object with higher).


---
### <a name="gamestate"></a> `GameState`
(`C++ type / Lua type / Java type / Python type` **name**)

- `unsigned int / number / unsigned int / int` **number**
- `std::vector<float> / DoubleTensor / double[] / numpy.double[]` **gameVariables / game_variables**
- `BufferPtr / ByteTensor / byte[] / numpy.uint8[]` **screenBuffer / screen_buffer**
- `BufferPtr / ByteTensor / byte[] / numpy.uint8[]` **depthBuffer / depth_buffer**
- `BufferPtr / ByteTensor / byte[] / numpy.uint8[]` **labelsBuffer / labels_buffer**
- `BufferPtr / ByteTensor / byte[] / numpy.uint8[]` **automapBuffer / automap_buffer**
- `std::vector<Label> / table / Label[] / list` **labels**


## <a name="enums"></a> Enums

---
### <a name="mode"></a> `Mode`

- **PLAYER** - synchronous player mode
- **SPECTATOR** - synchronous spectator mode
- **ASYNC_PLAYER** - asynchronous player mode
- **ASYNC_SPECTATOR** - asynchronous spectator mode


---
### <a name="screenformat"></a>`ScreenFormat`

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


---
### <a name="screenresolution"></a>`ScreenResolution`

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


---
### <a name="automapmode"></a> `AutomapMode`
- **NORMAL**    - Only level architecture the player has seen is shown.
- **WHOLE**     - All architecture is shown, regardless of whether or not the player has seen it.
- **OBJECTS**   - In addition to the previous, shows all things in the map as arrows pointing in the direction they are facing.
- **OBJECTS_WITH_SIZE** - In addition to the previous, all things are wrapped in a box showing their size.


---
### <a name="gamevariable"></a> `GameVariable`

#### Defined variables
- **KILLCOUNT**
- **ITEMCOUNT**
- **SECRETCOUNT**
- **FRAGCOUNT**
- **DEATHCOUNT**
- **HEALTH**        - Can be higher then 100!
- **ARMOR**         - Can be higher then 100!
- **DEAD**          - True if player is dead.
- **ON_GROUND**     - True if player is on the ground (not in the air).
- **ATTACK_READY**  - True if attack can be performed.
- **ALTATTACK_READY**       - True if altattack can be performed.
- **SELECTED_WEAPON**       - Selected weapon's number.
- **SELECTED_WEAPON_AMMO**  - Ammo for selected weapon.
- **AMMO0** - **AMMO9**     - Number of ammo for weapon in N slot.
- **WEAPON0** - **WEAPON9** - Number of weapons in N slot.
- **POSITION_X**            - Position of player
- **POSITION_Y**
- **POSITION_Z**
- **PLAYER_NUMBER**         - Player's number in multiplayer game.
- **PLAYER_COUNT**          - Number of players in multiplayer game.
- **PLAYER1_FRAGCOUNT** - **PLAYER8_FRAGCOUNT** - Number of N player's frags

#### User (ACS) variables  
- **USER1** - **USER60**

ACS global int variables can be accessed as USER GameVariables. 
global int 0 is reserved for reward and is always threaded as Doom's fixed point numeral.
Other from 1 to 60 (global int 1-60) can be access as USER1 - USER60 GameVariables.

See also:
- [ZDoom Wiki](http://zdoom.org/wiki/ACS)
- [`Utilities: doomFixedToDouble`](Utilities.md#doomFixedToDouble)

---
### <a name="button"></a> `Button`

#### <a name="binarybuttons"></a> Binary buttons

Binary buttons have only 2 states "not pressed" if value 0 and "pressed" if value greater then 0.

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

#### <a name="deltabuttons"></a> Delta buttons

Buttons whose value defines the speed of movement. 
A positive value indicates movement in the first specified direction and a negative value in the second direction. 
For example: value 10 for MOVE_LEFT_RIGHT_DELTA means slow movement to the right and -100 means fast movement to the left.

- **LOOK_UP_DOWN_DELTA**
- **TURN_LEFT_RIGHT_DELTA**
- **MOVE_FORWARD_BACKWARD_DELTA**
- **MOVE_LEFT_RIGHT_DELTA**
- **MOVE_UP_DOWN_DELTA**
