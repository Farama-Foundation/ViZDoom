# Types

## C++ only

* `Buffer (std::vector<uint8_t>)`
* `BufferPtr (std::shared_ptr<Buffer>)`
* `GameStatePtr (std::shared_ptr<GameState>)`


## Structures

### `Label`

* `unsigned int / number / unsigned int / int` **objectId / object_id**
* `std::string / string / String / str` **objectName / object_name**
* `uint8_t / number / byte / int` **value**


### `GameState`
* `unsigned int / number / unsigned int / int` **number**
* `std::vector<float> / list / float[] / numpy.float[]` **gameVariables / game_variables**
* `BufferPtr / ? / byte[] / numpy.ubyte[]` **screenBuffer / screen_buffer**
* `BufferPtr / ? / byte[] / numpy.ubyte[]` **depthBuffer / depth_buffer**
* `BufferPtr / ? / byte[] / numpy.ubyte[]` **labelsBuffer / labels_buffer**
* `BufferPtr / ? / byte[] / numpy.ubyte[]` **automapBuffer / automap_buffer**
* `std::vector<Label> / list / Label[] / Label[]` **labels**


## Enums
    
### `Mode`
* **PLAYER** - synchronous player mode
* **SPECTATOR** - synchronous spectator mode
* **ASYNC_PLAYER** - asynchronous player mode
* **ASYNC_SPECTATOR** - asynchronous spectator mode


### `ScreenFormat`
* **CRCGCB** - 3 channels of 8-bit values in RGB order
* **RGB24** - channel of RGB values stored in 24 bits, where R value is stored in the oldest 8 bits
* **RGBA32** - channel of RGBA values stored in 32 bits, where R value is stored in the oldest 8 bits
* **ARGB32** - channel of ARGB values stored in 32 bits, where A value is stored in the oldest 8 bits
* **CBCGCR** - 3 channels of 8-bit values in BGR order
* **BGR24** - channel of BGR values stored in 24 bits, where B value is stored in the oldest 8 bits
* **BGRA32** - channel of BGRA values stored in 32 bits, where B value is stored in the oldest 8 bits
* **ABGR32** - channel of ABGR values stored in 32 bits, where A value is stored in the oldest 8 bits
* **GRAY8** - 8-bit gray channel
* **DOOM_256_COLORS8** - 8-bit channel with Doom palette values


### `ScreenResolution`
* **RES_160X120** (4:3)
* **RES_200X125** (16:10)
* **RES_200X150** (4:3)
* **RES_256X144** (16:9)
* **RES_256X160** (16:10)
* **RES_256X192** (4:3)
* **RES_320X180** (16:9)
* **RES_320X200** (16:10)
* **RES_320X240** (4:3)
* **RES_320X256** (5:4)
* **RES_400X225** (16:9)
* **RES_400X250** (16:10)
* **RES_400X300** (4:3)
* **RES_512X288** (16:9)
* **RES_512X320** (16:10)
* **RES_512X384** (4:3)
* **RES_640X360** (16:9)
* **RES_640X400** (16:10)
* **RES_640X480** (4:3)
* **RES_800X450** (16:9)
* **RES_800X500** (16:10)
* **RES_800X600** (4:3)
* **RES_1024X576** (16:9)
* **RES_1024X640** (16:10)
* **RES_1024X768** (4:3)
* **RES_1280X720** (16:9)
* **RES_1280X800** (16:10)
* **RES_1280X960** (4:3)
* **RES_1280X1024** (5:4)
* **RES_1400X787** (16:9)
* **RES_1400X875** (16:10)
* **RES_1400X1050** (4:3)
* **RES_1600X900** (16:9)
* **RES_1600X1000** (16:10)
* **RES_1600X1200** (4:3)
* **RES_1920X1080** (16:9)


### `AutomapMode`
* **NORMAL** - Only level architecture the player has seen is shown.
* **WHOLE** - All architecture is shown, regardless of whether or not the player has seen it.
* **OBJECTS** - In addition to the previous, shows all things in the map as arrows pointing in the direction they are facing.
* **OBJECTS_WITH_SIZE** - In addition to the previous, all things are wrapped in a box showing their size.


### `GameVariable`
* **KILLCOUNT**
* **ITEMCOUNT**
* **SECRETCOUNT**
* **FRAGCOUNT**
* **DEATHCOUNT**
* **HEALTH**
* **ARMOR**
* **DEAD**
* **ON_GROUND**
* **ATTACK_READY**
* **ALTATTACK_READY**
* **SELECTED_WEAPON**
* **SELECTED_WEAPON_AMMO**
* **AMMO0**
* **AMMO1**
* **AMMO2**
* **AMMO3**
* **AMMO4**
* **AMMO5**
* **AMMO6**
* **AMMO7**
* **AMMO8**
* **AMMO9**
* **WEAPON0**
* **WEAPON1**
* **WEAPON2**
* **WEAPON3**
* **WEAPON4**
* **WEAPON5**
* **WEAPON6**
* **WEAPON7**
* **WEAPON8**
* **WEAPON9**
* **USER1**
* **USER2**
* **USER3**
* **USER4**
* **USER5**
* **USER6**
* **USER7**
* **USER8**
* **USER9**
* **USER10**
* **USER11**
* **USER12**
* **USER13**
* **USER14**
* **USER15**
* **USER16**
* **USER17**
* **USER18**
* **USER19**
* **USER20**
* **USER21**
* **USER22**
* **USER23**
* **USER24**
* **USER25**
* **USER26**
* **USER27**
* **USER28**
* **USER29**
* **USER30**
* **PLAYER_NUMBER**
* **PLAYER_COUNT**
* **PLAYER1_FRAGCOUNT**
* **PLAYER2_FRAGCOUNT**
* **PLAYER3_FRAGCOUNT**
* **PLAYER4_FRAGCOUNT**
* **PLAYER5_FRAGCOUNT**
* **PLAYER6_FRAGCOUNT**
* **PLAYER7_FRAGCOUNT**
* **PLAYER8_FRAGCOUNT**


### `Button`
#### Binary buttons
* **ATTACK**
* **USE**
* **JUMP**
* **CROUCH**
* **TURN180**
* **ALTATTACK**
* **RELOAD**
* **ZOOM**
* **SPEED**
* **STRAFE**
* **MOVE_RIGHT**
* **MOVE_LEFT**
* **MOVE_BACKWARD**
* **MOVE_FORWARD**
* **TURN_RIGHT**
* **TURN_LEFT**
* **LOOK_UP**
* **LOOK_DOWN**
* **MOVE_UP**
* **MOVE_DOWN**
* **LAND**
* **SELECT_WEAPON1**
* **SELECT_WEAPON2**
* **SELECT_WEAPON3**
* **SELECT_WEAPON4**
* **SELECT_WEAPON5**
* **SELECT_WEAPON6**
* **SELECT_WEAPON7**
* **SELECT_WEAPON8**
* **SELECT_WEAPON9**
* **SELECT_WEAPON0**
* **SELECT_NEXT_WEAPON**
* **SELECT_PREV_WEAPON**
* **DROP_SELECTED_WEAPON**
* **ACTIVATE_SELECTED_ITEM**
* **SELECT_NEXT_ITEM**
* **SELECT_PREV_ITEM**
* **DROP_SELECTED_ITEM**

#### Delta buttons
* **LOOK_UP_DOWN_DELTA**
* **TURN_LEFT_RIGHT_DELTA**
* **MOVE_FORWARD_BACKWARD_DELTA**
* **MOVE_LEFT_RIGHT_DELTA**
* **MOVE_UP_DOWN_DELTA**