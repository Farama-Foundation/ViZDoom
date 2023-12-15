# Enums

ViZDoom is using few types of enums as parameters for its functions.

```{eval-rst}
.. autoclass:: vizdoom.Mode
```

In **PLAYER** and **ASYNC_PLAYER** modes, the agent controls ingame character.

In **SPECTATOR** and **ASYNC_SPECTATOR** modes, ingame character should be controlled by the human and the agent gets information about the human action.

In **PLAYER** and **SPECTATOR** modes, the game waits for agent action or permission to continue.

In **ASYNC** modes the game progress with constant speed (default 35 tics per second, this can be set) without waiting for the agent actions.

All modes can be used in singleplayer and multiplayer.

See also:
- [`DoomGame.get_mode`](./doomGame.md#vizdoom.DoomGame.get_mode),
- [`DoomGame.set_mode`](./doomGame.md#vizdoom.DoomGame.set_mode),
- [`DoomGame.get_ticrate`](./doomGame.md#vizdoom.DoomGame.get_ticrate),
- [`DoomGame.set_ticrate`](./doomGame.md#vizdoom.DoomGame.set_ticrate).


```{eval-rst}
.. autoclass:: vizdoom.ScreenFormat
```

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


In **CRCGCB** and **CBCGCR** format **screenBuffer** and **automapBuffer** store all red 8-bit values then all green values and then all blue values, each channel is considered separately. As matrices (`np.ndarray`) they have `(3, y, x)` shape.

In **RGB24** and **BGR24** format **screenBuffer** and **automapBuffer** store 24 bit RGB triples. As matrices (`np.ndarray`) they have `(y, x, 3)` shape.

In **RGBA32**, **ARGB32**, **BGRA32** and **ABGR32** format **screenBuffer** and **automapBuffer** store 32 bit sets of RBG + alpha values. As matrices (`np.ndarray`) they have `(y, x, 4)` shape.

In **GRAY8** and **DOOM_256_COLORS8** format **screenBuffer** and **automapBuffer** store single 8 bit values. As matrices (`np.ndarray`) they have `(y, x)` shape.

**depthBuffer** and **lablesBuffer** always store single 8-bit values, so as matrices (`np.ndarray`) they always have `(y, x)` shape.

See also:
- [`DoomGame.get_screen_format`](./doomGame.md#vizdoom.DoomGame.get_screen_format),
- [`DoomGame.set_screen_format`](./doomGame.md#vizdoom.DoomGame.set_screen_format),
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py).


```{eval-rst}
.. autoclass:: vizdoom.ScreenResolution
```

Enum type that defines all supported resolutions - shapes of **screenBuffer**, **depthBuffer**, **labelsBuffer** and **automapBuffer** in **State**.

See also:
- [`DoomGame.set_screen_resolution`](./doomGame.md#vizdoom.DoomGame.set_screen_resolution),
- [`DoomGame.get_screen_width`](./doomGame.md#vizdoom.DoomGame.get_screen_width),
- [`DoomGame.get_screen_height`](./doomGame.md#vizdoom.DoomGame.get_screen_height).


```{eval-rst}
.. autoclass:: vizdoom.AutomapMode
```

Enum type that defines all **automapBuffer** modes.

- **NORMAL**    - Only level architecture the player has seen is shown.
- **WHOLE**     - All architecture is shown, regardless of whether or not the player has seen it.
- **OBJECTS**   - In addition to the previous, shows all things in the map as arrows pointing in the direction they are facing.
- **OBJECTS_WITH_SIZE** - In addition to the previous, all things are wrapped in a box showing their size.

See also:
- [`DoomGame.set_automap_mode`](./doomGame.md#vizdoom.DoomGame.set_automap_mode),
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py).


```{eval-rst}
.. autoclass:: vizdoom.GameVariable
```

Enum type that defines all variables that can be obtained from the game. Below we describe the meaning of each variable.

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
- **POSITION_X, POSITION_Y, POSITION_Z**        - Position of the player, not available if `viz_nocheat` is enabled.
- **ANGLE, PITCH, ROLL**                        - Orientation of the player, not available if `viz_nocheat` is enabled.
- **VIEW_HEIGHT**           - View high of the player, not available if `viz_nocheat` is enabled. Position of the camera in Z axis is equal to **POSITION_Z** + **VIEW_HEIGHT**. Note: added in 1.1.7.
- **VELOCITY_X, VELOCITY_Y, VELOCITY_Z**        - Velocity of the player, not available if `viz_nocheat` is enabled.
- **CAMERA_POSITION_X, CAMERA_POSITION_Y, CAMERA_POSITION_Z**   - Position of the camera, not available if `viz_nocheat` is enabled. Note: added in 1.1.7.
- **CAMERA_ANGLE, CAMERA_PITCH, CAMERA_ROLL**   - Orientation of the camera, not available if `viz_nocheat` is enabled. Note: added in 1.1.7.
- **CAMERA_FOV**            - Field of view in degrees, not available if `viz_nocheat` is enabled. Note: added in 1.1.7.
- **PLAYER_NUMBER**         - Player's number in multiplayer game.
- **PLAYER_COUNT**          - Number of players in multiplayer game.
- **PLAYER1_FRAGCOUNT** - **PLAYER16_FRAGCOUNT** - Number of player's frags (number of kills - suicides in multiplayer deathmatch).
- **USER1** - **USER60**    - user defined variables. ACS global int variables can be accessed as USER GameVariables.
global int 0 is reserved for reward and is always threaded as Doom fixed point numeral.
Other from 1 to 60 (global int 1-60) can be accessed as USER1 - USER60 GameVariables.
If you assign fixed point numeral to USER1 - USER60 GameVariables,
you can convert them to floating point by using [`doom_fixed_to_float`](./utils.md#vizdoom.doom_fixed_to_float) function.

See also:
- [ZDoom Wiki: ACS](http://zdoom.org/wiki/ACS),
- [`DoomGame.get_available_game_variables`](./doomGame.md#vizdoom.DoomGame.get_available_game_variables),
- [`DoomGame.set_available_game_variables`](./doomGame.md#vizdoom.DoomGame.set_available_game_variables),
- [`DoomGame.add_available_game_variable`](./doomGame.md#vizdoom.DoomGame.add_available_game_variable),
- [`DoomGame.get_game_variable`](./doomGame.md#vizdoom.DoomGame.get_game_variable),
- [`doom_fixed_to_float`](./utils.md#vizdoom.doom_fixed_to_float),
- [examples/python/basic.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/shaping.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/shaping.py).



```{eval-rst}
.. autoclass:: vizdoom.Button
```

Enum type that defines all buttons that can be "pressed" by the agent. They can be divided into two categories:

1. Delta buttons whose value defines the speed of movement.
A positive value indicates movement in the first specified direction and a negative value in the second direction.
For example: value 10 for MOVE_LEFT_RIGHT_DELTA means slow movement to the right and -100 means fast movement to the left.
- **LOOK_UP_DOWN_DELTA, TURN_LEFT_RIGHT_DELTA** - where value correspond to degrees.
- **MOVE_FORWARD_BACKWARD_DELTA, MOVE_LEFT_RIGHT_DELTA, MOVE_UP_DOWN_DELTA** - where values correspond to Doom Map unit (see Doom Wiki if you want to know how it translates into real life units).

2. Binary buttons
Binary buttons have only 2 states "not pressed" if value 0 and "pressed" if value other then 0. They are all the other buttons not listed above as delta buttons.

See also:
- [Doom Wiki: Map unit](https://doomwiki.org/wiki/Map_unit),
- [`DoomGame.get_available_buttons`](./doomGame.md#vizdoom.DoomGame.get_available_buttons),
- [`DoomGame.set_available_buttons`](./doomGame.md#vizdoom.DoomGame.set_available_buttons),
- [`DoomGame.add_available_button`](./doomGame.md#vizdoom.DoomGame.add_available_button),
- [`DoomGame.set_button_max_value`](./doomGame.md#vizdoom.DoomGame.set_button_max_value),
- [`DoomGame.get_button_max_value`](./doomGame.md#vizdoom.DoomGame.get_button_max_value),
- [examples/python/basic.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/delta_buttons.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/delta_buttons.py),
- [GitHub issue: Angle changes by executing certain commands](https://github.com/Farama-Foundation/ViZDoom/issues/182).


```{eval-rst}
.. autoclass:: vizdoom.SamplingRate
```

Enum type that defines all supported sampling rates for **audioBuffer** in **State**.

See also:
- [`DoomGame.set_audio_sampling_rate`](./doomGame.md#vizdoom.DoomGame.set_audio_sampling_rate).

Note: added in 1.1.9.
