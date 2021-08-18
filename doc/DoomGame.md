# DoomGame

## [Flow control methods](#flow)
* [init](#init)
* [close](#close)
* [newEpisode](#newEpisode)
* [replayEpisode](#replayEpisode)
* [isRunning](#isRunning)
* [isMultiplayerGame](#isMultiplayerGame)
* [isRecordingEpisode](#isRecordingEpisode)
* [isReplayingEpisode](#isReplayingEpisode)
* [setAction](#setAction)
* [advanceAction](#advanceAction)
* [makeAction](#makeAction)
* [isNewEpisode](#isNewEpisode)
* [isEpisodeFinished](#isEpisodeFinished)
* [isPlayerDead](#isPlayerDead)
* [respawnPlayer](#respawnPlayer)
* [sendGameCommand](#sendGameCommand)
* [getState](#getState)
* [getServerState](#getServerState)
* [getLastAction](#getLastAction)
* [getEpisodeTime](#getEpisodeTime)
* [save](#save)
* [load](#load)

## [Buttons settings methods](#buttons)
* [getAvailableButtons](#getAvailableButtons)
* [setAvailableButtons](#setAvailableButtons)
* [addAvailableButton](#addAvailableButton)
* [clearAvailableButtons](#clearAvailableButtons)
* [getAvailableButtonsSize](#getAvailableButtonsSize)
* [setButtonMaxValue](#setButtonMaxValue)
* [getButtonMaxValue](#getButtonMaxValue)
* [getButton](#getButton)

## [GameVariables methods](#vars)
* [getAvailableGameVariables](#getAvailableGameVariables)
* [setAvailableGameVariables](#setAvailableGameVariables)
* [addAvailableGameVariable](#addAvailableGameVariable)
* [clearAvailableGameVariables](#clearAvailableGameVariables)
* [getAvailableGameVariablesSize](#getAvailableGameVariablesSize)
* [getGameVariable](#getGameVariable)

## [Game Arguments methods](#args)
* [addGameArgs](#addGameArgs)
* [clearGameArgs](#clearGameArgs)

## [Rewards methods](#rewards)
* [getLivingReward](#getLivingReward)
* [setLivingReward](#setLivingReward)
* [getDeathPenalty](#getDeathPenalty)
* [setDeathPenalty](#setDeathPenalty)
* [getLastReward](#getLastReward)
* [getTotalReward](#getTotalReward)

## [General game configuration methods](#settings)
* [loadConfig](#loadConfig)
* [getMode](#getMode)
* [setMode](#setMode)
* [getTicrate](#getTicrate)
* [setTicrate](#setTicrate)
* [setViZDoomPath](#setViZDoomPath)
* [setDoomGamePath](#setDoomGamePath)
* [setDoomScenarioPath](#setDoomScenarioPath)
* [setDoomMap](#setDoomMap)
* [setDoomSkill](#setDoomSkill)
* [setDoomConfigPath](#setDoomConfigPath)
* [getSeed](#getSeed)
* [setSeed](#setSeed)
* [getEpisodeStartTime](#getEpisodeStartTime)
* [setEpisodeStartTime](#setEpisodeStartTime)
* [getEpisodeTimeout](#getEpisodeTimeout)
* [setEpisodeTimeout](#setEpisodeTimeout)

## [Output/rendering setting methods](#rendering)
* [setScreenResolution](#setScreenResolution)
* [getScreenFormat](#getScreenFormat)
* [setScreenFormat](#setScreenFormat)
* [isDepthBufferEnabled](#isDepthBufferEnabled)
* [setDepthBufferEnabled](#setDepthBufferEnabled)
* [isLabelsBufferEnabled](#isLabelsBufferEnabled)
* [setLabelsBufferEnabled](#setLabelsBufferEnabled)
* [isAutomapBufferEnabled](#isAutomapBufferEnabled)
* [setAutomapBufferEnabled](#setAutomapBufferEnabled)
* [setAutomapMode](#setAutomapMode)
* [setAutomapRotate](#setAutomapRotate)
* [setAutomapRenderTextures](#setAutomapRenderTextures)
* [setRenderHud](#setRenderHud)
* [setRenderMinimalHud](#setRenderMinimalHud)
* [setRenderWeapon](#setRenderWeapon)
* [setRenderCrosshair](#setRenderCrosshair)
* [setRenderDecals](#setRenderDecals)
* [setRenderParticles](#setRenderParticles)
* [setRenderEffectsSprites](#setRenderEffectsSprites)
* [setRenderMessages](#setRenderMessages)
* [setRenderCorpses](#setRenderCorpses)
* [setRenderScreenFlashes](#setRenderScreenFlashes)
* [setRenderAllFrames](#setRenderAllFrames)
* [setWindowVisible](#setWindowVisible)
* [setConsoleEnabled](#setConsoleEnabled)
* [setSoundEnabled](#setSoundEnabled)
* [getScreenWidth](#getScreenWidth)
* [getScreenHeight](#getScreenHeight)
* [getScreenChannels](#getScreenChannels)
* [getScreenPitch](#getScreenPitch)
* [getScreenSize](#getScreenSize)
* [isObjectsInfoEnabled](#isObjectsInfoEnabled)
* [setObjectsInfoEnabled](#setObjectsInfoEnabled)
* [isSectorsInfoEnabled](#isSectorsInfoEnabled)
* [setSectorsInfoEnabled](#setSectorsInfoEnabled)
* [isAudioBufferEnabled](#isAudioBufferEnabled)
* [is/setAudioBufferEnabled](#setAudioBufferEnabled)
* [getAudioSamplingFreq](#getAudioSamplingFreq)
* [setAudioSamplingFreq](#setAudioSamplingFreq)
* [getAudioBufferSize](#getAudioBufferSize)
* [setAudioBufferSize](#setAudioBufferSize)


## <a name="flow"></a> Flow control methods:

---
### <a name="init"></a> `init`

| C++    | `bool init()`    |
| :--    | :--              |
| Python | `bool init()`    |

Initializes ViZDoom game instance and starts newEpisode.
After calling this method, the first state from a new episode will be available.
Some configuration options cannot be changed after calling this method.
Init returns true when the game was started properly and false otherwise.


---
### <a name="close"></a> `close`

| C++    | `void close()` |
| :--    | :--            |
| Python | `void close()` |

Closes ViZDoom game instance.
It is automatically invoked by the destructor.
The game can be initialized again after being closed.


---
### <a name="newEpisode"></a> `newEpisode`

| C++    | `void newEpisode(std::string recordFilePath = "")` |
| :--    | :--                                                |
| Python | `void new_episode(str recordFilePath = "")`        |

Changed in 1.1.0

Initializes a new episode. All rewards, variables and state are restarted.
After calling this method, the first state from the new episode will be available.
If the recordFilePath is not empty, the new episode will be recorded to this file (as a Doom lump).

In a multiplayer game, the host can call this method to finish the game.
Then the rest of the players must also call this method to start a new episode.


---
### <a name="replayEpisode"></a> `replayEpisode`

| C++    | `void replayEpisode(std::string filePath, unsigned int player = 0)` |
| :--    | :--                                                                 |
| Python | `void replay_episode(str filePath, int player = 0)`                 |

Added in 1.1.0

Replays recorded episode from the given file and using the perspective of the specified player.
Players are numbered from 1, `player` equal to 0 results in replaying demo using the perspective
of default player in the recording file.
After calling this method, the first state from replay will be available.
All rewards, variables and state are available during replaying episode.

See also:
- [examples/python/record_episodes.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/record_episodes.py)
- [examples/python/record_multiplayer.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/record_multiplayer.py)


---
### <a name="isRunning"></a> `isRunning`

| C++    | `bool isRunning()`    |
| :--    | :--                   |
| Python | `bool is_running()`   |

Checks if the ViZDoom game instance is running.


---
### <a name="isMultiplayerGame"></a> `isMultiplayerGame`

| C++    | `bool isMultiplayerGame()`    |
| :--    | :--                           |
| Python | `bool is_multiplayer_game()`  |

Added in 1.1.2

Checks if the game is in multiplayer mode.


---
### <a name="isRecordingEpisode"></a> `isRecordingEpisode`

| C++    | `bool isRecordingEpisode()`    |
| :--    | :--                            |
| Python | `bool is_recording_episode()`  |

Added in 1.1.5

Checks if the game is in recording mode.


---
### <a name="isReplayingEpisode"></a> `isReplayingEpisode`

| C++    | `bool isReplayingEpisode()`    |
| :--    | :--                            |
| Python | `bool is_replaying_episode()`  |

Added in 1.1.5

Checks if the game is in replaying mode.


---
### <a name="setAction"></a> `setAction`

| C++    | `void setAction(std::vector<double> const &actions)` |
| :--    | :--                                                  |
| Python | `void set_action(list actions)`                      |

Sets the player's action for the next tics.
Each value corresponds to a button specified with [`addAvailableButton`](#addAvailableButton) method
or in the configuration file (in order of appearance).


---
### <a name="advanceAction"></a> `advanceAction`

| C++    | `void advanceAction(unsigned int tics = 1, bool updateState = true)` |
| :--    | :--                                                                  |
| Python | `void advance_action(int tics = 1, bool updateState = True)`         |

Processes a specified number of tics. If `updateState` is set the state will be updated after last processed tic
and a new reward will be calculated. To get new state use `getState` and to get the new reward use `getLastReward`.
If `updateState` is not set the state will not be updated.


---
### <a name="makeAction"></a> `makeAction`

| C++    | `double makeAction(std::vector<double> const &actions, unsigned int tics = 1)` |
| :--    | :--                                                                            |
| Python | `float make_action(list actions, int tics = 1)`                               |

Method combining usability of [`setAction`](#setAction), [`advanceAction`](#advanceAction) and [`getLastReward`](#getLastReward).
Sets the player's action for the next tics, processes a specified number of tics,
updates the state and calculates a new reward, which is returned.


---
### <a name="isNewEpisode"></a> `isNewEpisode`

| C++    | `bool isNewEpisode()`    |
| :--    | :--                      |
| Python | `bool is_new_episode()`  |

Returns true if the current episode is in the initial state - the first state, no actions were performed yet.


---
### <a name="isEpisodeFinished"></a> `isEpisodeFinished`

| C++    | `bool isEpisodeFinished()`    |
| :--    | :--                           |
| Python | `bool is_episode_finished()`  |

Returns true if the current episode is in the terminal state (is finished).
[`makeAction`](#makeAction) and [`advanceAction`](#advanceAction) methods will take no effect after this point (unless [`newEpisode`](#newEpisode) method is called).


---
### <a name="isPlayerDead"></a> `isPlayerDead`

| C++    | `bool isPlayerDead()`    |
| :--    | :--                      |
| Python | `bool is_player_dead()`  |

Returns true if the player is dead.
In singleplayer, player death is equivalent to the end of the episode.
In multiplayer, when the player is dead [`respawnPlayer`](#respawnPlayer) method can be called.


---
### <a name="respawnPlayer"></a> `respawnPlayer`

| C++    | `void respawnPlayer()`  |
| :--    | :--                     |
| Python | `void respawn_player()` |

This method respawns the player after death in multiplayer mode.
After calling this method, the first state after the respawn will be available.

See also:
- [`isMultiplayerGame`](#isMultiplayerGame)


---
### <a name="sendGameCommand"></a> `sendGameCommand`

| C++    | `void sendGameCommand(std::string cmd)` |
| :--    | :--                                     |
| Python | `void send_game_command(str cmd)`       |

Sends the command to Doom console. It can be used for controlling the game, changing settings, cheats, etc.
Some commands will be blocked in some modes.

See also: 
- [ZDoom Wiki: Console](http://zdoom.org/wiki/Console)
- [ZDoom Wiki: CVARs (console variables)](https://zdoom.org/wiki/CVARs) 
- [ZDoom Wiki: CCMD (console commands)](https://zdoom.org/wiki/CCMDs) 


---
### <a name="getState"></a> `getState`

| C++    | `GameStatePtr (std::shared_ptr<GameState>) GameState getState()` |
| :--    | :--                                                              |
| Python | `GameState get_state()`                                          |

Changed in 1.1.0

Returns [`GameState`](Types.md#gamestate) object with the current game state.
If the current episode is finished, `nullptr/null/None` will be returned.

See also:
- [`Types: GameState`](Types.md#gamestate)


---
### <a name="getServerState"></a> `getServerState`

| C++    | `ServerStatePtr (std::shared_ptr<ServerState>) ServerState getServerState()` |
| :--    | :--                                                                          |
| Python | `ServerState get_state_state()`                                              |

Added in 1.1.6

Returns [`ServerState`](Types.md#serverstate) object with the current server state. 

See also:
- [`Types: ServerState`](Types.md#serverstate)


---
### <a name="getLastAction"></a> `getLastAction`

| C++    | `std::vector<double> getLastAction()` |
| :--    | :--                                   |
| Python | `list get_last_action()`              |

Returns the last action performed.
Each value corresponds to a button added with `[addAvailableButton](#addAvailableButton)` (in order of appearance).
Most useful in `SPECTATOR` mode.


---
### <a name="getEpisodeTime"></a> `getEpisodeTime`

| C++    | `unsigned int getEpisodeTime()` |
| :--    | :--                             |
| Python | `int get_episode_time()`        |

Returns number of current episode tic.


---
### <a name="save"></a> `save`

| C++    | `void save(std::string filePath)` |
| :--    | :--                               |
| Python | `void save(str filePath)`         |

Added in 1.1.9

Saves current game state to the file.


---
### <a name="load"></a> `load`

| C++    | `void load(std::string filePath)` |
| :--    | :--                               |
| Python | `void load(str filePath)`         |

Added in 1.1.9

Loads game state from the file.
A new state is available after loading.
Loading the game state does not reset the current episode state,
tic counter/time and total reward state keep their values.


## <a name="buttons"></a> Buttons settings methods

---
### <a name="getAvailableButtons"></a> `getAvailableButtons`

| C++    | `std::vector<Button> getAvailableButtons()` |
| :--    | :--                                         |
| Python | `list get_available_buttons()`              |

Returns the list of available `Buttons`.

See also:
- [`Types: Button`](Types.md#button)
- [`addAvailableButton`](#addAvailableButton)
- [`setAvailableButtons`](#addAvailableButtons)


---
### <a name="setAvailableButtons"></a> `setAvailableButtons`

| C++    | `void setAvailableButtons(std::vector<Button> buttons)` |
| :--    | :--                                                     |
| Python | `void add_available_button(list)`                       |

Set given list of `Button`s (e.g. `TURN_LEFT`, `MOVE_FORWARD`) as available `Buttons`,

Config key: `availableButtons/available_buttons` (list)

See also:
- [`Types: Button`](Types.md#button)
- [`ConfigFile: List`](ConfigFile.md#list)
- [`addAvailableButton`](#addAvailableButton)


---
### <a name="addAvailableButton"></a> `addAvailableButton`

| C++    | `void addAvailableButton(Button button, double maxValue = 0)`  |
| :--    | :--                                                            |
| Python | `void add_available_button(Button button, float maxValue = 0)` |

Add [`Button`](Types.md#button) type (e.g. `TURN_LEFT`, `MOVE_FORWARD`) to available `Buttons` and sets the maximum allowed, absolute value for the specified button.
If the given button has already been added, it will not be added again, but the maximum value is overridden.

Config key: `availableButtons/available_buttons` (list)

See also:
- [`Types: Button`](Types.md#button)
- [`ConfigFile: List`](ConfigFile.md#list)
- [`setAvailableButtons`](#addAvailableButtons)
- [`setButtonMaxValue`](#setButtonMaxValue)


---
### <a name="clearAvailableButtons"></a> `clearAvailableButtons`

| C++    | `void clearAvailableButtons()`   |
| :--    | :--                              |
| Python | `void clear_available_buttons()` |

Clears all available `Buttons` added so far.

See also:
- [`Types: Button`](Types.md#button)


---
### <a name="getAvailableButtonsSize"></a> `getAvailableButtonsSize`

| C++    | `int getAvailableButtonsSize()`    |
| :--    | :--                                |
| Python | `int get_available_buttons_size()` |

Returns the number of available `Buttons`.

See also:
- [`Types: Button`](Types.md#button)


---
### <a name="setButtonMaxValue"></a> `setButtonMaxValue`

| C++    | `void setButtonMaxValue(Button button, double maxValue = 0)`   |
| :--    | :--                                                            |
| Python | `void set_button_max_value(Button button, float maxValue = 0)` |

Sets the maximum allowed, absolute value for the specified button.
Setting the maximum value to 0 results in no constraint at all (infinity).
This method makes sense only for delta buttons.
Constraints limit applies in all Modes.

See also:
- [`Types: Button`](Types.md#button)


---
### <a name="getButtonMaxValue"></a> `getButtonMaxValue`

| C++    | `unsigned int getButtonMaxValue(Button button)` |
| :--    | :--                                             |
| Python | `int get_button_max_value(Button button)`       |

Returns the maximum allowed, absolute value for the specified button.

See also:
- [`Types: Button`](Types.md#button)


---
### <a name="getButton"></a> `getButton`

| C++    | `double getButton(Button button)` |
| :--    | :--                               |
| Python | `float get_button(Button button)` |

Returns the current state of the specified button (`ATTACK`, `USE` etc.).

See also:
- [`Types: Button`](Types.md#button)


## <a name="vars"></a> GameVariables methods

---
### <a name="getAvailableGameVariable"></a> `getAvailableGameVariable`

| C++    | `std::vector<GameVariable> getAvailableGameVariables()` |
| :--    | :--                                                     |
| Python | `list get_available_game_variables()`                   |

Returns the list of available `GameVariables`.

See also:
- [`Types: GameVariable`](Types.md#gamevariable)
- [`addAvailableGameVariable`](#addAvailableGameVariable)
- [`setAvailableGameVariables`](#setAvailableGameVariables)


---
### <a name="setAvailableGameVariables"></a> `setAvailableGameVariables`

| C++    | `void setAvailableGameVariables(std::vector<GameVariable> variables)` |
| :--    | :--                                                                   |
| Python | `void set_available_game_variables(list variables)`                   |

Set list of [`GameVariable`](Types.md#gamevariable) as available `GameVariables` in the [`GameState`](Types.md#gamestate) returned by `getState` method.

Config key: `availableGameVariables/available_game_variables` (list)

See also:
- [`Types: GameVariable`](Types.md#gamevariable)
- [`ConfigFile: List`](ConfigFile.md#list)
- [`addAvailableGameVariable`](#addAvailableGameVariable)


---
### <a name="addAvailableGameVariable"></a> `addAvailableGameVariable`

| C++    | `void addAvailableGameVariable(GameVariable variable)`    |
| :--    | :--                                                       |
| Python | `void add_available_game_variable(GameVariable variable)` |

Adds the specified [`GameVariable`](Types.md#gamevariable) to the list of available game variables (e.g. `HEALTH`, `AMMO1`, `ATTACK_READY`) in the [`GameState`](Types.md#gamestate) returned by `getState` method.

Config key: `availableGameVariables/available_game_variables` (list)

See also:
- [`Types: GameVariable`](Types.md#gamevariable)
- [`ConfigFile: List`](ConfigFile.md#list)
- [`setAvailableGameVariables`](#setAvailableGameVariables)


---
### <a name="clearAvailableGameVariables"></a> `clearAvailableGameVariables`

| C++    | `void clearAvailableGameVariables()`    |
| :--    | :--                                     |
| Python | `void clear_available_game_variables()` |

Clears the list of available `GameVariables` that are included in the GameState returned by [`getState`](#getState) method.

See also:
- [`Types: GameVariable`](Types.md#gamevariable)
- [`ConfigFile: List`](ConfigFile.md#list)


---
### <a name="getAvailableGameVariablesSize"></a> `getAvailableGameVariablesSize`

| C++    | `unsigned int getAvailableGameVariablesSize()`     |
| :--    | :--                                                |
| Python | `int get_available_game_variables_size()`          |

Returns the number of available `GameVariables`.

See also:
- [`Types: GameVariable`](Types.md#gamevariable)
- [`ConfigFile: List`](ConfigFile.md#list)


---
### <a name="getGameVariable"></a> `getGameVariable`

| C++    | `double getGameVariable(GameVariable variable)`  |
| :--    | :--                                              |
| Python | `float get_game_variable(GameVariable variable)` |

Returns the current value of the specified game variable (`HEALTH`, `AMMO1` etc.).
The specified game variable does not need to be among available game variables (included in the state).
It could be used for e.g. shaping. Returns 0 in case of not finding given `GameVariable`.

See also:
- [`Types: GameVariable`](Types.md#gamevariable)


## <a name="args"></a> Game Arguments methods

---
### <a name="addGameArgs"></a> `addGameArgs`

| C++    | `void addGameArgs(std::string args)` |
| :--    | :--                                  |
| Python | `void add_game_args(str args)`       |

Adds a custom argument that will be passed to ViZDoom process during initialization.
Useful for changing additional game settings.

Config key: `gameArgs/game_args`

See also:
- [ZDoom Wiki: Command line parameters](http://zdoom.org/wiki/Command_line_parameters)
- [ZDoom Wiki: CVARs (Console Variables)](http://zdoom.org/wiki/CVARS)


---
### <a name="clearGameArgs"></a> `clearGameArgs`

| C++    | `void clearGameArgs()`   |
| :--    | :--                      |
| Python | `void clear_game_args()` |

Clears all arguments previously added with [`addGameArgs`](#addGameArgs) method.


## <a name="rewards"></a> Reward methods

---
### <a name="getLivingReward"></a> `getLivingReward`

| C++    | `double getLivingReward()`   |
| :--    | :--                          |
| Python | `double get_living_reward()` |

Returns the reward granted to the player after every tic.


---
### <a name="setLivingReward"></a> `setLivingReward`

| C++    | `void setLivingReward(double livingReward)`  |
| :--    | :--                                          |
| Python | `void set_living_reward(float livingReward)` |

Sets the reward granted to the player after every tic. A negative value is also allowed.

Default value: 0

Config key: `livingReward/living_reward`


---
### <a name="getDeathPenalty"></a> `getDeathPenalty`

| C++    | `double getDeathPenalty()`   |
| :--    | :--                          |
| Python | `double get_death_penalty()` |

Returns the penalty for player's death.


---
### <a name="setDeathPenalty"></a> `setDeathPenalty`

| C++    | `void setDeathPenalty(double deathPenalty)`  |
| :--    | :--                                          |
| Python | `void set_death_penalty(float deathPenalty)` |

Sets a penalty for player's death. Note that in case of a negative value, the player will be rewarded upon dying.

Default value: 0

Config key: `deathPenalty/death_penalty`


---
### <a name="getLastReward"></a> `getLastReward`

| C++    | `double getLastReward()`  |
| :--    | :--                       |
| Python | `float get_last_reward()` |

Returns a reward granted after the last update of state.


---
### <a name="getTotalReward"></a> `getTotalReward`

| C++    | `double getTotalReward()`  |
| :--    | :--                        |
| Python | `float get_total_reward()` |

Returns the sum of all rewards gathered in the current episode.


## <a name="settings"></a> General game setting methods

---
### <a name="loadConfig"></a> `loadConfig`

| C++    | `bool loadConfig(std::string filePath)` |
| :--    | :--                                     |
| Python | `bool load_config(str filePath)`        |

Loads configuration (resolution, available buttons, game variables etc.) from a configuration file.
In case of multiple invocations, older configurations will be overwritten by the recent ones.
Overwriting does not involve resetting to default values. Thus only overlapping parameters will be changed.
The method returns true if the whole configuration file was correctly read and applied,
false if the file contained errors.

See also:
- [ConfigFile](ConfigFile.md)


---
### <a name="getMode"></a> `getMode`

| C++    | `Mode getMode()`  |
| :--    | :--               |
| Python | `Mode get_mode()` |

Returns current mode.


---
### <a name="setMode"></a> `setMode`

| C++    | `void setMode(Mode mode)`  |
| :--    | :--                        |
| Python | `void set_mode(Mode mode)` |

Sets mode (`PLAYER`, `SPECTATOR`, `ASYNC_PLAYER`, `ASYNC_SPECTATOR`) in which the game will be running.

Default value: `PLAYER`.

Config key: `mode`

See also:
- [`Types: Mode`](Types.md#mode)


---
### <a name="getTicrate"></a> `getTicrate`

| C++    | `unsigned int getTicrate()` |
| :--    | :--                         |
| Python | `int get_ticrate()`         |

Added in 1.1.0

Returns current ticrate.


---
### <a name="setTicrate"></a> `setTicrate`

| C++    | `void setTicrate(unsigned int ticrate)` |
| :--    | :--                                     |
| Python | `void set_ticrate(int ticrate)`         |

Added in 1.1.0

Sets ticrate for ASNYC Modes - number of logic tics executed per second.
Default Doom ticrate is 35. This value will play a game at normal speed.

Default value: 35 (default Doom ticrate).

Config key: `ticrate`

See also:
- [exmaples/python/ticrate.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/ticrate.py)


---
### <a name="setViZDoomPath"></a> `setViZDoomPath`

| C++    | `void setViZDoomPath(std::string filePath)` |
| :--    | :--                                         |
| Python | `void set_vizdoom_path(str filePath)`       |

Sets path to ViZDoom engine executable.

Default value: "$VIZDOOM.SO_LOCATION/vizdoom", "vizdoom.exe" on Windows.

Config key: `ViZDoomPath/vizdoom_path`


---
### <a name="setDoomGamePath"></a> `setDoomGamePath`

| C++    | `void setDoomGamePath(std::string filePath)` |
| :--    | :--                                          |
| Python | `void set_doom_game_path(str filePath)`      |

Sets path to the Doom engine based game file (wad format). If not used DoomGame will look for doom2.wad and freedoom2.wad (in that order) in the directory of ViZDoom's installation (where vizdoom.so is).

Default value: "$VIZDOOM.SO_LOCATION/{doom2.wad or freedoom2.wad}"

Config key: `DoomGamePath/doom_game_path`


---
### <a name="setDoomScenarioPath"></a> `setDoomScenarioPath`

| C++    | `void setDoomScenarioPath(std::string filePath)` |
| :--    | :--                                              |
| Python | `void set_doom_scenario_path(str filePath)`      |

Sets path to additional scenario file (wad format).

Default value: ""

Config key: `DoomScenarioPath/set_doom_scenario_path`


---
### <a name="setDoomMap"></a> `setDoomMap`

| C++    | `void setDoomMap(std::string map)` |
| :--    | :--                                |
| Python | `void set_doom_map(str map)`       |

Sets the map name to be used.

Default value: "map01", if set to empty "map01" will be used.

Config key: `DoomMap/doom_map`


---
### <a name="setDoomSkill"></a> `setDoomSkill`

| C++    | `void setDoomSkill(int skill)`    |
| :--    | :--                               |
| Python | `void set_doom_skill(int skill)`  |

Sets Doom game difficulty level which is called skill in Doom.
The higher is the skill, the harder the game becomes.
Skill level affects monsters' aggressiveness, monsters' speed, weapon damage, ammunition quantities etc.
Takes effect from the next episode.

- 1 - VERY EASY, “I'm Too Young to Die” in Doom.
- 2 - EASY, “Hey, Not Too Rough" in Doom.
- 3 - NORMAL, “Hurt Me Plenty” in Doom.
- 4 - HARD, “Ultra-Violence” in Doom.
- 5 - VERY HARD, “Nightmare!” in Doom.

Default value: 3

Config key: `DoomSkill/doom_skill`


---
### <a name="setDoomConfigPath"></a> `setDoomConfigPath`

| C++    | `void setDoomConfigPath(std::string filePath)` |
| :--    | :--                                            |
| Python | `void set_doom_config_path(str filePath)`      |

Sets path for ViZDoom engine configuration file.
The file is responsible for configuration of Doom engine itself.
If it does not exist, it will be created after vizdoom executable is run.
This method is not needed for most of the tasks and is added for convenience of users with hacking tendencies.

Default value: "", if left empty "_vizdoom.ini" will be used.

Config key: `DoomConfigPath/doom_config_path`


---
### <a name="getSeed"></a> `getSeed`

| C++    | `unsigned int getSeed()` |
| :--    | :--                      |
| Python | `int getSeed()`          |

Return ViZDoom's seed.


---
### <a name="setSeed"></a> `setSeed`

| C++    | `void setSeed(unsigned int seed)` |
| :--    | :--                               |
| Python | `void set_seed(int seed)`         |

Sets the seed of the ViZDoom's RNG that generates seeds (initial state) for episodes.

Default value: randomized in constructor

Config key: `seed`

See also:
- [examples/python/seed.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/seed.py)



---
### <a name="getEpisodeStartTime"></a> `getEpisodeStartTime`

| C++    | `unsigned int getEpisodeStartTime()` |
| :--    | :--                                  |
| Python | `int get_episode_start_time()`       |

Returns start delay of every episode in tics.


---
### <a name="setEpisodeStartTime"></a> `setEpisodeStartTime`

| C++    | `void setEpisodeStartTime(unsigned int tics)` |
| :--    | :--                                           |
| Python | `void set_episode_start_time(int tics)`       |

Sets start delay of every episode in tics.
Every episode will effectively start (from the user's perspective) after given number of tics.

Default value: 1

Config key: `episodeStartTime/episode_start_time`


---
### <a name="getEpisodeTimeout"></a> `getEpisodeTimeout`

| C++    | `unsigned int getEpisodeTimeout()` |
| :--    | :--                                |
| Python | `int get_episode_timeout()`        |

Returns the number of tics after which the episode will be finished.


---
### <a name="setEpisodeTimeout"></a> `setEpisodeTimeout`

| C++    | `void setEpisodeTimeout(unsigned int tics)` |
| :--    | :--                                         |
| Python | `void set_episode_timeout(int tics)`        |

Sets the number of tics after which the episode will be finished. 0 will result in no timeout.

Config key: `episodeTimeout/episode_timeout`


## <a name="rendering"></a> Output/rendering setting methods
------------------------------------------------------------------------------------------------------------

---
### <a name="setScreenResolution"></a> `setScreenResolution`

| C++    | `void setScreenResolution(ScreenResolution resolution)`   |
| :--    | :--                                                       |
| Python | `void set_screen_resolution(ScreenResolution resolution)` |

Sets the screen resolution. ZDoom engine supports only specific resolutions,
supported resolutions are part of ScreenResolution enumeration (e.g. `RES_320X240`, `RES_640X480`, `RES_1920X1080`).
The buffers, as well as the content of ViZDoom's display window, will be affected.

Default value: `RES_320X240`

Config key: `screenResolution/screen_resolution`


See also:
- [`Types: ScreenResolution`](Types.md#screenresolution)


---
### <a name="getScreenFormat"></a> `getScreenFormat`

| C++    | `ScreenFormat getScreenFormat()`   |
| :--    | :--                                |
| Python | `ScreenFormat get_screen_format()` |

Returns the format of the screen buffer and the automap buffer.


---
### <a name="setScreenFormat"></a> `setScreenFormat`

| C++    | `void setScreenFormat(ScreenFormat format)`   |
| :--    | :--                                           |
| Python | `void set_screen_format(ScreenFormat format)` |

Sets the format of the screen buffer and the automap buffer.
Supported formats are defined in `ScreenFormat` enumeration type (e.g. `CRCGCB`, `RGB24`, `GRAY8`).
The format change affects only the buffers, so it will not have any effect on the content of ViZDoom's display window.

Default value: `CRCGCB`

Config key: `screenFormat/screen_format`

See also:
- [`Types: ScreenFormat`](Types.md#screenformat)


---
### <a name="isDepthBufferEnabled"></a> `isDepthBufferEnabled`

| C++    | `bool isDepthBufferEnabled()`    |
| :--    | :--                              |
| Python | `bool isDepthBufferEnabled()`    |

Added in 1.1.0

Returns true if the depth buffer is enabled.


---
### <a name="setDepthBufferEnabled"></a> `setDepthBufferEnabled`

| C++    | `void setDepthBufferEnabled(bool depthBuffer)`    |
| :--    | :--                                               |
| Python | `void set_depth_buffer_enabled(bool depthBuffer)` |

Added in 1.1.0

Enables rendering of the depth buffer, it will be available in the state.
Depth buffer will contain noise if `viz_nocheat` is enabled.

Default value: false

Config key: `depthBufferEnabled/depth_buffer_enabled`

See also:
- [`Types: GameState`](Types.md#gamestate)
- [examples/python/buffers.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/buffers.py)


---
### <a name="isLabelsBufferEnabled"></a> `isLabelsBufferEnabled`

| C++    | `bool isLabelsBufferEnabled()`    |
| :--    | :--                               |
| Python | `bool isLabelsBufferEnabled()`    |

Added in 1.1.0

Returns true if the labels buffer is enabled.


---
### <a name="setLabelsBufferEnabled"></a> `setLabelsBufferEnabled`

| C++    | `void setLabelsBufferEnabled(bool labelsBuffer)`    |
| :--    | :--                                                 |
| Python | `void set_labels_buffer_enabled(bool labelsBuffer)` |

Added in 1.1.0

Enables rendering of the labels buffer, it will be available in the state with the vector of `Label`s.
LabelsBuffer will contain noise if `viz_nocheat` is enabled.

Default value: false

Config key: `labelsBufferEnabled/labels_buffer_enabled`

See also:
- [`Types: Label`](Types.md#label)
- [`Types: GameState`](Types.md#gamestate)
- [examples/python/labels.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/labels.py)
- [examples/python/buffers.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/buffers.py)


---
### <a name="isAutomapBufferEnabled"></a> `isAutomapBufferEnabled`

| C++    | `bool isAutomapBufferEnabled()`    |
| :--    | :--                                |
| Python | `bool is_automap_buffer_enabled()` |

Added in 1.1.0

Returns true if the automap buffer is enabled.


---
### <a name="setAutomapBufferEnabled"></a> `setAutomapBufferEnabled`

| C++    | `void setAutomapBufferEnabled(bool automapBuffer)`    |
| :--    | :--                                                   |
| Python | `void set_automap_buffer_enabled(bool automapBuffer)` |

Added in 1.1.0

Enables rendering of the automap buffer, it will be available in the state.

Default value: false

Config key: `automapBufferEnabled/automap_buffer_enabled`

See also:
- [`Types: GameState`](Types.md#gamestate)
- [examples/python/buffers.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/buffers.py),


---
### <a name="setAutomapMode"></a> `setAutomapMode`

| C++    | `void setAutomapMode(AutomapMode mode)`   |
| :--    | :--                                       |
| Python | `void set_automap_mode(AutomapMode mode)` |

Added in 1.1.0

Sets the automap mode (`NORMAL`, `WHOLE`, `OBJECTS`, `OBJECTS_WITH_SIZE`), which determines what will be visible on it.

Default value: `NORMAL`

Config key: `automapMode/set_automap_mode`

See also:
- [`Types: AutomapMode`](Types.md#automapmode)


---
### <a name="setAutomapRotate"></a> `setAutomapRotate`

| C++    | `void setAutomapRotate(bool rotate)`    |
| :--    | :--                                     |
| Python | `void set_automap_rotate(bool rotate)`  |

Added in 1.1.0

Determine if the automap will be rotating with the player. If false, north always will be at the top of the buffer.

Default value: false

Config key: `automapRotate/automap_rotate`


---
### <a name="setAutomapRenderTextures"></a> `setAutomapRenderTextures`

| C++    | `void setAutomapRenderTextures(bool textures)`    |
| :--    | :--                                               |
| Python | `void set_automap_render_textures(bool textures)` |

Added in 1.1.0

Determine if the automap will be textured, showing the floor textures.

Default value: true

Config key: `automapRenderTextures/automap_render_textures`


---
### <a name="setRenderHud"></a> `setRenderHud`

| C++    | `void setRenderHud(bool hud)`    |
| :--    | :--                              |
| Python | `void set_render_hud(bool hud)`  |

Determine if the hud will be rendered in game.

Default value: false

Config key: `renderHud/render_hud`


---
### <a name="setRenderMinimalHud"></a> `setRenderMinimalHud`

| C++    | `void setRenderMinimalHud(bool minHud)`    |
| :--    | :--                                        |
| Python | `void set_render_minimal_hud(bool minHud)` |

Added in 1.1.0

Determine if the minimalistic version of the hud will be rendered instead of the full hud.

Default value: false

Config key: `renderMinimalHud/render_minimal_hud`


---
### <a name="setRenderWeapon"></a> `setRenderWeapon`

| C++    | `void setRenderWeapon(bool weapon)`    |
| :--    | :--                                    |
| Python | `void set_render_weapon(bool weapon)`  |

Determine if the weapon held by the player will be rendered in game.

Default value: true

Config key: `renderWeapon/render_weapon`


---
### <a name="setRenderCrosshair"></a> `setRenderCrosshair`

| C++    | `void setRenderCrosshair(bool crosshair)`    |
| :--    | :--                                          |
| Python | `void set_render_crosshair(bool crosshair)`  |

Determine if the crosshair will be rendered in game.

Default value: false

Config key: `renderCrosshair/render_crosshair`


---
### <a name="setRenderDecals"></a> `setRenderDecals`

| C++    | `void setRenderDecals(bool decals)`    |
| :--    | :--                                    |
| Python | `void set_render_decals(bool decals)`  |

Determine if the decals (marks on the walls) will be rendered in game.

Default value: true

Config key: `renderDecals/render_decals`


---
### <a name="setRenderParticles"></a> `setRenderParticles`

| C++    | `void setRenderParticles(bool particles)`    |
| :--    | :--                                          |
| Python | `void set_render_particles(bool particles)`  |

Determine if the particles will be rendered in game.

Default value: true

Config key: `renderParticles/render_particles`


---
### <a name="setRenderEffectsSprites"></a> `setRenderEffectsSprites`

| C++    | `void setRenderEffectsSprites(bool sprites)`    |
| :--    | :--                                             |
| Python | `void set_render_effects_sprites(bool sprites)` |

Added in 1.1.0

Determine if some effects sprites (gun puffs, blood splats etc.) will be rendered in game.

Default value: true

Config key: `renderEffectsSprites/render_effects_sprites`


---
### <a name="setRenderMessages"></a> `setRenderMessages`

| C++    | `void setRenderMessages(bool messages)`    |
| :--    | :--                                        |
| Python | `void set_render_messages(bool messages)`  |

Added in 1.1.0

Determine if ingame messages (information about pickups, kills etc.) will be rendered in game.

Default value: false

Config key: `renderMessages/render_messages`


---
### <a name="setRenderCorpses"></a> `setRenderCorpses`

| C++    | `void setRenderCorpses(bool corpses)`    |
| :--    | :--                                      |
| Python | `void set_render_corpsess(bool corpses)` |

Added in 1.1.0

Determine if actors' corpses will be rendered in game.

Default value: true

Config key: `renderCorpses/render_corpses`


---
### <a name="setRenderScreenFlashes"></a> `setRenderScreenFlashes`

| C++    | `void setRenderScreenFlashes(bool flashes)`    |
| :--    | :--                                            |
| Python | `void set_render_screen_flashes(bool flashes)` |

Added in 1.1.3

Determine if the screen flash effect upon taking damage or picking up items will be rendered in game.

Default value: true

Config key: `renderScreenFlashes/render_screen_flashes`


---
### <a name="setRenderAllFrames"></a> `setRenderAllFrames`

| C++    | `void setRenderAllFrames(bool allFrames)`     |
| :--    | :--                                           |
| Python | `void set_render_all_frames(bool all_frames)` |

Added in 1.1.3

Determine if all frames between states will be rendered (when skip greater than 1 is used).
Allows smooth preview but can reduce performance.
It only makes sense to use it if the window is visible.

Default value: false

Config key: `renderAllFrames/render_all_frames`

See also:
- [`setWindowVisible`](#setWindowVisible)


---
### <a name="setWindowVisible"></a> `setWindowVisible`

| C++    | `void setWindowVisible(bool visibility)`    |
| :--    | :--                                         |
| Python | `void set_window_visible(bool visibility)`  |

Determines if ViZDoom's window will be visible.
ViZDoom with window disabled can be used on Linux systems without X Server.

Default value: false

Config key: `windowVisible/window_visible`


---
### <a name="setConsoleEnabled"></a> `setConsoleEnabled`

| C++    | `void setConsoleEnabled(bool console)`    |
| :--    | :--                                       |
| Python | `void set_console_enabled(bool console)`  |

Determines if ViZDoom's console output will be enabled.

Default value: false

Config key: `consoleEnabled/console_enabled`


---
### <a name="setSoundEnabled"></a> `setSoundEnabled`

| C++    | `void setSoundEnabled(bool sound)`    |
| :--    | :--                                   |
| Python | `void set_sound_enabled(bool sound)`  |

Determines if ViZDoom's sound will be played.

Default value: false

Config key: `soundEnabled/sound_enabled`


---
### <a name="getScreenWidth"></a> `getScreenWidth`

| C++    | `int getScreenWidth()`    |
| :--    | :--                       |
| Python | `int get_screen_width()`  |

Returns game's screen width - width of all buffers.


---
### <a name="getScreenHeight"></a> `getScreenHeight`

| C++    | `int getScreenHeight()`    |
| :--    | :--                        |
| Python | `int get_screen_height()`  |

Returns game's screen height - height of all buffers.


---
### <a name="getScreenChannels"></a> `getScreenChannels`

| C++    | `int getScreenChannels()`    |
| :--    | :--                          |
| Python | `int get_screen_channels()`  |

Returns number of channels in screen buffer and map buffer (depth and labels buffer always have one channel).


---
### <a name="getScreenPitch"></a> `getScreenPitch`

| C++    | `size_t getScreenPitch()` |
| :--    | :--                       |
| Python | `int get_screen_pitch()`  |

Returns size in bytes of one row in screen buffer and map buffer.


---
### <a name="getScreenSize"></a> `getScreenSize`

| C++    | `size_t getScreenSize()` |
| :--    | :--                      |
| Python | `int get_screen_size()`  |

Returns size in bytes of screen buffer and map buffer.


---
### <a name="isObjectsInfoEnabled"></a> `isObjectsInfoEnabled`

| C++    | `bool isAutomapBufferEnabled()`    |
| :--    | :--                                |
| Python | `bool isAutomapBufferEnabled()`    |

Added in 1.1.8

Returns true if the objects information is enabled.


---
### <a name="setObjectsInfoEnabled"></a> `setObjectsInfoEnabled`

| C++    | `void setObjectsInfoEnabled(bool objectsInfo)`    |
| :--    | :--                                               |
| Python | `void set_objects_info_enabled(bool objectsInfo)` |

Added in 1.1.8

Enables information about all objects present in current episode/level, it will be available in the state.

Default value: false

Config key: `objectsInfoEnabled/objects_info_enabled`

See also:
- [`Types: GameState`](Types.md#gamestate)
- [`Types: Object`](Types.md#object)
- [examples/python/objects_and_sectors.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/objects_and_sectors.py),


---
### <a name="isSectorsInfoEnabled"></a> `isSectorsInfoEnabled`

| C++    | `bool isSectorsInfoEnabled()`    |
| :--    | :--                              |
| Python | `bool is_sectors_info_enabled()` |

Added in 1.1.8

Returns true if the sectors information is enabled.


---
### <a name="setSectorsInfoEnabled"></a> `setSectorsInfoEnabled`

| C++    | `void setSectorsInfoEnabled(bool sectorsInfo)`    |
| :--    | :--                                               |
| Python | `void set_sectors_info_enabled(bool sectorsInfo)` |

Added in 1.1.8

Enables information about all sectors (map layout) present in current episode/level, it will be available in the state.

Default value: false

Config key: `sectorsInfoEnabled/sectors_info_enabled`

See also:
- [`Types: GameState`](Types.md#gamestate)
- [`Types: Sector`](Types.md#sector)
- [examples/python/objects_and_sectors.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/objects_and_sectors.py)


---
### <a name="isAudioBufferEnabled"></a> `isAudioBufferEnabled`

| C++    | `bool isAudioBufferEnabled()`    |
| :--    | :--                              |
| Python | `bool is_audio_buffer_enabled()` |

Added in 1.1.9

Returns true if the audio buffer is enabled.


---
### <a name="setAudioBufferEnabled"></a> `setSectorsInfoEnabled`

| C++    | `void setAudioBufferEnabled(bool audioBuffer)`    |
| :--    | :--                                               |
| Python | `void set_audio_buffer_enabled(bool audioBuffer)` |

Added in 1.1.9

Returns true if the audio buffer is enabled.

Default value: false

Config key: `audioBufferEnabled/audio_buffer_enabled`

See also:
- [`Types: GameState`](Types.md#gamestate)
- [`Types: SamplingRate`](Types.md#sampling-rate)
- [examples/python/audio_buffer.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### <a name="getAudioSampliongRate"></a> `getAudioSamplingRate`

| C++    | `SamplingRate getAudioSamplingRate()`    |
| :--    | :--                                      |
| Python | `SamplingRate get_audio_sampling_rate()` |

Added in 1.1.9

Returns the sampling rate of audio buffer.


See also:
- [`Types: GameState`](Types.md#gamestate)
- [`Types: SamplingRate`](Types.md#sampling-rate)
- [examples/python/audio_buffer.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### <a name="setAudioSamplingRate"></a> `setAudioSamplingRate`

| C++    | `void setAudioSamplingRate(SamplingRate samplingRate)`    |
| :--    | :--                                                       |
| Python | `void set_audio_sampling_rate(SamplingRate samplingRate)` |

Added in 1.1.9

Sets the sampling rate of audio buffer.

Default value: false

Config key: `audioSamplingRate/audio_samping_rate`

See also:
- [`Types: GameState`](Types.md#gamestate)
- [`Types: SamplingRate`](Types.md#sampling-rate)
- [examples/python/audio_buffer.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### <a name="getAudioBufferSize"></a> `getAudioBufferSize`

| C++    | `int getAudioBufferSize()`    |
| :--    | :--                           |
| Python | `int get_audio_buffer_size()` |

Added in 1.1.9

Returns the size of audio buffer.


See also:
- [`Types: GameState`](Types.md#gamestate)
- [examples/python/audio_buffer.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### <a name="setAudioBufferSize"></a> `setAudioBufferSize`

| C++    | `void setAudioBufferSize(int size)`    |
| :--    | :--                                    |
| Python | `void set_audio_buffer_size(int size)` |

Added in 1.1.9

Sets the size of audio buffer. Size is defined in number of logic tics. 
After each action audio buffer will contain audio from specified number of last processed tics.
Doom uses 35 ticks per second.

Default value: 4

Config key: `audioBufferSize/audio_buffer_size`

See also:
- [`Types: GameState`](Types.md#gamestate)
- [examples/python/audio_buffer.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/audio_buffer.py)
