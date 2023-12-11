# DoomGame

DoomGame is the main object of the ViZDoom library, representing a single instance of the Doom game and providing the interface for a single agent/player to interact with the game. The object allows sending actions to the game, getting the game state, etc. The declarations of this class and its methods can be found in the `include/ViZDoomGame.h` header file.

Here we document all the methods of the DoomGame class and their corresponding Python bindings implemented as pybind11 module.


## Flow control methods

### `init`

| C++    | `bool init()`    |
| :--    | :--              |
| Python | `init() -> bool` |

Initializes ViZDoom game instance and starts a new episode.
After calling this method, the first state from a new episode will be available.
Some configuration options cannot be changed after calling this method.
Init returns true when the game was started properly and false otherwise.


---
### `close`

| C++    | `void close()`    |
| :--    | :--               |
| Python | `close() -> None` |

Closes ViZDoom game instance.
It is automatically invoked by the destructor.
The game can be initialized again after being closed.


---
### `newEpisode`

| C++    | `void newEpisode(std::string recordingFilePath = "")` |
| :--    | :--                                                   |
| Python | `new_episode(recordingFilePath: str = "") -> None`    |

Changed in 1.1.0

Initializes a new episode. The state of an environment is completely restarted (all variables and rewards are reset to their initial values).
After calling this method, the first state from the new episode will be available.
If the `recordingFilePath` is not empty, the new episode will be recorded to this file (as a Doom lump).

In a multiplayer game, the host can call this method to finish the game.
Then the rest of the players must also call this method to start a new episode.


---
### `replayEpisode`

| C++    | `void replayEpisode(std::string filePath, unsigned int player = 0)` |
| :--    | :--                                                                 |
| Python | `replay_episode(filePath: str, player: int = 0) -> None`            |

Added in 1.1.0

Replays the recorded episode from the given file using the perspective of the specified player.
Players are numbered from 1, If `player` is equal to 0, the episode will be replayed using the perspective of the default player in the recording file.
After calling this method, the first state from the replay will be available.
All rewards, variables, and states are available when replaying the episode.

See also:
- [examples/python/record_episodes.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/record_episodes.py)
- [examples/python/record_multiplayer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/record_multiplayer.py)


---
### `isRunning`

| C++    | `bool isRunning()`     |
| :--    | :--                    |
| Python | `is_running() -> bool` |

Checks if the controlled game instance is running.


---
### `isMultiplayerGame`

| C++    | `bool isMultiplayerGame()`      |
| :--    | :--                             |
| Python | `is_multiplayer_game() -> bool` |

Added in 1.1.2

Checks if the game is in multiplayer mode.


---
### `isRecordingEpisode`

| C++    | `bool isRecordingEpisode()`      |
| :--    | :--                              |
| Python | `is_recording_episode() -> bool` |

Added in 1.1.5

Checks if the game is in recording mode.


---
### `isReplayingEpisode`

| C++    | `bool isReplayingEpisode()`      |
| :--    | :--                              |
| Python | `is_replaying_episode() -> bool` |

Added in 1.1.5

Checks if the game is in replay mode.


---
### `setAction`

| C++    | `void setAction(std::vector<double> const &actions)`          |
| :--    | :--                                                           |
| Python | `set_action(actions: list | tuple | ndarray [float]) -> None` |

Sets the player's action for the next tics.
Each value corresponds to a button previosuly specified with [`addAvailableButton`](#addavailablebutton), or [`setAvailableButtons`](#setavailablebuttons) methods,
or in the configuration file (in order of appearance).


---
### `advanceAction`

| C++    | `void advanceAction(unsigned int tics = 1, bool updateState = true)` |
| :--    | :--                                                                  |
| Python | `advance_action(tics: int = 1, updateState: bool = True) -> None`    |

Processes the specified number of tics. If `updateState` is set,
the state will be updated after the last processed tic and a new reward will be calculated.
To get the new state, use `getState` and to get the new reward use `getLastReward`.
If `updateState` is not set, the state will not be updated.


---
### `makeAction`

| C++    | `double makeAction(std::vector<double> const &actions, unsigned int tics = 1)` |
| :--    | :--                                                                            |
| Python | `make_action(actions: list | tuple | ndarray [float], tics: int = 1) -> float` |

This method combines functionality of [`setAction`](#setaction), [`advanceAction`](#advanceaction) and [`getLastReward`](#getlastreward).
Sets the player's action for the next tics, processes the specified number of tics,
updates the state and calculates a new reward, which is returned.


---
### `isNewEpisode`

| C++    | `bool isNewEpisode()`      |
| :--    | :--                        |
| Python | `is_new_episode() -> bool` |

Returns true if the current episode is in the initial state - the first state, no actions were performed yet.


---
### `isEpisodeFinished`

| C++    | `bool isEpisodeFinished()`      |
| :--    | :--                             |
| Python | `is_episode_finished() -> bool` |

Returns true if the current episode is in the terminal state (is finished).
[`makeAction`](#makeaction) and [`advanceAction`](#advanceaction) methods will take no effect after this point (unless [`newEpisode`](#newepisode) method is called).


---
### `isPlayerDead`

| C++    | `bool isPlayerDead()`      |
| :--    | :--                        |
| Python | `is_player_dead() -> bool` |

Returns true if the player is dead.
In singleplayer, the player's death is equivalent to the end of the episode.
In multiplayer, when the player is dead [`respawnPlayer`](#respawnplayer) method can be called.


---
### `respawnPlayer`

| C++    | `void respawnPlayer()`     |
| :--    | :--                        |
| Python | `respawn_player() -> None` |

This method respawns the player after death in multiplayer mode.
After calling this method, the first state after the respawn will be available.

See also:
- [`isMultiplayerGame`](#ismultiplayergame)


---
### `sendGameCommand`

| C++    | `void sendGameCommand(std::string cmd)` |
| :--    | :--                                     |
| Python | `send_game_command(cmd: str) -> None`   |

Sends the command to Doom console. It can be used for controlling the game, changing settings, cheats, etc.
Some commands will be blocked in some modes.

See also:
- [ZDoom Wiki: Console](http://zdoom.org/wiki/Console)
- [ZDoom Wiki: CVARs (console variables)](https://zdoom.org/wiki/CVARs)
- [ZDoom Wiki: CCMD (console commands)](https://zdoom.org/wiki/CCMDs)


---
### `getState`

| C++    | `GameStatePtr (std::shared_ptr<GameState>) GameState getState()` |
| :--    | :--                                                              |
| Python | `get_state() -> GameState`                                       |

Changed in 1.1.0

Returns [`GameState`](./gameState.md#gamestate) object with the current game state.
If the current episode is finished, `nullptr/null/None` will be returned.

See also:
- [`GameState`](./gameState.md#gamestate)


---
### `getServerState`

| C++    | `ServerStatePtr (std::shared_ptr<ServerState>) ServerState getServerState()` |
| :--    | :--                                                                          |
| Python | `get_state_state() -> ServerState`                                           |

Added in 1.1.6

Returns [`ServerState`](./gameState.md#serverstate) object with the current server state.

See also:
- [`ServerState`](./gameState.md#serverstate)


---
### `getLastAction`

| C++    | `std::vector<double> getLastAction()` |
| :--    | :--                                   |
| Python | `get_last_action() -> list`           |

Returns the last action performed.
Each value corresponds to a button added with `[addAvailableButton](#addAvailableButton)` (in order of appearance).
Most useful in `SPECTATOR` mode.


---
### `getEpisodeTime`

| C++    | `unsigned int getEpisodeTime()` |
| :--    | :--                             |
| Python | `get_episode_time() -> int`        |

Returns number of current episode tic.


---
### `save`

| C++    | `void save(std::string filePath)` |
| :--    | :--                               |
| Python | `save(filePath: str) -> None`     |

Added in 1.1.9

Saves a game's internal state to the file using ZDoom's save game functionality.


---
### `load`

| C++    | `void load(std::string filePath)` |
| :--    | :--                               |
| Python | `load(filePath: str) -> None`     |

Added in 1.1.9

Loads a game's internal state from the file using ZDoom's load game functionality.
A new state is available after loading.
Loading the game state does not reset the current episode state,
tic counter/time and total reward state keep their values.


## Buttons settings methods

### `getAvailableButtons`

| C++    | `std::vector<Button> getAvailableButtons()` |
| :--    | :--                                         |
| Python | `get_available_buttons() -> list[Button]`   |

Returns the list of available `Buttons`.

See also:
- [`Enums: Button`](./enums.md#button)
- [`addAvailableButton`](#addavailablebutton)
- [`setAvailableButtons`](#addavailablebuttons)


---
### `setAvailableButtons`

| C++    | `void setAvailableButtons(std::vector<Button> buttons)`       |
| :--    | :--                                                           |
| Python | `add_available_buttons(buttons: list | tuple[Button]) -> None` |

Sets given list of `Button`s (e.g. `TURN_LEFT`, `MOVE_FORWARD`) as available `Buttons`.

Has no effect when the game is running.

Config key: `availableButtons/available_buttons` (list)

See also:
- [`Enums: Button`](./enums.md#button)
- [`ConfigFile: List`](./configurationFiles.md#list)
- [`addAvailableButton`](#addavailablebutton)


---
### `addAvailableButton`

| C++    | `void addAvailableButton(Button button, double maxValue = 0)`       |
| :--    | :--                                                                 |
| Python | `add_available_button(button: Button, maxValue: float = 0) -> None` |

Adds [`Button`](./enums.md#button) type (e.g. `TURN_LEFT`, `MOVE_FORWARD`) to available `Buttons` and sets the maximum allowed, absolute value for the specified button.
If the given button has already been added, it will not be added again, but the maximum value is overridden.

Has no effect when the game is running.

Config key: `availableButtons/available_buttons` (list)

See also:
- [`Enums: Button`](./enums.md#button)
- [`ConfigFile: List`](./configurationFiles.md#list)
- [`setAvailableButtons`](#addavailablebuttons)
- [`setButtonMaxValue`](#setbuttonmaxvalue)


---
### `clearAvailableButtons`

| C++    | `void clearAvailableButtons()`      |
| :--    | :--                                 |
| Python | `clear_available_buttons() -> None` |

Clears all available `Buttons` added so far.

Has no effect when the game is running.

See also:
- [`Enums: Button`](./enums.md#button)


---
### `getAvailableButtonsSize`

| C++    | `int getAvailableButtonsSize()`       |
| :--    | :--                                   |
| Python | `get_available_buttons_size() -> int` |

Returns the number of available `Buttons`.

See also:
- [`Enums: Button`](./enums.md#button)


---
### `setButtonMaxValue`

| C++    | `void setButtonMaxValue(Button button, double maxValue = 0)`        |
| :--    | :--                                                                 |
| Python | `set_button_max_value(button: Button, maxValue: float = 0) -> None` |

Sets the maximum allowed absolute value for the specified button.
Setting the maximum value to 0 results in no constraint at all (infinity).
This method makes sense only for delta buttons.
The constraints limit applies in all Modes.

Has no effect when the game is running.

See also:
- [`Enums: Button`](./enums.md#button)


---
### `getButtonMaxValue`

| C++    | `unsigned int getButtonMaxValue(Button button)` |
| :--    | :--                                             |
| Python | `set_button_max_value(button: Button) -> int`   |

Returns the maximum allowed absolute value for the specified button.

See also:
- [`Enums: Button`](./enums.md#button)


---
### `getButton`

| C++    | `double getButton(Button button)`     |
| :--    | :--                                   |
| Python | `set_button(button: Button) -> float` |

Returns the current state of the specified button (`ATTACK`, `USE` etc.).

See also:
- [`Enums: Button`](./enums.md#button)


## GameVariables methods


### `getAvailableGameVariables`

| C++    | `std::vector<GameVariable> getAvailableGameVariables()` |
| :--    | :--                                                     |
| Python | `get_available_game_variables() -> list[GameVariables]` |

Returns the list of available `GameVariables`.

See also:
- [`Enums: GameVariable`](./enums.md#gamevariable)
- [`addAvailableGameVariable`](#addavailablegamevariable)
- [`setAvailableGameVariables`](#setavailablegamevariables)


---
### `setAvailableGameVariables`

| C++    | `void setAvailableGameVariables(std::vector<GameVariable> variables)`          |
| :--    | :--                                                                            |
| Python | `set_available_game_variables(variables: list | tuple[GameVariables]) -> None` |

Sets list of [`GameVariable`](./enums.md#gamevariable) as available `GameVariables` in the [`GameState`](./gameState.md#gamestate) returned by [`getState`](#getstate) method.

Has no effect when the game is running.

Config key: `availableGameVariables/available_game_variables` (list)

See also:
- [`Enums: GameVariable`](./enums.md#gamevariable)
- [`ConfigFile: List`](./configurationFiles.md#list)
- [`addAvailableGameVariable`](#addavailablegamevariable)


---
### `addAvailableGameVariable`

| C++    | `void addAvailableGameVariable(GameVariable variable)`        |
| :--    | :--                                                           |
| Python | `add_available_game_variable(variable: GameVariable) -> None` |

Adds the specified [`GameVariable`](./enums.md#gamevariable) to the list of available game variables (e.g. `HEALTH`, `AMMO1`, `ATTACK_READY`) in the [`GameState`](./gameState.md#gamestate) returned by [`getState`](#getstate) method.

Has no effect when the game is running.

Config key: `availableGameVariables/available_game_variables` (list)

See also:
- [`Enums: GameVariable`](./enums.md#gamevariable)
- [`ConfigFile: List`](./configurationFiles.md#list)
- [`setAvailableGameVariables`](#setavailablegamevariables)


---
### `clearAvailableGameVariables`

| C++    | `void clearAvailableGameVariables()`       |
| :--    | :--                                        |
| Python | `clear_available_game_variables() -> None` |

Clears the list of available `GameVariables` that are included in the [`GameState`](./gameState.md#gamestate) returned by [`getState`](#getstate) method.

Has no effect when the game is running.

See also:
- [`Enums: GameVariable`](./enums.md#gamevariable)
- [`ConfigFile: List`](./configurationFiles.md#list)


---
### `getAvailableGameVariablesSize`

| C++    | `unsigned int getAvailableGameVariablesSize()` |
| :--    | :--                                            |
| Python | `get_available_game_variables_size() -> int`   |

Returns the number of available `GameVariables`.

See also:
- [`Enums: GameVariable`](./enums.md#gamevariable)
- [`ConfigFile: List`](./configurationFiles.md#list)


---
### `getGameVariable`

| C++    | `double getGameVariable(GameVariable variable)`      |
| :--    | :--                                                  |
| Python | `get_game_variable(variable: GameVariable) -> float` |

Returns the current value of the specified game variable (`HEALTH`, `AMMO1` etc.).
The specified game variable does not need to be among available game variables (included in the state).
It could be used for e.g. shaping. Returns 0 in case of not finding given `GameVariable`.

See also:
- [`Enums: GameVariable`](./enums.md#gamevariable)


## Game arguments methods


### `setGameArgs`

| C++    | `void setGameArgs(std::string args)` |
| :--    | :--                                  |
| Python | `set_game_args(args: str) -> None`   |

Added in 1.3.0

Sets custom arguments that will be passed to ViZDoom process during initialization.
It is useful for changing additional game settings.
Use with caution, as in rare cases it may prevent the library from working properly.
Using this method is equivalent to first calling [`clearGameArgs`](#cleargameargs) and then [`addGameArgs`](#addgameargs).

Config key: `gameArgs/game_args`

See also:
- [ZDoom Wiki: Command line parameters](http://zdoom.org/wiki/Command_line_parameters)
- [ZDoom Wiki: CVARs (Console Variables)](http://zdoom.org/wiki/CVARS)


---
### `addGameArgs`

| C++    | `void addGameArgs(std::string args)` |
| :--    | :--                                  |
| Python | `add_game_args(args: str) -> None`   |

Adds custom arguments that will be passed to ViZDoom process during initialization.
It is useful for changing additional game settings.
Use with caution, as in rare cases it may prevent the library from working properly.

Config key: `gameArgs/game_args`

See also:
- [ZDoom Wiki: Command line parameters](http://zdoom.org/wiki/Command_line_parameters)
- [ZDoom Wiki: CVARs (Console Variables)](http://zdoom.org/wiki/CVARS)


---
### `clearGameArgs`

| C++    | `void clearGameArgs()`      |
| :--    | :--                         |
| Python | `clear_game_args() -> None` |

Clears all arguments previously added with [`setGameArgs`](#setgameargs) or/and [`addGameArgs`](#addgameargs) methods.


### `getGameArgs`

| C++    | `std::string getGameArgs()` |
| :--    | :--                         |
| Python | `get_game_args() -> str`    |

Returns the additional arguments for ViZDoom process set with [`setGameArgs`](#setgameargs) or/and [`addGameArgs`](#addgameargs) methods.


## Reward methods


### `getLivingReward`

| C++    | `double getLivingReward()`     |
| :--    | :--                            |
| Python | `get_living_reward() -> float` |

Returns the reward granted to the player after every tic.


---
### `setLivingReward`

| C++    | `void setLivingReward(double livingReward)`      |
| :--    | :--                                              |
| Python | `set_living_reward(livingReward: float) -> None` |

Sets the reward granted to the player after every tic. A negative value is also allowed.

Default value: 0

Config key: `livingReward/living_reward`


---
### `getDeathPenalty`

| C++    | `double getDeathPenalty()`     |
| :--    | :--                            |
| Python | `get_death_penalty() -> float` |

Returns the penalty for the player's death.


---
### `setDeathPenalty`

| C++    | `void setDeathPenalty(double deathPenalty)`      |
| :--    | :--                                              |
| Python | `set_death_penalty(deathPenalty: float) -> None` |

Sets a penalty for the player's death. Note that in case of a negative value, the player will be rewarded upon dying.

Default value: 0

Config key: `deathPenalty/death_penalty`


---
### `getLastReward`

| C++    | `double getLastReward()`     |
| :--    | :--                          |
| Python | `get_last_reward() -> float` |

Returns a reward granted after the last update of state.


---
### `getTotalReward`

| C++    | `double getTotalReward()`     |
| :--    | :--                           |
| Python | `get_total_reward() -> float` |

Returns the sum of all rewards gathered in the current episode.


## General game setting methods


### `loadConfig`

| C++    | `bool loadConfig(std::string filePath)` |
| :--    | :--                                     |
| Python | `load_config(filePath: str) -> bool`    |

Loads configuration (resolution, available buttons, game variables etc.) from a configuration file.
In case of multiple invocations, older configurations will be overwritten by the recent ones.
Overwriting does not involve resetting to default values. Thus only overlapping parameters will be changed.
The method returns true if the whole configuration file was correctly read and applied,
false if the file contained errors.

If the file relative path is given, it will be searched for in the following order: current directory, current directory + `/scenarios/`, ViZDoom's installation directory + `/scenarios/`.

See also:
- [ConfigFile](./configurationFiles.md)


---
### `getMode`

| C++    | `Mode getMode()`     |
| :--    | :--                  |
| Python | `get_mode() -> Mode` |

Returns the current mode (`PLAYER`, `SPECTATOR`, `ASYNC_PLAYER`, `ASYNC_SPECTATOR`).

See also:
- [`Enums: Mode`](./enums.md#mode)


---
### `setMode`

| C++    | `void setMode(Mode mode)`      |
| :--    | :--                            |
| Python | `set_mode(mode: Mode) -> None` |

Sets the mode (`PLAYER`, `SPECTATOR`, `ASYNC_PLAYER`, `ASYNC_SPECTATOR`) in which the game will be running.

Default value: `PLAYER`.

Has no effect when the game is running.

Config key: `mode`

See also:
- [`Enums: Mode`](./enums.md#mode)


---
### `getTicrate`

| C++    | `unsigned int getTicrate()` |
| :--    | :--                         |
| Python | `get_ticrate() -> int`      |

Added in 1.1.0

Returns current ticrate.


---
### `setTicrate`

| C++    | `void setTicrate(unsigned int ticrate)` |
| :--    | :--                                     |
| Python | `set_ticrate(ticrate: int) -> None`     |

Added in 1.1.0

Sets the ticrate for ASNYC Modes - number of logic tics executed per second.
The default Doom ticrate is 35. This value will play a game at normal speed.

Default value: 35 (default Doom ticrate).

Has no effect when the game is running.

Config key: `ticrate`

See also:
- [examples/python/ticrate.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/ticrate.py)


---
### `setViZDoomPath`

| C++    | `void setViZDoomPath(std::string filePath)` |
| :--    | :--                                         |
| Python | `set_vizdoom_path(filePath: str) -> None`       |

Sets the path to the ViZDoom engine executable vizdoom.

Default value: "{vizdoom.so location}/{vizdoom or vizdoom.exe (on Windows)}".

Config key: `ViZDoomPath/vizdoom_path


---
### `setDoomGamePath`

| C++    | `void setDoomGamePath(std::string filePath)` |
| :--    | :--                                          |
| Python | `set_doom_game_path(filePath: str) -> None`      |

Sets the path to the Doom engine based game file (wad format).
If not used DoomGame will look for doom2.wad and freedoom2.wad (in that order) in the directory of ViZDoom's installation (where vizdoom.so/pyd is).

Default value: "{vizdoom.so location}/{doom2.wad, doom.wad, freedoom2.wad or freedoom.wad}"

Config key: `DoomGamePath/doom_game_path`


---
### `setDoomScenarioPath`

| C++    | `void setDoomScenarioPath(std::string filePath)` |
| :--    | :--                                              |
| Python | `set_doom_scenario_path(filePath: str) -> None`      |

Sets the path to an additional scenario file (wad format).
If not provided, the default Doom single-player maps will be loaded.

Default value: ""

Config key: `DoomScenarioPath/set_doom_scenario_path`


---
### `setDoomMap`

| C++    | `void setDoomMap(std::string map)` |
| :--    | :--                                |
| Python | `set_doom_map(map: str) -> None`   |

Sets the map name to be used.

Default value: "map01", if set to empty "map01" will be used.

Config key: `DoomMap/doom_map`


---
### `setDoomSkill`

| C++    | `void setDoomSkill(unsigned int skill)` |
| :--    | :--                                     |
| Python | `set_doom_skill(skill: int) -> None`    |

Sets Doom game difficulty level, which is called skill in Doom.
The higher the skill, the harder the game becomes.
Skill level affects monsters' aggressiveness, monsters' speed, weapon damage, ammunition quantities, etc.
Takes effect from the next episode.

- 1 - VERY EASY, “I'm Too Young to Die” in Doom.
- 2 - EASY, “Hey, Not Too Rough" in Doom.
- 3 - NORMAL, “Hurt Me Plenty” in Doom.
- 4 - HARD, “Ultra-Violence” in Doom.
- 5 - VERY HARD, “Nightmare!” in Doom.

Default value: 3

Config key: `DoomSkill/doom_skill`


---
### `setDoomConfigPath`

| C++    | `void setDoomConfigPath(std::string filePath)` |
| :--    | :--                                            |
| Python | `set_doom_config_path(filePath: str) -> None`  |

Sets the path for ZDoom's configuration file.
The file is responsible for the configuration of the ZDoom engine itself.
If it does not exist, it will be created after the `vizdoom` executable is run.
This method is not needed for most of the tasks and is added for the convenience of users with hacking tendencies.

Default value: "", if left empty "_vizdoom.ini" will be used.

Config key: `DoomConfigPath/doom_config_path`


---
### `getSeed`

| C++    | `unsigned int getSeed()` |
| :--    | :--                      |
| Python | `getSeed() -> int`       |

Returns ViZDoom's seed.


---
### `setSeed`

| C++    | `void setSeed(unsigned int seed)` |
| :--    | :--                               |
| Python | `set_seed(seed: int) -> None`     |

Sets the seed of ViZDoom's RNG that generates seeds (initial state) for episodes.

Default value: randomized in constructor

Config key: `seed`

See also:
- [examples/python/seed.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/seed.py)



---
### `getEpisodeStartTime`

| C++    | `unsigned int getEpisodeStartTime()` |
| :--    | :--                                  |
| Python | `get_episode_start_time() -> int`    |

Returns the start time (delay) of every episode in tics.


---
### `setEpisodeStartTime`

| C++    | `void setEpisodeStartTime(unsigned int tics)` |
| :--    | :--                                           |
| Python | `set_episode_start_time(tics: int) -> None`   |

Sets the start time (delay) of every episode in tics.
Every episode will effectively start (from the user's perspective) after the provided number of tics.

Default value: 1

Config key: `episodeStartTime/episode_start_time`


---
### `getEpisodeTimeout`

| C++    | `unsigned int getEpisodeTimeout()` |
| :--    | :--                                |
| Python | `get_episode_timeout() -> int`     |

Returns the number of tics after which the episode will be finished.


---
### `setEpisodeTimeout`

| C++    | `void setEpisodeTimeout(unsigned int tics)` |
| :--    | :--                                         |
| Python | `set_episode_timeout(tics: int) -> None`    |

Sets the number of tics after which the episode will be finished. 0 will result in no timeout.

Default value: 0

Config key: `episodeTimeout/episode_timeout`


## Output/rendering setting methods


### `setScreenResolution`

| C++    | `void setScreenResolution(ScreenResolution resolution)`       |
| :--    | :--                                                           |
| Python | `set_screen_resolution(resolution: ScreenResolution) -> None` |

Sets the screen resolution. ZDoom engine supports only specific resolutions.
Supported resolutions are part of ScreenResolution enumeration (e.g., `RES_320X240`, `RES_640X480`, `RES_1920X1080`).
The buffers, as well as the content of ViZDoom's display window, will be affected.

Default value: `RES_320X240`

Has no effect when the game is running.

Config key: `screenResolution/screen_resolution`


See also:
- [`Enums: ScreenResolution`](./enums.md#screenresolution)


---
### `getScreenFormat`

| C++    | `ScreenFormat getScreenFormat()`      |
| :--    | :--                                   |
| Python | `get_screen_format() -> ScreenFormat` |

Returns the format of the screen buffer and the automap buffer.


---
### `setScreenFormat`

| C++    | `void setScreenFormat(ScreenFormat format)`       |
| :--    | :--                                               |
| Python | `set_screen_format(format: ScreenFormat) -> None` |

Sets the format of the screen buffer and the automap buffer.
Supported formats are defined in `ScreenFormat` enumeration type (e.g. `CRCGCB`, `RGB24`, `GRAY8`).
The format change affects only the buffers, so it will not have any effect on the content of ViZDoom's display window.

Default value: `CRCGCB`

Has no effect when the game is running.

Config key: `screenFormat/screen_format`

See also:
- [`Enums: ScreenFormat`](./enums.md#screenformat)


---
### `isDepthBufferEnabled`

| C++    | `bool isDepthBufferEnabled()`       |
| :--    | :--                                 |
| Python | `is_depth_buffer_enabled() -> None` |

Added in 1.1.0

Returns true if the depth buffer is enabled.


---
### `setDepthBufferEnabled`

| C++    | `void setDepthBufferEnabled(bool depthBuffer)`        |
| :--    | :--                                                   |
| Python | `set_depth_buffer_enabled(depthBuffer: bool) -> None` |

Added in 1.1.0

Enables rendering of the depth buffer, it will be available in the state.
Depth buffer will contain noise if `viz_nocheat` is enabled.

Default value: false

Has no effect when the game is running.

Config key: `depthBufferEnabled/depth_buffer_enabled`

See also:
- [`GameState`](./gameState.md#gamestate)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py)


---
### `isLabelsBufferEnabled`

| C++    | `bool isLabelsBufferEnabled()`       |
| :--    | :--                                  |
| Python | `is_labels_buffer_enabled() -> None` |

Added in 1.1.0

Returns true if the labels buffer is enabled.


---
### `setLabelsBufferEnabled`

| C++    | `void setLabelsBufferEnabled(bool labelsBuffer)`       |
| :--    | :--                                                    |
| Python | `set_labels_buffer_enabled(bool labelsBuffer) -> None` |

Added in 1.1.0

Enables rendering of the labels buffer, it will be available in the state with the vector of `Label`s.
LabelsBuffer will contain noise if `viz_nocheat` is enabled.

Default value: false

Has no effect when the game is running.

Config key: `labelsBufferEnabled/labels_buffer_enabled`

See also:
- [`GameState: Label`](./gameState.md#label)
- [`GameState`](./gameState.md#gamestate)
- [examples/python/labels.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/labels.py)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py)


---
### `isAutomapBufferEnabled`

| C++    | `bool isAutomapBufferEnabled()`       |
| :--    | :--                                   |
| Python | `is_automap_buffer_enabled() -> bool` |

Added in 1.1.0

Returns true if the automap buffer is enabled.


---
### `setAutomapBufferEnabled`

| C++    | `void setAutomapBufferEnabled(bool automapBuffer)`        |
| :--    | :--                                                       |
| Python | `set_automap_buffer_enabled(automapBuffer: bool) -> None` |

Added in 1.1.0

Enables rendering of the automap buffer, it will be available in the state.

Default value: false

Has no effect when the game is running.

Config key: `automapBufferEnabled/automap_buffer_enabled`

See also:
- [`GameState`](./gameState.md#gamestate)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py),


---
### `setAutomapMode`

| C++    | `void setAutomapMode(AutomapMode mode)`       |
| :--    | :--                                           |
| Python | `set_automap_mode(mode: AutomapMode) -> None` |

Added in 1.1.0

Sets the automap mode (`NORMAL`, `WHOLE`, `OBJECTS`, `OBJECTS_WITH_SIZE`),
which determines what will be visible on it.

Default value: `NORMAL`

Config key: `automapMode/set_automap_mode`

See also:
- [`Enums: AutomapMode`](./enums.md#automapmode)


---
### `setAutomapRotate`

| C++    | `void setAutomapRotate(bool rotate)`       |
| :--    | :--                                        |
| Python | `set_automap_rotate(rotate: bool) -> None` |

Added in 1.1.0

Determine if the automap will be rotating with the player.
If false, north always will be at the top of the buffer.

Default value: false

Config key: `automapRotate/automap_rotate`


---
### `setAutomapRenderTextures`

| C++    | `void setAutomapRenderTextures(bool textures)`        |
| :--    | :--                                                   |
| Python | `set_automap_render_textures(textures: bool) -> None` |

Added in 1.1.0

Determine if the automap will be textured, showing the floor textures.

Default value: true

Config key: `automapRenderTextures/automap_render_textures`


---
### `setRenderHud`

| C++    | `void setRenderHud(bool hud)`       |
| :--    | :--                                 |
| Python | `set_render_hud(hud: bool) -> None` |

Determine if the hud will be rendered in the game.

Default value: false

Config key: `renderHud/render_hud`


---
### `setRenderMinimalHud`

| C++    | `void setRenderMinimalHud(bool minHud)`        |
| :--    | :--                                            |
| Python | `set_render_minimal_hud(minHud: bool) -> None` |

Added in 1.1.0

Determine if the minimalistic version of the hud will be rendered instead of the full hud.

Default value: false

Config key: `renderMinimalHud/render_minimal_hud`


---
### `setRenderWeapon`

| C++    | `void setRenderWeapon(bool weapon)`       |
| :--    | :--                                       |
| Python | `set_render_weapon(weapon: bool) -> None` |

Determine if the weapon held by the player will be rendered in the game.

Default value: true

Config key: `renderWeapon/render_weapon`


---
### `setRenderCrosshair`

| C++    | `void setRenderCrosshair(bool crosshair)`       |
| :--    | :--                                             |
| Python | `set_render_crosshair(crosshair: bool) -> None` |

Determine if the crosshair will be rendered in the game.

Default value: false

Config key: `renderCrosshair/render_crosshair`


---
### `setRenderDecals`

| C++    | `void setRenderDecals(bool decals)`       |
| :--    | :--                                       |
| Python | `set_render_decals(decals: bool) -> None` |

Determine if the decals (marks on the walls) will be rendered in the game.

Default value: true

Config key: `renderDecals/render_decals`


---
### `setRenderParticles`

| C++    | `void setRenderParticles(bool particles)`       |
| :--    | :--                                             |
| Python | `set_render_particles(particles: bool) -> None` |

Determine if the particles will be rendered in the game.

Default value: true

Config key: `renderParticles/render_particles`


---
### `setRenderEffectsSprites`

| C++    | `void setRenderEffectsSprites(bool sprites)`        |
| :--    | :--                                                 |
| Python | `set_render_effects_sprites(sprites: bool) -> None` |

Added in 1.1.0

Determine if some effects sprites (gun puffs, blood splats etc.) will be rendered in the game.

Default value: true

Config key: `renderEffectsSprites/render_effects_sprites`


---
### `setRenderMessages`

| C++    | `void setRenderMessages(bool messages)`       |
| :--    | :--                                           |
| Python | `set_render_messages(messages: bool) -> None` |

Added in 1.1.0

Determine if in-game messages (information about pickups, kills, etc.) will be rendered in the game.

Default value: false

Config key: `renderMessages/render_messages`


---
### `setRenderCorpses`

| C++    | `void setRenderCorpses(bool corpses)`        |
| :--    | :--                                          |
| Python | `set_render_corpsess(corpses: bool) -> None` |

Added in 1.1.0

Determine if actors' corpses will be rendered in the game.

Default value: true

Config key: `renderCorpses/render_corpses`


---
### `setRenderScreenFlashes`

| C++    | `void setRenderScreenFlashes(bool flashes)`        |
| :--    | :--                                                |
| Python | `set_render_screen_flashes(flashes: bool) -> None` |

Added in 1.1.3

Determine if the screen flash effect upon taking damage or picking up items will be rendered in the game.

Default value: true

Config key: `renderScreenFlashes/render_screen_flashes`


---
### `setRenderAllFrames`

| C++    | `void setRenderAllFrames(bool allFrames)`         |
| :--    | :--                                               |
| Python | `set_render_all_frames(all_frames: bool) -> None` |

Added in 1.1.3

Determine if all frames between states will be rendered (when skip greater than 1 is used).
Allows smooth preview but can reduce performance.
It only makes sense to use it if the window is visible.

Default value: false

Config key: `renderAllFrames/render_all_frames`

See also:
- [`setWindowVisible`](#setwindowvisible)


---
### `setWindowVisible`

| C++    | `void setWindowVisible(bool visibility)`       |
| :--    | :--                                            |
| Python | `set_window_visible(visibility: bool) -> None` |

Determines if ViZDoom's window will be visible.
ViZDoom with window disabled can be used on Linux systems without X Server.

Default value: false

Has no effect when the game is running.

Config key: `windowVisible/window_visible`


---
### `setConsoleEnabled`

| C++    | `void setConsoleEnabled(bool console)`       |
| :--    | :--                                          |
| Python | `set_console_enabled(console: bool) -> None` |

Determines if ViZDoom's console output will be enabled.

Default value: false

Config key: `consoleEnabled/console_enabled`


---
### `setSoundEnabled`

| C++    | `void setSoundEnabled(bool sound)`       |
| :--    | :--                                      |
| Python | `set_sound_enabled(sound: bool) -> None` |

Determines if ViZDoom's sound will be played.

Default value: false

Config key: `soundEnabled/sound_enabled`


---
### `getScreenWidth`

| C++    | `int getScreenWidth()`      |
| :--    | :--                         |
| Python | `get_screen_width() -> int` |

Returns game's screen width - width of all buffers.


---
### `getScreenHeight`

| C++    | `int getScreenHeight()`      |
| :--    | :--                          |
| Python | `get_screen_height() -> int` |

Returns game's screen height - height of all buffers.


---
### `getScreenChannels`

| C++    | `int getScreenChannels()`      |
| :--    | :--                            |
| Python | `get_screen_channels() -> int` |

Returns number of channels in screen buffer and map buffer (depth and labels buffer always have one channel).


---
### `getScreenPitch`

| C++    | `size_t getScreenPitch()`   |
| :--    | :--                         |
| Python | `get_screen_pitch() -> int` |

Returns size in bytes of one row in screen buffer and map buffer.


---
### `getScreenSize`

| C++    | `size_t getScreenSize()`   |
| :--    | :--                        |
| Python | `get_screen_size() -> int` |

Returns size in bytes of screen buffer and map buffer.


---
### `isObjectsInfoEnabled`

| C++    | `bool isObjectInfoEnabled()`       |
| :--    | :--                                |
| Python | `is_object_info_enabled() -> bool` |

Added in 1.1.8

Returns true if the objects information is enabled.


---
### `setObjectsInfoEnabled`

| C++    | `void setObjectsInfoEnabled(bool objectsInfo)`       |
| :--    | :--                                                  |
| Python | `set_objects_info_enabled(bool objectsInfo) -> None` |

Added in 1.1.8

Enables information about all objects present in the current episode/level.
It will be available in the state.

Default value: false

Has no effect when the game is running.

Config key: `objectsInfoEnabled/objects_info_enabled`

See also:
- [`GameState`](./gameState.md#gamestate)
- [`GameState: Object`](./gameState.md#object)
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py),


---
### `isSectorsInfoEnabled`

| C++    | `bool isSectorsInfoEnabled()`       |
| :--    | :--                                 |
| Python | `is_sectors_info_enabled() -> bool` |

Added in 1.1.8

Returns true if the information about sectors is enabled.


---
### `setSectorsInfoEnabled`

| C++    | `void setSectorsInfoEnabled(bool sectorsInfo)`       |
| :--    | :--                                                  |
| Python | `set_sectors_info_enabled(bool sectorsInfo) -> None` |

Added in 1.1.8

Enables information about all sectors (map layout) present in the current episode/level.
It will be available in the state.

Default value: false

Has no effect when the game is running.

Config key: `sectorsInfoEnabled/sectors_info_enabled`

See also:
- [`GameState`](./gameState.md#gamestate)
- [`GameState: Sector`](./gameState.md#sector)
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py)


---
### `isAudioBufferEnabled`

| C++    | `bool isAudioBufferEnabled()`       |
| :--    | :--                                 |
| Python | `is_audio_buffer_enabled() -> bool` |

Added in 1.1.9

Returns true if the audio buffer is enabled.


---
### `setAudioBufferEnabled`

| C++    | `void setAudioBufferEnabled(bool audioBuffer)`       |
| :--    | :--                                                  |
| Python | `set_audio_buffer_enabled(bool audioBuffer) -> None` |

Added in 1.1.9

Returns true if the audio buffer is enabled.

Default value: false

Has no effect when the game is running.

Config key: `audioBufferEnabled/audio_buffer_enabled`

See also:
- [`GameState`](./gameState.md#gamestate)
- [`Enums: SamplingRate`](./enums.md#sampling-rate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### `getAudioSamplingRate`

| C++    | `int getAudioSamplingRate()`       |
| :--    | :--                                |
| Python | `get_audio_sampling_rate() -> int` |

Added in 1.1.9

Returns the sampling rate of the audio buffer.


See also:
- [`GameState`](./gameState.md#gamestate)
- [`Enums: SamplingRate`](./enums.md#sampling-rate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### `setAudioSamplingRate`

| C++    | `void setAudioSamplingRate(SamplingRate samplingRate)`       |
| :--    | :--                                                          |
| Python | `set_audio_sampling_rate(SamplingRate samplingRate) -> None` |

Added in 1.1.9

Sets the sampling rate of the audio buffer.

Default value: false

Has no effect when the game is running.

Config key: `audioSamplingRate/audio_samping_rate`

See also:
- [`GameState`](./gameState.md#gamestate)
- [`Enums: SamplingRate`](./enums.md#sampling-rate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### `getAudioBufferSize`

| C++    | `int getAudioBufferSize()`       |
| :--    | :--                              |
| Python | `get_audio_buffer_size() -> int` |

Added in 1.1.9

Returns the size of the audio buffer.


See also:
- [`GameState`](./gameState.md#gamestate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### `setAudioBufferSize`

| C++    | `void setAudioBufferSize(unsigned int size)` |
| :--    | :--                                          |
| Python | `set_audio_buffer_size(size: int) -> None`   |

Added in 1.1.9

Sets the size of the audio buffer. The size is defined by a number of logic tics.
After each action audio buffer will contain audio from the specified number of the last processed tics.
Doom uses 35 ticks per second.

Default value: 4

Has no effect when the game is running.

Config key: `audioBufferSize/audio_buffer_size`

See also:
- [`GameState`](./gameState.md#gamestate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)
