# DoomGame

DoomGame is the main object of the ViZDoom library, representing a single instance of the Doom game and providing the interface for a single agent/player to interact with the game.
The object allows sending actions to the game, getting the game state, etc.
The declarations of this class and its methods can be found in the `include/ViZDoomGame.h` header file.

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
| Python | `new_episode(recording_file_path: str = "") -> None`  |

Initializes a new episode. The state of an environment is completely restarted (all variables and rewards are reset to their initial values).
After calling this method, the first state from the new episode will be available.
If the `recordingFilePath` argument is not empty, the new episode will be recorded to this file (as a Doom lump).

In a multiplayer game, the host can call this method to finish the game.
Then the rest of the players must also call this method to start a new episode.

Note: Changed in 1.1.0

---
### `replayEpisode`

| C++    | `void replayEpisode(std::string filePath, unsigned int player = 0)` |
| :--    | :--                                                                 |
| Python | `replay_episode(file_path: str, player: int = 0) -> None`           |

Replays the recorded episode from the given file using the perspective of the specified player.
Players are numbered from 1, If `player` argument is equal to 0,
the episode will be replayed using the perspective of the default player in the recording file.
After calling this method, the first state from the replay will be available.
All rewards, variables, and states are available when replaying the episode.

See also:
- [examples/python/record_episodes.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/record_episodes.py)
- [examples/python/record_multiplayer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/record_multiplayer.py)

Note: added in 1.1.0.


---
### `isRunning`

| C++    | `bool isRunning()`     |
| :--    | :--                    |
| Python | `is_running() -> bool` |

Returns true if the controlled game instance is running.


---
### `isMultiplayerGame`

| C++    | `bool isMultiplayerGame()`      |
| :--    | :--                             |
| Python | `is_multiplayer_game() -> bool` |

Returns true if the game is in multiplayer mode.

See also:
- [examples/python/multiple_instances.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/multiple_instances.py)
- [examples/python/cig_multiplayer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/cig_multiplayer.py)
- [examples/python/cig_multiplayer_host.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/cig_multiplayer_host.py)

Note: added in 1.1.2.


---
### `isRecordingEpisode`

| C++    | `bool isRecordingEpisode()`      |
| :--    | :--                              |
| Python | `is_recording_episode() -> bool` |

Returns true if the game is in recording mode.

Note: added in 1.1.5.


---
### `isReplayingEpisode`

| C++    | `bool isReplayingEpisode()`      |
| :--    | :--                              |
| Python | `is_replaying_episode() -> bool` |

Returns true if the game is in replay mode.

Note: added in 1.1.5.

---
### `setAction`

| C++    | `void setAction(std::vector<double> const &action)`          |
| :--    | :--                                                          |
| Python | `set_action(action: list | tuple | ndarray [float]) -> None` |

Sets the player's action for the following tics until the method is called again with new action.
Each value corresponds to a button previously specified
with [`addAvailableButton`](#addavailablebutton), or [`setAvailableButtons`](#setavailablebuttons) methods,
or in the configuration file (in order of appearance).


---
### `advanceAction`

| C++    | `void advanceAction(unsigned int tics = 1, bool updateState = true)` |
| :--    | :--                                                                  |
| Python | `advance_action(tics: int = 1, update_state: bool = True) -> None`   |

Processes the specified number of tics, the last action set with [`setAction`](#setaction)
method will be repeated for each tic. If `updateState` argument is set,
the state will be updated after the last processed tic
and a new reward will be calculated based on all processed tics since last the last state update.
To get the new state, use [`getState`](#getstate) and to get the new reward use [`getLastReward`](#getlastreward).


---
### `makeAction`

| C++    | `double makeAction(std::vector<double> const &actions, unsigned int tics = 1)` |
| :--    | :--                                                                            |
| Python | `make_action(actions: list | tuple | ndarray [float], tics: int = 1) -> float` |

This method combines functionality of [`setAction`](#setaction), [`advanceAction`](#advanceaction),
and [`getLastReward`](#getlastreward) called in this sequance.
Sets the player's action for all the next tics (the same action will be repeated for each tic),
processes the specified number of tics, updates the state and calculates a new reward from all processed tics, which is returned.


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
[`makeAction`](#makeaction) and [`advanceAction`](#advanceaction) methods
will take no effect after this point (unless [`newEpisode`](#newepisode) method is called).


---
### `isEpisodeTimeoutReached`

| C++    | `bool isEpisodeTimeoutReached()`      |
| :--    | :--                             |
| Python | `is_episode_timeout_reached() -> bool` |

Returns true if the current episode is in the terminal state due to exceeding the time limit (timeout)
set with [`setEpisodeTimeout`](#setepisodetimeout) method or via `+timelimit` parameter.


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

| C++    | `GameStatePtr (std::shared_ptr<GameState>) getState()` |
| :--    | :--                                                              |
| Python | `get_state() -> GameState`                                       |

Returns [`GameState`](./game_state.md#gamestate) object with the current game state.
If the current episode is finished, `nullptr/null/None` will be returned.

Note: Changed in 1.1.0


---
### `getServerState`

| C++    | `ServerStatePtr (std::shared_ptr<ServerState>) getServerState()` |
| :--    | :--                                                                          |
| Python | `get_state_state() -> ServerState`                                           |

Returns [`ServerState`](./game_state.md#serverstate) object with the current server state.

Note: added in 1.1.6.


---
### `getLastAction`

| C++    | `std::vector<double> getLastAction()` |
| :--    | :--                                   |
| Python | `get_last_action() -> list`           |

Returns the last action performed.
Each value corresponds to a button added with [`setAvailableButtons`](#setAvailableButtons)
or/and [`addAvailableButton`](#addAvailableButton) (in order of appearance).
Most useful in `SPECTATOR` mode.


---
### `getEpisodeTime`

| C++    | `unsigned int getEpisodeTime()` |
| :--    | :--                             |
| Python | `get_episode_time() -> int`     |

Returns number of current episode tic.


---
### `save`

| C++    | `void save(std::string filePath)` |
| :--    | :--                               |
| Python | `save(file_path: str) -> None`    |

Saves a game's internal state to the file using ZDoom save game functionality.

Note: added in 1.1.9.


---
### `load`

| C++    | `void load(std::string filePath)` |
| :--    | :--                               |
| Python | `load(file_path: str) -> None`    |

Loads a game's internal state from the file using ZDoom load game functionality.
A new state is available after loading.
Loading the game state does not reset the current episode state,
tic counter/time and total reward state keep their values.

Note: added in 1.1.9.


## Buttons settings methods

### `getAvailableButtons`

| C++    | `std::vector<Button> getAvailableButtons()` |
| :--    | :--                                         |
| Python | `get_available_buttons() -> list[Button]`   |

Returns the list of available [`Button`](./enums.md#button) s,
that were added with [`setAvailableButtons`](#setavailablebuttons) or/and [`addAvailableButton`](#addavailablebutton) methods.


---
### `setAvailableButtons`

| C++    | `void setAvailableButtons(std::vector<Button> buttons)`        |
| :--    | :--                                                            |
| Python | `add_available_buttons(buttons: list | tuple[Button]) -> None` |

Sets given list of [`Button`](./enums.md#button) s (e.g. `TURN_LEFT`, `MOVE_FORWARD`) as available buttons.

Has no effect when the game is running.

Default value: `[]` (empty vector/list, no buttons).

Config key: `availableButtons`/`available_buttons` (list of values)


---
### `addAvailableButton`

| C++    | `void addAvailableButton(Button button, double maxValue = 0)`       |
| :--    | :--                                                                 |
| Python | `add_available_button(button: Button, maxValue: float = 0) -> None` |

Adds [`Button`](./enums.md#button) type (e.g. `TURN_LEFT`, `MOVE_FORWARD`) to available buttons and sets the maximum allowed, absolute value for the specified button.
If the given button has already been added, it will not be added again, but the maximum value will be overridden.

Has no effect when the game is running.

Config key: `availableButtons`/`available_buttons` (list of values)


---
### `clearAvailableButtons`

| C++    | `void clearAvailableButtons()`      |
| :--    | :--                                 |
| Python | `clear_available_buttons() -> None` |

Clears all available [`Button`](./enums.md#button)s added so far.

Has no effect when the game is running.


---
### `getAvailableButtonsSize`

| C++    | `int getAvailableButtonsSize()`       |
| :--    | :--                                   |
| Python | `get_available_buttons_size() -> int` |

Returns the number of available [`Button`](./enums.md#button) s.


---
### `setButtonMaxValue`

| C++    | `void setButtonMaxValue(Button button, double maxValue = 0)`        |
| :--    | :--                                                                 |
| Python | `set_button_max_value(button: Button, maxValue: float = 0) -> None` |

Sets the maximum allowed absolute value for the specified [`Button`](./enums.md#button).
Setting the maximum value to 0 results in no constraint at all (infinity).
This method makes sense only for delta buttons.
The constraints limit applies in all Modes.

Default value: 0 (no constraint, infinity).

Has no effect when the game is running.


---
### `getButtonMaxValue`

| C++    | `unsigned int getButtonMaxValue(Button button)` |
| :--    | :--                                             |
| Python | `set_button_max_value(button: Button) -> int`   |

Returns the maximum allowed absolute value for the specified [`Button`](./enums.md#button).


---
### `getButton`

| C++    | `double getButton(Button button)`     |
| :--    | :--                                   |
| Python | `set_button(button: Button) -> float` |

Returns the current state of the specified [`Button`](./enums.md#button) (`ATTACK`, `USE` etc.).


## GameVariables methods


### `getAvailableGameVariables`

| C++    | `std::vector<GameVariable> getAvailableGameVariables()` |
| :--    | :--                                                     |
| Python | `get_available_game_variables() -> list[GameVariables]` |

Returns the list of available [`GameVariable`](./enums.md#gamevariable) s,
that were added with [`setAvailableGameVariables`](#setavailablegamevariables) or/and [`addAvailableGameVariable`](#addavailablegamevariable) methods.


---
### `setAvailableGameVariables`

| C++    | `void setAvailableGameVariables(std::vector<GameVariable> variables)`          |
| :--    | :--                                                                            |
| Python | `set_available_game_variables(variables: list | tuple[GameVariables]) -> None` |

Sets list of [`GameVariable`](./enums.md#gamevariable) s as available game variables in the [`GameState`](./game_state.md#gamestate) returned by [`getState`](#getstate) method.

Has no effect when the game is running.

Default value: `[]` (empty vector/list, no game variables).

Config key: `availableGameVariables`/`available_game_variables` (list of values)


---
### `addAvailableGameVariable`

| C++    | `void addAvailableGameVariable(GameVariable variable)`        |
| :--    | :--                                                           |
| Python | `add_available_game_variable(variable: GameVariable) -> None` |

Adds the specified [`GameVariable`](./enums.md#gamevariable) to the list of available game variables (e.g. `HEALTH`, `AMMO1`, `ATTACK_READY`) in the [`GameState`](./game_state.md#gamestate) returned by [`getState`](#getstate) method.

Has no effect when the game is running.

Config key: `availableGameVariables`/`available_game_variables` (list of values)


---
### `clearAvailableGameVariables`

| C++    | `void clearAvailableGameVariables()`       |
| :--    | :--                                        |
| Python | `clear_available_game_variables() -> None` |

Clears the list of available [`GameVariable`](./enums.md#gamevariable) s that are included in the [`GameState`](./game_state.md#gamestate) returned by [`getState`](#getstate) method.

Has no effect when the game is running.


---
### `getAvailableGameVariablesSize`

| C++    | `unsigned int getAvailableGameVariablesSize()` |
| :--    | :--                                            |
| Python | `get_available_game_variables_size() -> int`   |

Returns the number of available [`GameVariable`](./enums.md#gamevariable).
It corresponds to taking the size of the list returned by [`getAvailableGameVariables`](#getavailablegamevariables).


---
### `getGameVariable`

| C++    | `double getGameVariable(GameVariable variable)`      |
| :--    | :--                                                  |
| Python | `get_game_variable(variable: GameVariable) -> float` |

Returns the current value of the specified [`GameVariable`](./enums.md#gamevariable) (`HEALTH`, `AMMO1` etc.).
The specified game variable does not need to be among available game variables (included in the state).
It could be used for e.g. shaping. Returns 0 in case of not finding given [`GameVariable`](./enums.md#gamevariable).


## Game arguments methods


### `setGameArgs`

| C++    | `void setGameArgs(std::string args)` |
| :--    | :--                                  |
| Python | `set_game_args(args: str) -> None`   |

Sets custom arguments that will be passed to ViZDoom process during initialization.
It is useful for changing additional game settings.
Use with caution, as in rare cases it may prevent the library from working properly.
Using this method is equivalent to first calling [`clearGameArgs`](#cleargameargs) and then [`addGameArgs`](#addgameargs).

Default value: `""` (empty string, no additional arguments).

Config key: `gameArgs`/`game_args`

See also:
- [ZDoom Wiki: Command line parameters](http://zdoom.org/wiki/Command_line_parameters)
- [ZDoom Wiki: CVARs (Console Variables)](http://zdoom.org/wiki/CVARS)

Note: added in 1.2.3.


---
### `addGameArgs`

| C++    | `void addGameArgs(std::string args)` |
| :--    | :--                                  |
| Python | `add_game_args(args: str) -> None`   |

Adds custom arguments that will be passed to ViZDoom process during initialization.
It is useful for changing additional game settings.
Use with caution, as in rare cases it may prevent the library from working properly.

Config key: `gameArgs`/`game_args`

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

Note: added in 1.2.3.


## Reward methods


### `getLivingReward`

| C++    | `double getLivingReward()`     |
| :--    | :--                            |
| Python | `get_living_reward() -> float` |

Returns the reward granted to the player after every tic.


---
### `setLivingReward`

| C++    | `void setLivingReward(double livingReward)`       |
| :--    | :--                                               |
| Python | `set_living_reward(living_reward: float) -> None` |

Sets the reward granted to the player after every tic. A negative value is also allowed.

Default value: 0

Config key: `livingReward`/`living_reward`


---
### `getDeathPenalty`

| C++    | `double getDeathPenalty()`     |
| :--    | :--                            |
| Python | `get_death_penalty() -> float` |

Returns the penalty for the player's death.


---
### `setDeathPenalty`

| C++    | `void setDeathPenalty(double deathPenalty)`       |
| :--    | :--                                               |
| Python | `set_death_penalty(death_penalty: float) -> None` |

Sets a penalty for the player's death. Note that in case of a negative value, the player will be rewarded upon dying.

Default value: 0

Config key: `deathPenalty`/`death_penalty`


---
### `getDeathReward`

| C++    | `double getDeathReward()`     |
| :--    | :--                           |
| Python | `get_death_reward() -> float` |

Returns the reward for the player's death. It is equal to negation of value returned by [`getDeathReward`](#getdeathreward).

Note: added in 1.3.0


---
### `setDeathReward`

| C++    | `void setDeathReward(double deathReward)`       |
| :--    | :--                                             |
| Python | `set_death_reward(death_reward: float) -> None` |

Sets a reward for the player's death. A negative value is also allowed.

Default value: 0

Config key: `deathReward`/`death_reward`

Note: added in 1.3.0


---
### `getMapExitReward`

| C++    | `double getMapExitReward()`      |
| :--    | :--                              |
| Python | `get_map_exit_reward() -> float` |

Returns the reward for finishing a map.

Note: added in 1.3.0


---
### `setMapExitReward`

| C++    | `void setMapExitReward(double mapExitReward)`         |
| :--    | :--                                                   |
| Python | `set_map_exit_reward(map_exit_reward: float) -> None` |

Sets a reward for finishing a map (finding an exit or succeeding in other programmed objective). A negative value is also allowed.

Default value: 0

Config key: `mapExitReward`/`map_exit_reward`

Note: added in 1.3.0


---
### `getKillReward`

| C++    | `double getKillReward()`     |
| :--    | :--                          |
| Python | `get_kill_reward() -> float` |

Returns the reward granted to the player for killing an enemy.

Note: added in 1.3.0


---
### `setKillReward`

| C++    | `void setKillReward(double killReward)`       |
| :--    | :--                                           |
| Python | `set_kill_reward(kill_reward: float) -> None` |

Sets the reward granted to the player for killing an enemy. A negative value is also allowed.

Default value: 0

Config key: `killReward`/`kill_reward`

Note: added in 1.3.0


---
### `getItemReward`

| C++    | `double getItemReward()`     |
| :--    | :--                          |
| Python | `get_item_reward() -> float` |

Returns the reward granted to the player for picking up an item.

Note: added in 1.3.0


---
### `setItemReward`

| C++    | `void setItemReward(double itemReward)`       |
| :--    | :--                                           |
| Python | `set_item_reward(item_reward: float) -> None` |

Sets the reward granted to the player for picking up an item. A negative value is also allowed.

Default value: 0

Config key: `itemReward`/`item_reward`

Note: added in 1.3.0


---
### `getSecretReward`

| C++    | `double getSecretReward()`     |
| :--    | :--                            |
| Python | `get_secret_reward() -> float` |

Returns the reward granted to the player for discovering a secret.

Note: added in 1.3.0


---
### `setSecretReward`

| C++    | `void setSecretReward(double secretReward)`       |
| :--    | :--                                               |
| Python | `set_secret_reward(secret_reward: float) -> None` |

Sets the reward granted to the player for discovering a secret. A negative value is also allowed.

Default value: 0

Config key: `secretReward`/`secret_reward`

Note: added in 1.3.0


---
### `getFragReward`

| C++    | `double getFragReward()`     |
| :--    | :--                          |
| Python | `get_frag_reward() -> float` |

Returns the reward granted to the player for scoring a frag (killing another player in multiplayer).

Note: added in 1.3.0


---
### `setFragReward`

| C++    | `void setFragReward(double fragReward)`       |
| :--    | :--                                           |
| Python | `set_frag_reward(frag_reward: float) -> None` |

Sets the reward granted to the player for scoring a frag. A negative value is also allowed.

Default value: 0

Config key: `fragReward`/`frag_reward`

Note: added in 1.3.0


---
### `getHitReward`

| C++    | `double getHitReward()`     |
| :--    | :--                         |
| Python | `get_hit_reward() -> float` |

Returns the reward granted to the player for hitting (damaging) an enemy.

Note: added in 1.3.0


---
### `setHitReward`

| C++    | `void setHitReward(double hitReward)`       |
| :--    | :--                                         |
| Python | `set_hit_reward(hit_reward: float) -> None` |

Sets the reward granted to the player for hitting (damaging) an enemy.
The reward is the same despite the amount of damage dealt.
A negative value is also allowed.

Default value: 0

Config key: `hitReward`/`hit_reward`

Note: added in 1.3.0


---
### `getHitTakenReward`

| C++    | `double getHitTakenReward()`      |
| :--    | :--                               |
| Python | `get_hit_taken_reward() -> float` |

Returns the reward granted to the player when hit (damaged) by an enemy.
The reward is the same despite the amount of damage taken.

Note: added in 1.3.0


---
### `setHitTakenReward`

| C++    | `void setHitTakenReward(double hitTakenReward)`         |
| :--    | :--                                                     |
| Python | `set_hit_taken_reward(hit_taken_reward: float) -> None` |

Sets the reward granted to the player when hit (damaged) by an enemy.
The reward is the same despite the amount of damage taken.
A negative value is also allowed.

Default value: 0

Config key: `hitTakenReward`/`hit_taken_reward`

Note: added in 1.3.0


---
### `getHitTakenPenalty`

| C++    | `double getHitTakenPenalty()`      |
| :--    | :--                                |
| Python | `get_hit_taken_penalty() -> float` |

Returns the penalty for the player when hit (damaged) by an enemy.
The penalty is the same despite the amount of damage taken.
It is equal to negation of value returned by [`getHitTakenReward`](#gethittakenreward).

Note: added in 1.3.0


---
### `setHitTakenPenalty`

| C++    | `void setHitTakenPenalty(double hitTakenPenalty)`         |
| :--    | :--                                                       |
| Python | `set_hit_taken_penalty(hit_taken_penalty: float) -> None` |

Sets a penalty for the player when hit (damaged) by an enemy.
The penalty is the same despite the amount of damage taken.
Note that in case of a negative value, the player will be rewarded upon being hit.

Default value: 0

Config key: `hitTakenPenalty`/`hit_taken_penalty`

Note: added in 1.3.0


---
### `getDamageMadeReward`

| C++    | `double getDamageMadeReward()`      |
| :--    | :--                                 |
| Python | `get_damage_made_reward() -> float` |

Returns the reward granted to the player for damaging an enemy, proportional to the damage dealt.
Every point of damage dealt to an enemy will result in a reward equal to the value returned by this method.

Note: added in 1.3.0


---
### `setDamageMadeReward`

| C++    | `void setDamageMadeReward(double damageMadeReward)`         |
| :--    | :--                                                         |
| Python | `set_damage_made_reward(damage_made_reward: float) -> None` |

Sets the reward granted to the player for damaging an enemy, proportional to the damage dealt.
Every point of damage dealt to an enemy will result in a reward equal to the value returned by this method.
A negative value is also allowed.


Default value: 0

Config key: `damageMadeReward`/`damage_made_reward`

Note: added in 1.3.0


---
### `getDamageTakenReward`

| C++    | `double getDamageTakenReward()`      |
| :--    | :--                                  |
| Python | `get_damage_taken_reward() -> float` |

Returns the reward granted to the player when damaged by an enemy, proportional to the damage received.
Every point of damage taken will result in a reward equal to the value returned by this method.

Note: added in 1.3.0


---
### `setDamageTakenReward`

| C++    | `void setDamageTakenReward(double DamageTakenReward)`         |
| :--    | :--                                                           |
| Python | `set_damage_taken_reward(damage_taken_reward: float) -> None` |

Sets the reward granted to the player when damaged by an enemy, proportional to the damage received.
Every point of damage taken will result in a reward equal to the set value.
A negative value is also allowed.

Default value: 0

Config key: `damageTakenReward`/`damage_taken_reward`

Note: added in 1.3.0


---
### `getDamageTakenPenalty`

| C++    | `double getDamageTakenPenalty()`      |
| :--    | :--                                   |
| Python | `get_damage_taken_penalty() -> float` |

Returns the penalty for the player when damaged by an enemy, proportional to the damage received.
Every point of damage taken will result in a penalty equal to the value returned by this method.
It is equal to negation of value returned by [`getDamageTakenReward`](#getdamagetakenreward).

Note: added in 1.3.0


---
### `setDamageTakenPenalty`

| C++    | `void setDamageTakenPenalty(double DamageTakenPenalty)`         |
| :--    | :--                                                             |
| Python | `set_damage_taken_penalty(damage_taken_penalty: float) -> None` |

Sets a penalty for the player when damaged by an enemy, proportional to the damage received.
Every point of damage taken will result in a penalty equal to the set value.
Note that in case of a negative value, the player will be rewarded upon receiving damage.

Default value: 0

Config key: `damageTakenPenalty`/`damage_taken_penalty`

Note: added in 1.3.0


---
### `getHealthReward`
| C++    | `double getHealthReward()`     |
| :--    | :--                            |
| Python | `get_health_reward() -> float` |

Returns the reward granted to the player for getting health points.

Note: added in 1.3.0


---
### `setHealthReward`
| C++    | `void setHealthReward(double healthReward)`       |
| :--    | :--                                               |
| Python | `set_health_reward(health_reward: float) -> None` |

Sets the reward granted to the player for getting health points. A negative value is also allowed.

Default value: 0

Config key: `healthReward`/`health_reward`

Note: added in 1.3.0


---
### `getArmorReward`
| C++    | `double getArmorReward()`     |
| :--    | :--                           |
| Python | `get_armor_reward() -> float` |

Returns the reward granted to the player for getting armor points.

Note: added in 1.3.0


---
### `setArmorReward`
| C++    | `void setArmorReward(double armorReward)`       |
| :--    | :--                                             |
| Python | `set_armor_reward(armor_reward: float) -> None` |

Sets the reward granted to the player for getting armor points. A negative value is also allowed.

Default value: 0

Config key: `armorReward`/`armor_reward`

Note: added in 1.3.0


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
| Python | `load_config(file_path: str) -> bool`   |

Loads configuration (resolution, available buttons, game variables etc.) from a configuration file.
In case of multiple invocations, older configurations will be overwritten by the recent ones.
Overwriting does not involve resetting to default values. Thus only overlapping parameters will be changed.
The method returns true if the whole configuration file was correctly read and applied,
false if the file contained errors.

If the file relative path is given, it will be searched for in the following order: `<current directory>`, `<current directory>/scenarios/`, `<ViZDoom library location>/scenarios/`.


---
### `getMode`

| C++    | `Mode getMode()`     |
| :--    | :--                  |
| Python | `get_mode() -> Mode` |

Returns the current [`Mode`](./enums.md#mode) (`PLAYER`, `SPECTATOR`, `ASYNC_PLAYER`, `ASYNC_SPECTATOR`).


---
### `setMode`

| C++    | `void setMode(Mode mode)`      |
| :--    | :--                            |
| Python | `set_mode(mode: Mode) -> None` |

Sets the [`Mode`](./enums.md#mode) (`PLAYER`, `SPECTATOR`, `ASYNC_PLAYER`, `ASYNC_SPECTATOR`) in which the game will be running.

Default value: `PLAYER`.

Has no effect when the game is running.

Config key: `mode`


---
### `getTicrate`

| C++    | `unsigned int getTicrate()` |
| :--    | :--                         |
| Python | `get_ticrate() -> int`      |

Returns current ticrate.

Note: added in 1.1.0.


---
### `setTicrate`

| C++    | `void setTicrate(unsigned int ticrate)` |
| :--    | :--                                     |
| Python | `set_ticrate(ticrate: int) -> None`     |

Sets the ticrate for ASNYC Modes - number of logic tics executed per second.
The default Doom ticrate is 35. This value will play a game at normal speed.

Default value: 35 (default Doom ticrate).

Has no effect when the game is running.

Config key: `ticrate`

See also:
- [examples/python/ticrate.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/ticrate.py)

Note: added in 1.1.0.


---
### `setViZDoomPath`

| C++    | `void setViZDoomPath(std::string filePath)` |
| :--    | :--                                         |
| Python | `set_vizdoom_path(file_path: str) -> None`  |

Sets the path to the ViZDoom engine executable vizdoom.

Default value: `<ViZDoom library location>/<vizdoom or vizdoom.exe on Windows>`.

Config key: `ViZDoomPath`/`vizdoom_path`


---
### `setDoomGamePath`

| C++    | `void setDoomGamePath(std::string filePath)` |
| :--    | :--                                          |
| Python | `set_doom_game_path(file_path: str) -> None` |

Sets the path to the Doom engine based game file (wad format).
If not used DoomGame will look for doom2.wad and freedoom2.wad (in that order) in the directory of ViZDoom's installation (where vizdoom library/pyd is).

Default value: `<ViZDoom library location>/<doom2.wad, doom.wad, freedoom2.wad, or freedoom.wad - in this order>`

Config key: `DoomGamePath`/`doom_game_path`


---
### `setDoomScenarioPath`

| C++    | `void setDoomScenarioPath(std::string filePath)` |
| :--    | :--                                              |
| Python | `set_doom_scenario_path(file_path: str) -> None` |

Sets the path to an additional scenario file (wad format).
If not provided, the default Doom single-player maps will be loaded.

Default value: `""`

Config key: `DoomScenarioPath`/`doom_scenario_path`


---
### `setDoomMap`

| C++    | `void setDoomMap(std::string map)` |
| :--    | :--                                |
| Python | `set_doom_map(map: str) -> None`   |

Sets the map name to be used.

Default value: `"map01"`, if set to empty `"map01"` will be used.

Config key: `DoomMap`/`doom_map`


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

Config key: `DoomSkill`/`doom_skill`


---
### `setDoomConfigPath`

| C++    | `void setDoomConfigPath(std::string filePath)`  |
| :--    | :--                                             |
| Python | `set_doom_config_path(file_path: str) -> None`  |

Sets the path for ZDoom's configuration file.
The file is responsible for the configuration of the ZDoom engine itself.
If it does not exist, it will be created after the `vizdoom` executable is run.
This method is not needed for most of the tasks and is added for the convenience of users with hacking tendencies.

Default value: `""`, if left empty `"_vizdoom.ini"` will be used.

Config key: `DoomConfigPath`/`doom_config_path`


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

Config key: `episodeStartTime`/`episode_start_time`


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

Config key: `episodeTimeout`/`episode_timeout`


## Output/rendering setting methods


### `setScreenResolution`

| C++    | `void setScreenResolution(ScreenResolution resolution)`       |
| :--    | :--                                                           |
| Python | `set_screen_resolution(resolution: ScreenResolution) -> None` |

Sets the screen resolution and additional buffers (depth, labels, and automap). ZDoom engine supports only specific resolutions.
Supported resolutions are part of [`ScreenResolution`](./enums.md#screenresolution) enumeration (e.g., `RES_320X240`, `RES_640X480`, `RES_1920X1080`).
The buffers, as well as the content of ViZDoom's display window, will be affected.

Default value: `RES_320X240`

Has no effect when the game is running.

Config key: `screenResolution`/`screen_resolution`


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
Supported formats are defined in [`ScreenFormat`](./enums.md#screenformat) enumeration type (e.g. `CRCGCB`, `RGB24`, `GRAY8`).
The format change affects only the buffers, so it will not have any effect on the content of ViZDoom's display window.

Default value: `CRCGCB`

Has no effect when the game is running.

Config key: `screenFormat`/`screen_format`


---
### `isDepthBufferEnabled`

| C++    | `bool isDepthBufferEnabled()`       |
| :--    | :--                                 |
| Python | `is_depth_buffer_enabled() -> None` |

Returns true if the depth buffer is enabled.


---
### `setDepthBufferEnabled`

| C++    | `void setDepthBufferEnabled(bool depthBuffer)`         |
| :--    | :--                                                    |
| Python | `set_depth_buffer_enabled(depth_buffer: bool) -> None` |

Enables rendering of the depth buffer, it will be available in the state.
The buffer always has the same resolution as the screen buffer.
Depth buffer will contain noise if `viz_nocheat` flag is enabled.

Default value: false

Has no effect when the game is running.

Config key: `depthBufferEnabled`/`depth_buffer_enabled`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py)

Note: added in 1.1.0.


---
### `isLabelsBufferEnabled`

| C++    | `bool isLabelsBufferEnabled()`       |
| :--    | :--                                  |
| Python | `is_labels_buffer_enabled() -> None` |

Returns true if the labels buffer is enabled.

Note: added in 1.1.0.


---
### `setLabelsBufferEnabled`

| C++    | `void setLabelsBufferEnabled(bool labelsBuffer)`         |
| :--    | :--                                                      |
| Python | `set_labels_buffer_enabled(labels_buffer: bool) -> None` |

Enables rendering of the labels buffer, it will be available in the state with the vector of [`Label`](./game_state.md#label) s.
The buffer always has the same resolution as the screen buffer.
LabelsBuffer will contain noise if `viz_nocheat` is enabled.

Default value: false

Has no effect when the game is running.

Config key: `labelsBufferEnabled`/`labels_buffer_enabled`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/labels.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/labels.py)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py)

Note: added in 1.1.0.


---
### `isAutomapBufferEnabled`

| C++    | `bool isAutomapBufferEnabled()`       |
| :--    | :--                                   |
| Python | `is_automap_buffer_enabled() -> bool` |

Returns true if the automap buffer is enabled.

Note: added in 1.1.0.


---
### `setAutomapBufferEnabled`

| C++    | `void setAutomapBufferEnabled(bool automapBuffer)`         |
| :--    | :--                                                        |
| Python | `set_automap_buffer_enabled(automap_buffer: bool) -> None` |

Enables rendering of the automap buffer, it will be available in the state.
The buffer always has the same resolution as the screen buffer.

Default value: false

Has no effect when the game is running.

Config key: `automapBufferEnabled`/`automap_buffer_enabled`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py),

Note: added in 1.1.0.


---
### `setAutomapMode`

| C++    | `void setAutomapMode(AutomapMode mode)`       |
| :--    | :--                                           |
| Python | `set_automap_mode(mode: AutomapMode) -> None` |

Sets the [`AutomapMode`](./enums.md#automapmode) (`NORMAL`, `WHOLE`, `OBJECTS`, `OBJECTS_WITH_SIZE`),
which determines what will be visible on it.

Default value: `NORMAL`

Config key: `automapMode`/`automap_mode`

Note: added in 1.1.0.


---
### `setAutomapRotate`

| C++    | `void setAutomapRotate(bool rotate)`       |
| :--    | :--                                        |
| Python | `set_automap_rotate(rotate: bool) -> None` |

Determine if the automap will be rotating with the player.
If false, north always will be at the top of the buffer.

Default value: false

Config key: `automapRotate`/`automap_rotate`

Note: added in 1.1.0.


---
### `setAutomapRenderTextures`

| C++    | `void setAutomapRenderTextures(bool textures)`        |
| :--    | :--                                                   |
| Python | `set_automap_render_textures(textures: bool) -> None` |

Determine if the automap will be textured, showing the floor textures.

Default value: true

Config key: `automapRenderTextures`/`automap_render_textures`

Note: added in 1.1.0.


---
### `setRenderHud`

| C++    | `void setRenderHud(bool hud)`       |
| :--    | :--                                 |
| Python | `set_render_hud(hud: bool) -> None` |

Determine if the hud will be rendered in the game.

Default value: false

Config key: `renderHud`/`render_hud`


---
### `setRenderMinimalHud`

| C++    | `void setRenderMinimalHud(bool minHud)`         |
| :--    | :--                                             |
| Python | `set_render_minimal_hud(min_hud: bool) -> None` |

Determine if the minimalistic version of the hud will be rendered instead of the full hud.

Default value: false

Config key: `renderMinimalHud`/`render_minimal_hud`

Note: added in 1.1.0.


---
### `setRenderWeapon`

| C++    | `void setRenderWeapon(bool weapon)`       |
| :--    | :--                                       |
| Python | `set_render_weapon(weapon: bool) -> None` |

Determine if the weapon held by the player will be rendered in the game.

Default value: true

Config key: `renderWeapon`/`render_weapon`


---
### `setRenderCrosshair`

| C++    | `void setRenderCrosshair(bool crosshair)`       |
| :--    | :--                                             |
| Python | `set_render_crosshair(crosshair: bool) -> None` |

Determine if the crosshair will be rendered in the game.

Default value: false

Config key: `renderCrosshair`/`render_crosshair`


---
### `setRenderDecals`

| C++    | `void setRenderDecals(bool decals)`       |
| :--    | :--                                       |
| Python | `set_render_decals(decals: bool) -> None` |

Determine if the decals (marks on the walls) will be rendered in the game.

Default value: true

Config key: `renderDecals`/`render_decals`


---
### `setRenderParticles`

| C++    | `void setRenderParticles(bool particles)`       |
| :--    | :--                                             |
| Python | `set_render_particles(particles: bool) -> None` |

Determine if the particles will be rendered in the game.

Default value: true

Config key: `renderParticles`/`render_particles`


---
### `setRenderEffectsSprites`

| C++    | `void setRenderEffectsSprites(bool sprites)`        |
| :--    | :--                                                 |
| Python | `set_render_effects_sprites(sprites: bool) -> None` |

Determine if some effects sprites (gun puffs, blood splats etc.) will be rendered in the game.

Default value: true

Config key: `renderEffectsSprites`/`render_effects_sprites`

Note: added in 1.1.0.


---
### `setRenderMessages`

| C++    | `void setRenderMessages(bool messages)`       |
| :--    | :--                                           |
| Python | `set_render_messages(messages: bool) -> None` |

Determine if in-game messages (information about pickups, kills, etc.) will be rendered in the game.

Default value: false

Config key: `renderMessages`/`render_messages`

Note: added in 1.1.0.


---
### `setRenderCorpses`

| C++    | `void setRenderCorpses(bool corpses)`        |
| :--    | :--                                          |
| Python | `set_render_corpsess(corpses: bool) -> None` |

Determine if actors' corpses will be rendered in the game.

Default value: true

Config key: `renderCorpses`/`render_corpses`

Note: added in 1.1.0.


---
### `setRenderScreenFlashes`

| C++    | `void setRenderScreenFlashes(bool flashes)`        |
| :--    | :--                                                |
| Python | `set_render_screen_flashes(flashes: bool) -> None` |

Determine if the screen flash effect upon taking damage or picking up items will be rendered in the game.

Default value: true

Config key: `renderScreenFlashes`/`render_screen_flashes`

Note: added in 1.1.0.


---
### `setRenderAllFrames`

| C++    | `void setRenderAllFrames(bool allFrames)`         |
| :--    | :--                                               |
| Python | `set_render_all_frames(all_frames: bool) -> None` |

Determine if all frames between states will be rendered (when skip greater than 1 is used).
Allows smooth preview but can reduce performance.
It only makes sense to use it if the window is visible.

Default value: false

Config key: `renderAllFrames`/`render_all_frames`

See also:
- [`setWindowVisible`](#setwindowvisible)

Note: added in 1.1.3.


---
### `setWindowVisible`

| C++    | `void setWindowVisible(bool visibility)`       |
| :--    | :--                                            |
| Python | `set_window_visible(visibility: bool) -> None` |

Determines if ViZDoom's window will be visible.
ViZDoom with window disabled can be used on Linux systems without X Server.

Default value: false

Has no effect when the game is running.

Config key: `windowVisible`/`window_visible`


---
### `setConsoleEnabled`

| C++    | `void setConsoleEnabled(bool console)`       |
| :--    | :--                                          |
| Python | `set_console_enabled(console: bool) -> None` |

Determines if ViZDoom's console output will be enabled.

Default value: false

Config key: `consoleEnabled`/`console_enabled`


---
### `setSoundEnabled`

| C++    | `void setSoundEnabled(bool sound)`       |
| :--    | :--                                      |
| Python | `set_sound_enabled(sound: bool) -> None` |

Determines if ViZDoom's sound will be played.

Default value: false

Config key: `soundEnabled`/`sound_enabled`


---
### `getScreenWidth`

| C++    | `int getScreenWidth()`      |
| :--    | :--                         |
| Python | `get_screen_width() -> int` |

Returns game's screen width - width of screen, depth, labels, and automap buffers.


---
### `getScreenHeight`

| C++    | `int getScreenHeight()`      |
| :--    | :--                          |
| Python | `get_screen_height() -> int` |

Returns game's screen height - height of screen, depth, labels, and automap buffers.


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

Returns true if the objects information is enabled.

Note: added in 1.1.8.


---
### `setObjectsInfoEnabled`

| C++    | `void setObjectsInfoEnabled(bool objectsInfo)`         |
| :--    | :--                                                    |
| Python | `set_objects_info_enabled(objects_info: bool) -> None` |

Enables information about all [`Object`](./game_state.md#object) s present in the current episode/level.
It will be available in the state.

Default value: false

Has no effect when the game is running.

Config key: `objectsInfoEnabled`/`objects_info_enabled`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py),

Note: added in 1.1.8.


---
### `isSectorsInfoEnabled`

| C++    | `bool isSectorsInfoEnabled()`       |
| :--    | :--                                 |
| Python | `is_sectors_info_enabled() -> bool` |

Returns true if the information about sectors is enabled.

Note: added in 1.1.8.


---
### `setSectorsInfoEnabled`

| C++    | `void setSectorsInfoEnabled(bool sectorsInfo)`         |
| :--    | :--                                                    |
| Python | `set_sectors_info_enabled(sectors_info: bool) -> None` |

Enables information about all [`Sector`](./game_state.md#sector) s (map layout) present in the current episode/level.
It will be available in the state.

Default value: false

Has no effect when the game is running.

Config key: `sectorsInfoEnabled`/`sectors_info_enabled`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py)

Note: added in 1.1.8.


---
### `isAudioBufferEnabled`

| C++    | `bool isAudioBufferEnabled()`       |
| :--    | :--                                 |
| Python | `is_audio_buffer_enabled() -> bool` |

Returns true if the audio buffer is enabled.

Note: added in 1.1.9.


---
### `setAudioBufferEnabled`

| C++    | `void setAudioBufferEnabled(bool audioBuffer)`         |
| :--    | :--                                                    |
| Python | `set_audio_buffer_enabled(audio_buffer: bool) -> None` |

Enables rendering of the audio buffer, it will be available in the state.
The audio buffer will contain audio from the number of the last tics specified by [`setAudioBufferSize`](#setaudiobuffersize) method.
Sampling rate can be set with [`setAudioSamplingRate`](#setaudiosamplingrate) method.

Default value: false

Has no effect when the game is running.

Config key: `audioBufferEnabled`/`audio_buffer_enabled`

See also:
- [`GameState`](./game_state.md#gamestate)
- [`SamplingRate`](./enums.md#sampling-rate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)

Note: added in 1.1.9.


---
### `getAudioSamplingRate`

| C++    | `int getAudioSamplingRate()`       |
| :--    | :--                                |
| Python | `get_audio_sampling_rate() -> int` |

Returns the [`SamplingRate`](./enums.md#sampling-rate) of the audio buffer.

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)

Note: added in 1.1.9.


---
### `setAudioSamplingRate`

| C++    | `void setAudioSamplingRate(SamplingRate samplingRate)`         |
| :--    | :--                                                            |
| Python | `set_audio_sampling_rate(sampling_rate: SamplingRate) -> None` |

Sets the [`SamplingRate`](./enums.md#sampling-rate) of the audio buffer.

Default value: false

Has no effect when the game is running.

Config key: `audioSamplingRate`/`audio_sampling_rate`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)

Note: added in 1.1.9.


---
### `getAudioBufferSize`

| C++    | `int getAudioBufferSize()`       |
| :--    | :--                              |
| Python | `get_audio_buffer_size() -> int` |

Returns the size of the audio buffer.

Note: added in 1.1.9.


See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)


---
### `setAudioBufferSize`

| C++    | `void setAudioBufferSize(unsigned int tics)` |
| :--    | :--                                          |
| Python | `set_audio_buffer_size(tics: int) -> None`   |

Sets the size/length of the audio buffer. The size is defined by a number of logic tics.
After each action audio buffer will contain audio from the specified number of the last processed tics.
Doom uses 35 ticks per second.

Default value: 1

Has no effect when the game is running.

Config key: `audioBufferSize`/`audio_buffer_size`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py)

Note: added in 1.1.9.


---
### `isNotificationsBufferEnabled`

| C++    | `bool isNotificationsBufferEnabled()`       |
| :--    | :--                                         |
| Python | `is_notifications_buffer_enabled() -> bool` |

Returns true if the notify buffer is enabled.

Note: added in 1.3.0.


---
### `setNotificationsBufferEnabled`

| C++    | `void setNotificationsBufferEnabled(bool notificationsBuffer)`         |
| :--    | :--                                                                    |
| Python | `set_notifications_buffer_enabled(notifications_buffer: bool) -> None` |

Enables notification buffer, it will be available in the state.
The notification buffer will contain text notifications from the number of the last tics specified by [`setNotificationsBufferSize`](#setNotificationsBuffersize) method.

Default value: false

Has no effect when the game is running.

Config key: `notificationsBufferEnabled`/`notifications_buffer_enabled`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py)

Note: added in 1.3.0.


---
### `getNotificationsBufferSize`

| C++    | `int getNotificationsBufferSize()`       |
| :--    | :--                                      |
| Python | `get_notifications_buffer_size() -> int` |

Returns the size of the notify buffer.

Note: added in 1.3.0.


See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py)


---
### `setNotificationsBufferSize`

| C++    | `void setNotificationsBufferSize(unsigned int tics)` |
| :--    | :--                                                  |
| Python | `set_notifications_buffer_size(tics: int) -> None`   |

Sets the size of the notify buffer. The size is defined by a number of logic tics.
After each action notify buffer will contain text notifications from the specified number of the last processed tics.
Doom uses 35 ticks per second.

Default value: 1

Has no effect when the game is running.

Config key: `notificationsBufferSize`/`notifications_buffer_size`

See also:
- [`GameState`](./game_state.md#gamestate)
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py)

Note: added in 1.3.0.
