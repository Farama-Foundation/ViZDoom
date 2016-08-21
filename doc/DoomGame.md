# DoomGame

## Flow control methods
---

### `init`

| C++    | `bool init()`    |
| :--    | :--              |
| Lua    | `boolean init()` |
| Java   | `boolean init()` |
| Python | `bool init()`    |

Initializes ViZDoom game instance and starts newEpisode.
After calling this method, first state from new episode will be available.
Some configuration options cannot be changed after calling method.
Init returns true when the game was started properly and false otherwise.


### `close`

| C++    | `void close()` |
| :--    | :--            |
| Lua    | `void close()` |
| Java   | `void close()` |
| Python | `void close()` |
        
Closes ViZDoom game instance.
It is automatically invoked by the destructor.
Game can be initialized again after being closed.


### `newEpisode`

| C++    | `void newEpisode(std::string filePath = "")` |
| :--    | :--                                          |
| Lua    | `void newEpisode(string filePath = "")`      |
| Java   | `void newEpisode(String filePath = "")`      |
| Python | `void new_episode(str filePath = "")`        |

Changed in 1.1

Initializes a new episode. All rewards, variables and state are restarted.
After calling this method, first state from new episode will be available.
If the filePath is not empty, given episode will be recorded to this file.

In multiplayer game, host can call this method to finish the game.
Then the rest of the players must also call this method to start a new episode.


### `replayEpisode`

| C++    | `void replayEpisode(std::string filePath, unsigned int player = 0)` |
| :--    | :--                                                                 |
| Lua    | `void replayEpisode(string filePath, number player = 0)`            |
| Java   | `void replayEpisode(String filePath, unsigned int player = 0)`      |
| Python | `void replay_episode(str filePath, int player = 0)`                 |

Added in 1.1

Replays recorded episode from the given file and using perspective of the specified player.
Players are numered from 1, `player` equal to 0 results in replaying demo using perspective 
of default player in record file.
After calling this method, first state from replay will be available.
All rewards, variables and state are available during replaying episode.

See also: 
* examples/python/record_episodes.py,
* examples/python/record_multiplayer.py.


### `isRunning`

| C++    | `bool isRunning()`    |
| :--    | :--                   |
| Lua    | `boolean isRunning()` |
| Java   | `boolean isRunning()` |
| Python | `bool is_running()`   |

Checks if the ViZDoom game instance is running.


### `setAction`

| C++    | `void setAction(std::vector<int> const &actions)` |
| :--    | :--                                               |
| Lua    | `void setAction(table actions)`                   |
| Java   | `void setAction(int[] actions)`                   |
| Python | `void set_action(list actions)`                   |

Sets the player's action for the next tics.
Each value corresponds to a button specified with `addAvailableButton` method
or in configuration file (in order of appearance).


### `advanceAction`

| C++    | `void advanceAction(unsigned int tics = 1, bool updateState = true, bool renderOnly = false)`       |
| :--    | :--                                                                                                 |
| Lua    | `void advanceAction(number tics = 1, boolean updateState = true, boolean renderOnly = false)`       |
| Java   | `void advanceAction(unsigned int tics = 1, boolean updateState = true, boolean renderOnly = false)` |
| Python | `void advance_action(int tics = 1, bool updateState = True, bool renderOnly = False)`               |

Processes a specified number of tics. If `updateState` is set the state will be updated after last processed tic
and a new reward will be calculated. To get new state use `getState` and to get the new reward use `getLastReward`.
If `updateState` is not set but `renderOnly` is turned on, the state will not be updated but a new frame
will be rendered after last processed tic.


### `makeAction`

| C++    | `double makeAction(std::vector<int> const &actions, unsigned int tics = 1)` |
| :--    | :--                                                                         |
| Lua    | `number makeAction(table actions, number tics = 1);`                        |
| Java   | `double makeAction(int[] actions, unsigned int tics = 1);`                  |
| Python | `double make_action(actions, tics = 1);`                                    |

Method combining usability of `setAction`, `advanceAction` and `getLastReward`.
Sets the player's action for the next tics, processes a specified number of tics,
updates the state and calculates a new reward, which is returned.


### `isNewEpisode`

| C++    | `bool isNewEpisode()`    |
| :--    | :--                      |
| Lua    | `boolean isNewEpisode()` |
| Java   | `boolean isNewEpisode()` |
| Python | `bool is_new_episode()`  |

Returns true if the current episode is in the initial state - first state, no actions were performed yet.


### `isEpisodeFinished`

| C++    | `bool isEpisodeFinished()`    |
| :--    | :--                           |
| Lua    | `boolean isEpisodeFinished()` |
| Java   | `boolean isEpisodeFinished()` |
| Python | `bool is_episode_finished()`  |

Returns true if the current episode is in the terminal state (is finished).
`makeAction` and `advanceAction` methods will take no effect after this point (unless `newEpisode` method is called).


### `isPlayerDead`

| C++    | `bool isPlayerDead()`    |
| :--    | :--                      |
| Lua    | `boolean isPlayerDead()` |
| Java   | `boolean isPlayerDead()` |
| Python | `bool is_player_dead()`  |

Returns true if the player is dead state.
In singleplayer player death is equivalent to the end of the episode.
In multiplayer when player is dead `respawnPlayer` can be called.


### `respawnPlayer`

| C++    | `void respawnPlayer()`  |
| :--    | :--                     |
| Lua    | `void respawnPlayer()`  |
| Java   | `void respawnPlayer()`  |
| Python | `void respawn_player()` |

This method respawns player after death in multiplayer mode.
After calling this method, first state after respawn will be available.


### `sendGameCommand`

| C++    | `void sendGameCommand(std::string cmd)` |
| :--    | :--                                     |
| Lua    | `void sendGameCommand(string cmd)`      |
| Java   | `void sendGameCommand(String cmd)`      |
| Python | `void send_game_command(cmd)`           |

Sends the command to Doom console. Can be used for cheats, multiplayer etc.
Some commands will be block in some modes.

See also: 
* ZDoom Wiki: http://zdoom.org/wiki/Console


### `getState`

| C++    | `GameStatePtr (std::shared_ptr<GameState>) GameState getState()` |
| :--    | :--                                                              |
| Lua    | `GameState getState()`                                           |
| Java   | `GameState getState()`                                           |
| Python | `GameState get_state()`                                          |

Changed in 1.1

Returns `GameState` object with the current game state.
If game is not running or episode is finished `nullptr/null/None` will be returned.

See also: 
* Types -> `GameState`


### `getLastAction`

| C++    | `std::vector<int> getLastAction()` |
| :--    | :--                                |
| Lua    | `table getLastAction()`            |
| Java   | `int[] getLastAction()`            |
| Python | `list get_last_action()`           |

Returns the last action performed.
Each value corresponds to a button added with `addAvailableButton` (in order of appearance).
Most useful in `SPECTATOR` mode.


### `getEpisodeTime`

| C++    | `unsigned int getEpisodeTime()`   |
| :--    | :--                               |
| Lua    | `unsigned int getEpisodeTime()`   |
| Java   | `unsigned int getEpisodeTime()`   |
| Python | `unsigned int get_episode_time()` |

Returns number of current episode tic.


## Buttons settings methods
---

### `addAvailableButton`

| C++    | `void addAvailableButton(Button button, unsigned int maxValue = 0)` |
| :--    | :--                                                                 |
| Lua    | `void addAvailableButton(Button button, number maxValue  = 0)`      |
| Java   | `void addAvailableButton(Button button, unsigned int maxValue = 0)` |
| Python | `void add_available_button(Button button, int maxValue = 0)`        |

Add `Button` type (e.g. `TURN_LEFT`, `MOVE_FORWARD`) to `Buttons` available in action
and sets the maximum allowed, absolute value for the specified button.
If the given button has already been added, it will not be added again but the maximum value is overridden.

Config key: availableButtons / available_buttons (list)

See also:
* Types -> `Button`
* setButtonMaxValue
* ConfigFile -> List


### `clearAvailableButtons`

| C++    | `void clearAvailableButtons()`   |
| :--    | :--                              |
| Lua    | `void clearAvailableButtons()`   |
| Java   | `void clearAvailableButtons()`   |
| Python | `void clear_available_buttons()` |

Clears all available `Buttons` added so far.


### `getAvailableButtonsSize`

| C++    | `int getAvailableButtonsSize()`    |
| :--    | :--                                |
| Lua    | `number getAvailableButtonsSize()` |
| Java   | `int getAvailableButtonsSize()`    |
| Python | `int get_available_buttons_size()` |

Returns the number of available `Buttons`.


### `setButtonMaxValue`

| C++    | `void setButtonMaxValue(Button button, unsigned int maxValue = 0)` |
| :--    | :--                                                                |
| Lua    | `void setButtonMaxValue(Button button, number maxValue = 0)`       |
| Java   | `void setButtonMaxValue(Button button, unsigned int maxValue = 0)` |
| Python | `void set_button_max_value(Button button, int maxValue = 0)`       |

Sets the maximum allowed, absolute value for the specified button.
Setting maximum value equal to 0 results in no constraint at all (infinity).
This method makes sense only for delta buttons.
Constraints limit applies in all Modes.


### `getButtonMaxValue`

| C++    | `unsigned int getButtonMaxValue(Button button)` |
| :--    | :--                                             |
| Lua    | `number getButtonMaxValue(Button button)`       |
| Java   | `int getButtonMaxValue(Button button)`          |
| Python | `int get_button_max_value(Button button)`       |

Returns the maximum allowed, absolute value for the specified button.


## GameVariables methods
---

### `addAvailableGameVariable`

| C++    | `void addAvailableGameVariable(GameVariable variable)`    |
| :--    | :--                                                       |
| Lua    | `void addAvailableGameVariable(GameVariable variable)`    |
| Java   | `void addAvailableGameVariable(GameVariable variable)`    |
| Python | `void add_available_game_variable(GameVariable variable)` |

Adds the specified `GameVariable` to the list of game variables (e.g. `HEALTH`, `AMMO1`, `ATTACK_READY`)
that are included in the `GameState` returned by `getState` method.

Config key: availableGameVariables / available_game_variables (list)

See also: 
* Types -> `GameVariable`
* ConfigFile -> List

### `clearAvailableGameVariables`

| C++    | `void clearAvailableGameVariables()`    |
| :--    | :--                                     |
| Lua    | `void clearAvailableGameVariables()`    |
| Java   | `void clearAvailableGameVariables()`    |
| Python | `void clear_available_game_variables()` |

Clears the list of available `GameVariables` that are included in the GameState returned by `getState` method.


### `getAvailableGameVariablesSize`

| C++    | `unsigned int getAvailableGameVariablesSize()`     |
| :--    | :--                                                |
| Lua    | `number getAvailableGameVariablesSize()`           |
| Java   | `unsigned int getAvailableGameVariablesSize()`     |
| Python | `unsigned int get_available_game_variables_size()` |

Returns the number of available `GameVariables`.


### `getGameVariable`

| C++    | `int getGameVariable(GameVariable variable)` |
| :--    | :--                                          |
| Lua    | `int getGameVariable(GameVariable var)`      |
| Java   | `int getGameVariable(GameVariable var)`      |
| Python | `int get_game_variable(GameVariable var)`    |

Returns the current value of the specified game variable (`HEALTH`, `AMMO1` etc.).
The specified game variable does not need to be among available game variables (included in the state).
It could be used for e.g. shaping. Returns 0 in case of not finding given `GameVariable`.

See also: 
* Types -> `GameVariable`


Game Arguments methods
---

### `addGameArgs`

| C++    | `void addGameArgs(std::string args)` |
| :--    | :--                                  |
| Lua    | `void addGameArgs(string args)`      |
| Java   | `void addGameArgs(String args)`      |
| Python | `void add_game_args(args)`           |

Adds a custom argument that will be passed to ViZDoom process during initialization.

Config key: gameArgs / game_args

See also:
* ZDoom Wiki: http://zdoom.org/wiki/Command_line_parameters
* ZDoom Wiki: http://zdoom.org/wiki/CVARS


### `clearGameArgs`

| C++    | `void clearGameArgs()`   |
| :--    | :--                      |
| Lua    | `void clearGameArgs()`   |
| Java   | `void clearGameArgs()`   |
| Python | `void clear_game_args()` |

Clears all arguments previously added with addGameArgs method.


## Reward methods
---
         
### `getLivingReward`

| C++    | `double getLivingReward()`   |
| :--    | :--                          |
| Lua    | `number getLivingReward()`   |
| Java   | `double getLivingReward()`   |
| Python | `double get_living_reward()` |

Returns the reward granted to the player after every tic.


### `setLivingReward`

| C++    | `void setLivingReward(double livingReward)`  |
| :--    | :--                                          |
| Lua    | `void setLivingReward(number livingReward)`  |
| Java   | `void setLivingReward(double livingReward)`  |
| Python | `void set_living_reward(float livingReward)` |

Sets the reward granted to the player after every tic. A negative value is also allowed.

Default value: 0

Config key: livingReward / living_reward


### `getDeathPenalty`

| C++    | `double getDeathPenalty()`   |
| :--    | :--                          |
| Lua    | `double getDeathPenalty()`   |
| Java   | `double getDeathPenalty()`   |
| Python | `double get_death_penalty()` |

Returns the penalty for player's death.


### `setDeathPenalty`

| C++    | `void setDeathPenalty(double deathPenalty)`  |
| :--    | :--                                          |
| Lua    | `void setDeathPenalty(number deathPenalty)`  |
| Java   | `void setDeathPenalty(double deathPenalty)`  |
| Python | `void set_death_penalty(float deathPenalty)` |

Sets a penalty for player's death. Note that in case of a negative value, the player will be rewarded upon dying.

Default value: 0

Config key: deathPenalty / death_penalty


### `getLastReward`

| C++    | `double getLastReward()`  |
| :--    | :--                       |
| Lua    | `number getLastReward()`  |
| Java   | `double getLastReward()`  |
| Python | `float get_last_reward()` |

Returns a reward granted after last update of state.


### `getTotalReward`

| C++    | `double getTotalReward()`  |
| :--    | :--                        |
| Lua    | `number getTotalReward()`  |
| Java   | `double getTotalReward()`  |
| Python | `float get_total_reward()` |

Returns the sum of all rewards gathered in the current episode.


## General game setting methods
---
         
### `loadConfig`

| C++    | `bool loadConfig(std::string filePath)` |
| :--    | :--                                     |
| Lua    | `boolean loadConfig(string filePath)`   |
| Java   | `boolean loadConfig(String filePath)`   |
| Python | `bool load_config(str filePath)`        |

Loads configuration (resolution, available buttons, game variables etc.) from a configuration file.
In case of multiple invocations, older configurations will be overwritten by the recent ones.
Overwriting does not involve resetting to default values, thus only overlapping parameters will be changed.
The method returns true if the whole configuration file was correctly read and applied,
false if file was contained errors.


### `getMode`

| C++    | `Mode getMode()`  |
| :--    | :--               |
| Lua    | `Mode getMode()`  |
| Java   | `Mode getMode()`  |
| Python | `Mode get_mode()` |

Returns current mode.


### `setMode`

| C++    | `void setMode(Mode mode)`  |
| :--    | :--                        |
| Lua    | `void setMode(Mode mode)`  |
| Java   | `void setMode(Mode mode)`  |
| Python | `void set_mode(Mode mode)` |

Sets mode (`PLAYER`, `SPECTATOR`, `ASYNC_PLAYER`, `ASYNC_SPECTATOR`) in which the game will be running.

Default value: `PLAYER`.

Config key: mode

See also: 
* Types -> `Mode`


### `getTicrate`

| C++    | `unsigned int getTicrate()` |
| :--    | :--                         |
| Lua    | `number getTicrate()`       |
| Java   | `unsigned int getTicrate()` |
| Python | `int get_ticrate()`         |

Added in 1.1

Returns current ticrate.


### `setTicrate`

| C++    | `void setTicrate(unsigned int ticrate)` |
| :--    | :--                                     |
| Lua    | `void setTicrate(number ticrate)`       |
| Java   | `void setTicrate(unsigned int ticrate)` |
| Python | `void set_ticrate(int ticrate)`         |

Added in 1.1

Sets ticrate for ASNYC Modes - number of tics executed per second.

Default value: 35 (default Doom ticrate).

Config key: ticrate

See also:
* exmaples/python/ticrate.py


### `setViZDoomPath`

| C++    | `void setViZDoomPath(std::string filePath)` |
| :--    | :--                                         |
| Lua    | `void setViZDoomPath(string filePath)`      |
| Java   | `void setViZDoomPath(String filePath)`      |
| Python | `void set_vizdoom_path(str filePath)`       |

Sets path to ViZDoom engine executable.

Default value: "vizdoom", "vizdoom.exe" on Windows.

Config key: ViZDoomPath / vizdoom_path


### `setDoomGamePath`

| C++    | `void setDoomGamePath(std::string filePath)` |
| :--    | :--                                          |
| Lua    | `void setDoomGamePath(string filePath)`      |
| Java   | `void setDoomGamePath(String filePath)`      |
| Python | `void set_doom_game_path(str filePath)`      |

Sets path to the Doom engine based game file (wad format).

Default value: "doom2.wad"

Config key: DoomGamePath / doom_game_path


### `setDoomScenarioPath`

| C++    | `void setDoomScenarioPath(std::string filePath)` |
| :--    | :--                                              |
| Lua    | `void setDoomScenarioPath(string filePath)`      |
| Java   | `void setDoomScenarioPath(String filePath)`      |
| Python | `void set_doom_scenario_path(str filePath)`      |

Sets path to additional scenario file (wad format).

Default value: ""

Config key: DoomScenarioPath / set_doom_scenario_path


### `setDoomMap`

| C++    | `void setDoomMap(std::string map)` |
| :--    | :--                                |
| Lua    | `void setDoomMap(string map)`      |
| Java   | `void setDoomMap(String map)`      |
| Python | `void set_doom_map(str map)`       |

Sets the map name to be used.

Default value: "map01", if set to empty "map01" will be used.

Config key: DoomMap / doom_map


### `setDoomSkill`

| C++    | `void setDoomSkill(int skill)`    |
| :--    | :--                               |
| Lua    | `void setDoomSkill(number skill)` |
| Java   | `void setDoomSkill(int skill)`    |
| Python | `void setDoomSkill(int skill)`    |

Sets Doom game difficulty level which is called skill in Doom.
The higher is the skill the harder the game becomes.
Skill level affects monsters' aggressiveness, monsters' speed, weapon damage, ammunition quantities etc.
Takes effect from the next episode.

* 1 - VERY EASY, “I'm Too Young to Die” in Doom.
* 2 - EASY, “Hey, Not Too Rough" in Doom.
* 3 - NORMAL, “Hurt Me Plenty” in Doom.
* 4 - HARD, “Ultra-Violence” in Doom.
* 5 - VERY HARD, “Nightmare!” in Doom.

Default value: 3

Config key: skill


### `setDoomConfigPath`

| C++    | `void setDoomConfigPath(std::string filePath)` |
| :--    | :--                                            |
| Lua    | `void setDoomConfigPath(string filePath)`      |
| Java   | `void setDoomConfigPath(String filePath)`      |
| Python | `void set_doom_config_path(str filePath)`      |

Sets path for ViZDoom engine configuration file.
The file is responsible for configuration of Doom engine itself.
If it doesn't exist, it will be created after vizdoom executable is run.
This method is not needed for most of the tasks and is added for convenience of users with hacking tendencies.

Default value: "", if leave empty "_vizdoom.ini" will be used.

Config key: DoomConfigPath / doom_config_path


### `getSeed`

| C++    | `unsigned int getSeed()` |
| :--    | :--                      |
| Lua    | `number getSeed()`       |
| Java   | `unsigned int getSeed()` |
| Python | `int getSeed()`          |

Return ViZDoom's seed.


### `setSeed`

| C++    | `void setSeed(unsigned int seed)` |
| :--    | :--                               |
| Lua    | `void setSeed(number seed)`       |
| Java   | `void setSeed(unsigned int seed)` |
| Python | `void set_seed(int seed)`         |

Sets the seed of the ViZDoom's RNG that generates seeds (initial state) for episodes.

Default value: randomized in constructor

Config key: seed

See also: 
* examples/python/seed.py


### `getEpisodeStartTime`

| C++    | `unsigned int getEpisodeStartTime()` |
| :--    | :--                                  |
| Lua    | `number getEpisodeStartTime()`       |
| Java   | `unsigned int getEpisodeStartTime()` |
| Python | `int get_episode_start_time()`       |

Returns start delay of every episode in tics.


### `setEpisodeStartTime`

| C++    | `void setEpisodeStartTime(unsigned int tics)` |
| :--    | :--                                           |
| Lua    | `void setEpisodeStartTime(number tics)`       |
| Java   | `setEpisodeStartTime(unsigned int tics)`      |
| Python | `void set_episode_start_time(int tics)`       |

Sets start delay of every episode in tics.
Every episode will effectively start (from the user's perspective) after given number of tics.

Default value: 0

Config key: episodeStartTime / episode_start_time


### `getEpisodeTimeout`

| C++    | `unsigned int getEpisodeTimeout()` |
| :--    | :--                                |
| Lua    | `number getEpisodeTimeout()`       |
| Java   | `unsigned int getEpisodeTimeout()` |
| Python | `int get_episode_timeout()`        |

Returns the number of tics after which the episode will be finished.


### `setEpisodeTimeout`

| C++    | `void setEpisodeTimeout(unsigned int tics)` |
| :--    | :--                                         |
| Lua    | `void setEpisodeTimeout(number tics)`       |
| Java   | `void setEpisodeTimeout(unsigned int tics)` |
| Python | `void set_episode_timeout(int tics)`        |

Sets the number of tics after which the episode will be finished. 0 will result in no timeout.

Config key: episodeTimeout / episode_timeout


## Output/rendering setting methods
------------------------------------------------------------------------------------------------------------

### `setScreenResolution`

| C++    | `void setScreenResolution(ScreenResolution resolution)`   |
| :--    | :--                                                       |
| Lua    | `void setScreenResolution(ScreenResolution resolution)`   |
| Java   | `void setScreenResolution(ScreenResolution resolution)`   |
| Python | `void set_screen_resolution(ScreenResolution resolution)` |

Sets the screen resolution.
Supported resolutions are part of ScreenResolution enumeration (e.g. `RES_320X240`, `RES_640X480`, `RES_1920X1080`).
The buffers as well as content of ViZDoom's display window will be affected.

Default value: `RES_320X240`

Config key: screenResolution / screen_resolution

See also: 
* Types -> `ScreenResolution`


### `getScreenFormat`

| C++    | `ScreenFormat getScreenFormat()`   |
| :--    | :--                                |
| Lua    | `ScreenFormat getScreenFormat()`   |
| Java   | `ScreenFormat getScreenFormat()`   |
| Python | `ScreenFormat get_screen_format()` |

Returns the format of the screen buffer and the automap buffer.


### `setScreenFormat`

| C++    | `void setScreenFormat(ScreenFormat format)`   |
| :--    | :--                                           |
| Lua    | `void setScreenFormat(ScreenFormat format)`   |
| Java   | `void setScreenFormat(ScreenFormat format)`   |
| Python | `void set_screen_format(ScreenFormat format)` |

Sets the format of the screen buffer and the automap buffer.
Supported formats are defined in ScreenFormat enumeration type (e.g. `CRCGCB`, `RGB24`, `GRAY8`).
The format change affects only the buffers so it will not have any effect on the content of ViZDoom's display window.

Default value: `CRCGCB`

Config key: screenFormat / screen_format

See also: 
* Types -> `ScreenFormat`


### `isDepthBufferEnabled`

| C++    | `bool isDepthBufferEnabled()`    |
| :--    | :--                              |
| Lua    | `boolean isDepthBufferEnabled()` |
| Java   | `boolean isDepthBufferEnabled()` |
| Python | `bool isDepthBufferEnabled()`    |

Added in 1.1

Returns true if the depth buffer is enabled.


### `setDepthBufferEnabled`

| C++    | `void setDepthBufferEnabled(bool depthBuffer)`    |
| :--    | :--                                               |
| Lua    | `void setDepthBufferEnabled(boolean depthBuffer)` |
| Java   | `void setDepthBufferEnabled(boolean depthBuffer)` |
| Python | `void set_depth_buffer_enabled(bool depthBuffer)` |

Added in 1.1

Enables rendering of the depth buffer, it will be available in state.

Default value: false

Config key: depthBufferEnabled / depth_buffer_enabled

See also: 
* Types -> `GameState`
* examples/python/buffers.py


### `isLabelsBufferEnabled`

| C++    | `bool isLabelsBufferEnabled()`    |
| :--    | :--                               |
| Lua    | `boolean isLabelsBufferEnabled()` |
| Java   | `boolean isLabelsBufferEnabled()` |
| Python | `bool isLabelsBufferEnabled()`    |

Added in 1.1

Returns true if the labels buffer is enabled.


### `setLabelsBufferEnabled`

| C++    | `void setLabelsBufferEnabled(bool labelsBuffer)`    |
| :--    | :--                                                 |
| Lua    | `void setLabelsBufferEnabled(boolean labelsBuffer)` |
| Java   | `void setLabelsBufferEnabled(boolean labelsBuffer)` |
| Python | `void set_labels_buffer_enabled(bool labelsBuffer)` |

Added in 1.1

Enables rendering of the labels buffer, it will be available in state with vector of `Label`s.

Default value: false

Config key: labelsBufferEnabled / labels_buffer_enabled

See also: 
* Types -> `Label`
* Types -> `GameState`
* examples/python/labels.py
* examples/python/buffers.py


### `isAutomapBufferEnabled`

| C++    | `bool isAutomapBufferEnabled()`    |
| :--    | :--                                |
| Lua    | `boolean isAutomapBufferEnabled()` |
| Java   | `boolean isAutomapBufferEnabled()` |
| Python | `bool isAutomapBufferEnabled()`    |

Added in 1.1

Returns true if the automap buffer is enabled.


### `setAutomapBufferEnabled`

| C++    | `void setAutomapBufferEnabled(bool automapBuffer)`    |
| :--    | :--                                                   |
| Lua    | `void setAutomapBufferEnabled(boolean automapBuffer)` |
| Java   | `void setAutomapBufferEnabled(boolean automapBuffer)` |
| Python | `void set_automap_buffer_enabled(bool automapBuffer)` |

Added in 1.1

Enables rendering of the automap buffer, it will be available in state.

Default value: false

Config key: automapBufferEnabled / automap_buffer_enabled

See also: 
* Types -> `GameState`
* examples/python/buffers.py


### `setAutomapMode`

| C++    | `void setAutomapMode(AutomapMode mode)`   |
| :--    | :--                                       |
| Lua    | `void setAutomapMode(AutomapMode mode)`   |
| Java   | `void setAutomapMode(AutomapMode mode)`   |
| Python | `void set_automap_mode(AutomapMode mode)` |

Added in 1.1

Sets the automap mode (`NORMAL`, `WHOLE`, `OBJECTS`, `OBJECTS_WITH_SIZE`) with determine what will be visible on it.

Default value: `NORMAL`

Config key: automapMode / set_automap_mode

See also:
* Types -> `AutomapMode`

### `setAutomapRotate`

| C++    | `void setAutomapRotate(bool rotate)`    |
| :--    | :--                                     |
| Lua    | `void setAutomapRotate(boolean rotate)` |
| Java   | `void setAutomapRotate(boolean rotate)` |
| Python | `void set_automap_rotate(bool rotate)`  |

Added in 1.1

Determine if the automap will be rotating with player, if false, north always will be at the top of the buffer.

Default value: false

Config key: automapRotate / render_hud


### `setAutomapRenderTextures`

| C++    | `setAutomapRenderTextures(bool textures)`    |
| :--    | :--                                          |
| Lua    | `setAutomapRenderTextures(boolean textures)` |
| Java   | `setAutomapRenderTextures(boolean textures)` |
| Python | `set_automap_render_textures(bool textures)` |

Added in 1.1

Determine if the automap will be textured, showing the floor textures.

Default value: true

Config key: automapRenderTextures / automap_render_textures


### `setRenderHud`

| C++    | `void setRenderHud(bool hud)`    |
| :--    | :--                              |
| Lua    | `void setRenderHud(boolean hud)` |
| Java   | `void setRenderHud(boolean hud)` |
| Python | `void set_render_hud(bool hud)`  |

Determine if hud will be rendered in game.

Default value: false

Config key: renderHud / render_hud


### `setRenderMinimalHud`

| C++    | `void setRenderMinimalHud(bool minHud)`    |
| :--    | :--                                        |
| Lua    | `void setRenderMinimalHud(boolean minHud)` |
| Java   | `void setRenderMinimalHud(boolean minHud)` |
| Python | `void set_render_minimal_hud(bool minHud)` |

Added in 1.1

Determine if minimalistic version of hud will be rendered instead of full hud.

Default value: false

Config key: renderMinimalHud / render_minimal_hud


### `setRenderWeapon`

| C++    | `void setRenderWeapon(bool weapon)`    |
| :--    | :--                                    |
| Lua    | `void setRenderWeapon(boolean weapon)` |
| Java   | `void setRenderWeapon(boolean weapon)` |
| Python | `void set_render_weapon(bool weapon)`  |

Determine if weapon held by player will be rendered in game.

Default value: true

Config key: renderWeapon / render_weapon


### `setRenderCrosshair`

| C++    | `void setRenderCrosshair(bool crosshair)`    |
| :--    | :--                                          |
| Lua    | `void setRenderCrosshair(boolean crosshair)` |
| Java   | `void setRenderCrosshair(boolean crosshair)` |
| Python | `void set_render_crosshair(bool crosshair)`  |

Determine if crosshair will be rendered in game.

Default value: false

Config key: renderCrosshair / render_crosshair


### `setRenderDecals`

| C++    | `void setRenderDecals(bool decals)`    |
| :--    | :--                                    |
| Lua    | `void setRenderDecals(boolean decals)` |
| Java   | `void setRenderDecals(boolean decals)` |
| Python | `void set_render_decals(bool decals)`  |

Determine if decals (marks on the walls) will be rendered in game.

Default value: true

Config key: renderDecals / render_decals


### `setRenderParticles`

| C++    | `void setRenderParticles(bool particles)`    |
| :--    | :--                                          |
| Lua    | `void setRenderParticles(boolean particles)` |
| Java   | `void setRenderParticles(boolean particles)` |
| Python | `void set_render_particles(bool particles)`  |

Determine if particles will be rendered in game.

Default value: true

Config key: renderParticles / render_particles


### `setRenderEffectsSprites`

| C++    | `void setRenderEffectsSprites(bool sprites)`    |
| :--    | :--                                             |
| Lua    | `void setRenderEffectsSprites(boolean sprites)` |
| Java   | `void setRenderEffectsSprites(boolean sprites)` |
| Python | `void set_render_effects_sprites(bool sprites)` |

Added in 1.1

Determine if some effects sprites (gun pufs, blood splats etc.) will be rendered in game.

Default value: true

Config key: renderEffectsSprites / render_effects_sprites


### `setRenderMessages`

| C++    | `void setRenderMessages(bool messages)`    |
| :--    | :--                                        |
| Lua    | `void setRenderMessages(boolean messages)` |
| Java   | `void setRenderMessages(boolean messages)` |
| Python | `void set_render_messages(bool messages`   |

Added in 1.1

Determine if ingame messages (information about pickups, kills etc.) will be rendered in game.

Default value: true

Config key: renderMessages / render_messages


### `setWindowVisible`

| C++    | `void setWindowVisible(bool visibility)`    |
| :--    | :--                                         |
| Lua    | `void setWindowVisible(boolean visibility)` |
| Java   | `void setWindowVisible(boolean visibility)` |
| Python | `void set_window_visible(bool visibility)`  |

Determines if ViZDoom's window will be visible.
ViZDoom with window disabled can be used on Linux system without X Server.

Default value: false

Config key: windowVisible / window_visible


### `setConsoleEnabled`

| C++    | `void setConsoleEnabled(bool console)`    |
| :--    | :--                                       |
| Lua    | `void setConsoleEnabled(boolean console)` |
| Java   | `void setConsoleEnabled(boolean console)` |
| Python | `void set_console_enabled(bool console)`  |

Determines if ViZDoom's console output will be enabled.

Default value: false

Config key: consoleEnabled / console_enabled


### `setSoundEnabled`

| C++    | `void setSoundEnabled(bool sound)`    |
| :--    | :--                                   |
| Lua    | `void setSoundEnabled(boolean sound)` |
| Java   | `void setSoundEnabled(boolean sound)` |
| Python | `void set_sound_enabled(bool sound)`  |

Determines if ViZDoom's sound will be played.

Default value: false

Config key: soundEnabled / sound_enabled


### `getScreenWidth`

| C++    | `int getScreenWidth()`   |
| :--    | :--                      |
| Lua    | `int getScreenWidth()`   |
| Java   | `int getScreenWidth()`   |
| Python | `int get_screen_width()` |

Returns game's screen width - width of all buffers.


### `getScreenHeight`

| C++    | `int getScreenHeight()`   |
| :--    | :--                       |
| Lua    | `int getScreenHeight()`   |
| Java   | `int getScreenHeight()`   |
| Python | `int get_screen_height()` |

Returns game's screen height - height of all buffers.


### `getScreenChannels`

| C++    | `int getScreenChannels()`   |
| :--    | :--                         |
| Lua    | `int getScreenChannels()`   |
| Java   | `int getScreenChannels()`   |
| Python | `int get_screen_channels()` |

Returns number of channels in screen buffer and map buffer (depth and labels buffer have always one channel).


### `getScreenPitch`

| C++    | `size_t getScreenPitch()`   |
| :--    | :--                         |
| Lua    | `size_t getScreenPitch()`   |
| Java   | `size_t getScreenPitch()`   |
| Python | `size_t get_screen_pitch()` |

Returns size in bytes of one row in screen buffer and map buffer.


### `getScreenSize`

| C++    | `size_t getScreenSize()` |
| :--    | :--                      |
| Lua    | `number getScreenSize()` |
| Java   | `int getScreenSize()`    |
| Python | `int get_screen_size()`  |

Returns size in bytes of screen buffer and map buffer.
