# DoomGame

TODO:
- Link to ZDoom wiki pages, examples and other doc files.
- Add new methods

## Flow control methods
---

### init

|C++|bool init()|
|:--|:--|
|Lua|boolean init()|
|Java|boolean init()|
|Python|bool init()|

Initializes ViZDoom game instance and starts newEpisode.
After calling this method, first state from new episode will be available.
Some configuration options cannot be changed after calling method.
Init returns true when the game was started properly and false otherwise.


### close

|C++|void close()|
|:--|:--|
|Lua|void close()|
|Java|void close()|
|Python|void close()|
        
Closes ViZDoom game instance.
It is automatically invoked by the destructor.
Game can be initialized again after being closed.


### newEpisode

|C++|void newEpisode(std::string filePath = "")|
|:--|:--|
|Lua|void newEpisode(string filePath = "")|
|Java|void newEpisode(String filePath = "")|
|Python|void new_episode(str filePath = "")|

Initializes a new episode. All rewards, variables and state are restarted.
After calling this method, first state from new episode will be available.
If the filePath is not empty, given episode will be recorded to this file.

In multiplayer game, host can call this method to finish the game.
Then the rest of the players must also call this method to start a new episode.


### replayEpisode

|C++|void replayEpisode(std::string filePath)|
|:--|:--|
|Lua|void replayEpisode(string filePath)|
|Java|void replayEpisode(String filePath)|
|Python|void replay_episode(str filePath)|

Replays recorded episode from the given filePath.
After calling this method, first state from replay will be available.
All rewards, variables and state are available during replaying episode.

See: record_episodes.py and record_multiplayer.py examples.


### isRunning

|C++|bool isRunning()|
|:--|:--|
|Lua|boolean isRunning()|
|Java|boolean isRunning()|
|Python|bool is_running()|

Checks if the ViZDoom game instance is running.


### setAction

|C++|void setAction(std::vector<int> const &actions)|
|:--|:--|
|Lua|void setAction(table actions)|
|Java|void setAction(int[] actions)|
|Python|void set_action(list actions)|

Sets the player's action for the next tics.
Each value corresponds to a button specified with addAvailableButton method
or in configuration file (in order of appearance).


### advanceAction

|C++|void advanceAction(unsigned int tics = 1, bool updateState = true, bool renderOnly = false)|
|:--|:--|
|Lua|void advanceAction(number tics = 1, boolean updateState = true, boolean renderOnly = false)|
|Java|void advanceAction(unsigned int tics = 1, boolean updateState = true, boolean renderOnly = false)|
|Python|void advance_action(int tics = 1, bool updateState = True, bool renderOnly = False)|

Processes a specified number of tics. If updateState is set the state will be updated after last processed tic
and a new reward will be calculated. To get new state use getState and to get the new reward use getLastReward.
If updateState is not set but renderOnly is turned on, the state will not be updated but a new frame
will be rendered after last processed tic.


### makeAction

|C++|double makeAction(std::vector<int> const &actions, unsigned int tics = 1)|
|:--|:--|
|Lua|number makeAction(table actions, number tics = 1);|
|Java|double makeAction(int[] actions, unsigned int tics = 1);|
|Python|double make_action(actions, tics = 1);|

Method combining usability of setAction, advanceAction and getLastReward.
Sets the player's action for the next tics, processes a specified number of tics,
updates the state and calculates a new reward, which is returned.


### isNewEpisode

|C++|bool isNewEpisode()|
|:--|:--|
|Lua|boolean isNewEpisode()|
|Java|boolean isNewEpisode()|
|Python|bool is_new_episode()|

Returns true if the current episode is in the initial state - first state, no actions were performed yet.


### isEpisodeFinished

|C++|bool isEpisodeFinished()|
|:--|:--|
|Lua|bool isEpisodeFinished()|
|Java|bool isEpisodeFinished()|
|Python|bool is_episode_finished()|

Returns true if the current episode is in the terminal state (is finished).
makeAction and advanceAction methods will take no effect after this point (unless newEpisode method is called).


### isPlayerDead

|C++|bool isPlayerDead()|
|:--|:--|
|Lua|boolean isPlayerDead()|
|Java|boolean isPlayerDead()|
|Python|bool is_player_dead()|

Returns true if the player is dead state.
In singleplayer player death is equivalent to the end of the episode.
In multiplayer when player is dead respawnPlayer can be called.


### respawnPlayer

|C++|void respawnPlayer()|
|:--|:--|
|Lua|void respawnPlayer()|
|Java|void respawnPlayer()|
|Python|void respawn_player()|

This method respawns player after death in multiplayer mode.
After calling this method, first state after respawn will be available.


### sendGameCommand

|C++|void sendGameCommand(std::string cmd)|
|:--|:--|
|Lua|void sendGameCommand(string cmd)|
|Java|void sendGameCommand(String cmd)|
|Python|void send_game_command(cmd)|

Sends the command to Doom console. Can be used for cheats, multiplayer etc.
Some commands will be block in some modes.

For more details consult ZDoom Wiki: http://zdoom.org/wiki/Console


### getState

|C++|GameStatePtr (std::shared_ptr<GameState>) GameState getState()|
|:--|:--|
|Lua|GameState getState()|
|Java|GameState getState()|
|Python|GameState getState()|

Returns GameState object with the current game state.
If game is not running or episode is finished nullptr/null/None will be returned.

See also: GameState.


### getLastAction

|C++|std::vector<int> getLastAction()|
|:--|:--|
|Lua|table getLastAction()|
|Java|int[] getLastAction()|
|Python|list getLastAction()|

Returns the last action performed.
Each value corresponds to a button added with addAvailableButton (in order of appearance).
Most useful in SPECTATOR mode.

### getEpisodeTime

|C++|unsigned int getEpisodeTime()|
|:--|:--|
|Lua|unsigned int getEpisodeTime()|
|Java|unsigned int getEpisodeTime()|
|Python|unsigned int get_episode_time()|

Returns number of current episode tic.


## Buttons settings methods
---

### addAvailableButton(button);

|C++|void addAvailableButton(Button button, unsigned int maxValue = 0)|
|:--|:--|
|Lua|void addAvailableButton(Button button, number maxValue  = 0)|
|Java|void addAvailableButton(Button button, unsigned int maxValue = 0)|
|Python|void addAvailableButton(Button button, int max_value = 0)|

Add Button type (e.g. TURN_LEFT, MOVE_FORWARD) to Buttons available in action
and sets the maximum allowed, absolute value for the specified button.
If the given button has already been added, it will not be added again but the maximum value is overridden.

See also: setButtonMaxValue.


### clearAvailableButtons

|C++|void clearAvailableButtons()|
|:--|:--|
|Lua|void clearAvailableButtons()|
|Java|void clearAvailableButtons()|
|Python|void clear_available_buttons()|

Clears all available Buttons added so far.


### getAvailableButtonsSize

|C++|int getAvailableButtonsSize()|
|:--|:--|
|Lua|number getAvailableButtonsSize()|
|Java|int getAvailableButtonsSize()|
|Python|int get_available_buttons_size()|

Returns the number of available Buttons.


### setButtonMaxValue

|C++|void setButtonMaxValue(Button button, unsigned int maxValue)|
|:--|:--|
|Lua|void setButtonMaxValue(Button button, number maxValue)|
|Java|void setButtonMaxValue(Button button, unsigned int maxValue)|
|Python|void setButtonMaxValue(Button button, int max_value)|

Sets the maximum allowed, absolute value for the specified button.
Setting maximum value equal to 0 results in no constraint at all (infinity).
This method makes sense only for delta buttons.
Constraints limit applies in all Modes.


### getButtonMaxValue

|C++|unsigned int getButtonMaxValue(Button button)|
|:--|:--|
|Lua|number getButtonMaxValue(Button button)|
|Java|int getButtonMaxValue(Button button)|
|Python|int getButtonMaxValue(Button button)|

Returns the maximum allowed, absolute value for the specified button.


## GameVariables methods
---

### addAvailableGameVariable

|C++|void addAvailableGameVariable(GameVariable variable)|
|:--|:--|
|Lua|void addAvailableGameVariable(GameVariable variable)|
|Java|void addAvailableGameVariable(GameVariable variable)|
|Python|void add_vailable_game_variable(GameVariable variable)|

Adds the specified GameVariable to the list of game variables (e.g. AMMO1, HEALTH, ATTACK\_READY)
that are included in the GameState returned by getState method.


### clearAvailableGameVariables();

|C++|void clearAvailableGameVariables()|
|:--|:--|
|Lua|void clearAvailableGameVariables()|
|Java|void clearAvailableGameVariables()|
|Python|void clear_available_game_variables()|

Clears the list of available GameVariables that are included in the GameState returned by getState method.


### getAvailableGameVariablesSize

|C++|unsigned int getAvailableGameVariablesSize()|
|:--|:--|
|Lua|number getAvailableGameVariablesSize()|
|Java|unsigned int getAvailableGameVariablesSize()|
|Python|unsigned int get_available_game_variables_size()|

Returns the number of available GameVariables.


### getGameVariable

|C++|int getGameVariable(GameVariable variable)|
|:--|:--|
|Lua|int getGameVariable(GameVariable var)|
|Java|int getGameVariable(GameVariable var)|
|Python|int getGameVariable(GameVariable var)|

Returns the current value of the specified game variable (AMMO1, HEALTH etc.).
The specified game variable does not need to be among available game variables (included in the state).
It could be used for e.g. shaping. Returns 0 in case of not finding given GameVariable.


Game Arguments methods
---

### addGameArgs

|C++|void addGameArgs(std::string args)|
|:--|:--|
|Lua|void addGameArgs(string args)|
|Java|void addGameArgs(String args)|
|Python|void add_game_args(args)|

Adds a custom argument that will be passed to ViZDoom process during initialization.

For more details consult ZDoom Wiki:
http://zdoom.org/wiki/Command_line_parameters
http://zdoom.org/wiki/CVARS


### clearGameArgs

|C++|void clearGameArgs()|
|:--|:--|
|Lua|void clearGameArgs()|
|Java|void clearGameArgs()|
|Python|void clear_Game_args()|

Clears all arguments previously added with addGameArgs method.


## Reward methods
---
         
### getLivingReward

|C++|double getLivingReward()|
|:--|:--|
|Lua|number getLivingReward()|
|Java|double getLivingReward()|
|Python|double get_living_reward()|

Returns the reward granted to the player after every tic.


### setLivingReward

|C++|void setLivingReward(double livingReward)|
|:--|:--|
|Lua|void setLivingReward(double livingReward)|
|Java|void setLivingReward(double livingReward)|
|Python|void setLivingReward(double livingReward)|

Sets the reward granted to the player after every tic. A negative value is also allowed.


### getDeathPenalty

|C++|double getDeathPenalty()|
|:--|:--|
|Lua|double getDeathPenalty()|
|Java|double getDeathPenalty()|
|Python|double get_death_penalty()|

Returns the penalty for player's death.


void setDeathPenalty(double deathPenalty);

|C++|void setDeathPenalty(double deathPenalty)|
|:--|:--|
|Lua|void setDeathPenalty(double deathPenalty)|
|Java|void setDeathPenalty(double deathPenalty)|
|Python|void setDeathPenalty(double deathPenalty)|

Sets a penalty for player's death. Note that in case of a negative value, the player will be rewarded upon dying.


### getLastReward

|C++|double getLastReward()|
|:--|:--|
|Lua|number getLastReward()|
|Java|double getLastReward()|
|Python|double get_last_reward()|

Returns a reward granted after last update of State.


### getTotalReward

|C++|double getLastReward()|
|:--|:--|
|Lua|number getLastReward()|
|Java|double getLastReward()|
|Python|double get_last_reward()|

Returns the sum of all rewards gathered in the current episode.


## General game setting methods
---
         
### loadConfig

|C++|bool loadConfig(std::string filePath)|
|:--|:--|
|Lua|boolean loadConfig(string filePath)|
|Java|boolean loadConfig(String filePath)|
|Python|bool loadConfig(str filePath)|

Loads configuration (resolution, available buttons, game variables etc.) from a configuration file.
In case of multiple invocations, older configurations will be overwritten by the recent ones.
Overwriting does not involve resetting to default values, thus only overlapping parameters will be changed.
The method returns true if the whole configuration file was correctly read and applied,
false if file was contained errors.


### getMode

|C++|Mode getMode()|
|:--|:--|
|Lua|Mode getMode()|
|Java|Mode getMode()|
|Python|Mode getMode()|

Returns current mode.


### setMode

|C++|void setMode(Mode mode)|
|:--|:--|
|Lua|void setMode(Mode mode)|
|Java|void setMode(Mode mode)|
|Python|void set_mode(Mode mode)|

Sets mode (PLAYER, SPECTATOR, ASYNC_PLAYER, ASYNC_SPECTATOR) in which the game will be started.
Default value: PLAYER.

See: GameMode.

unsigned int getTicrate();

|C++|unsigned int getTicrate()|
|:--|:--|
|Lua|number getTicrate()|
|Java|unsigned int getTicrate()|
|Python|int getTicrate()|

Returns current ticrate.


### setTicrate

|C++|void setTicrate(unsigned int ticrate)|
|:--|:--|
|Lua|void setTicrate(number ticrate)|
|Java|void setTicrate(unsigned int ticrate)|
|Python|void set_ticrate(int ticrate)|

Sets ticrate for ASNYC Modes - number of tics executed per second.

See example: ticrate.py
Default value: 35 (default Doom ticrate)

### setViZDoomPath

|C++|void setViZDoomPath(std::string filePath)|
|:--|:--|
|Lua|void setViZDoomPath(string filePath)|
|Java|void setViZDoomPath(String filePath)|
|Python|void set_vizdoom_path(str filePath)|

Sets path to ViZDoom engine executable.
Default value: "vizdoom", "vizdoom.exe" on Windows.


### setDoomGamePath

|C++|void setDoomGamePath(std::string filePath)|
|:--|:--|
|Lua|void setDoomGamePath(string filePath)|
|Java|void setDoomGamePath(String filePath)|
|Python|void set_doom_game_path(str filePath)|

Sets path to the Doom engine based game file (wad format).
Default value: "doom2.wad"


### setDoomScenarioPath

|C++|void setDoomScenarioPath(std::string filePath)|
|:--|:--|
|Lua|void setDoomScenarioPath(string filePath)|
|Java|void setDoomScenarioPath(String filePath)|
|Python|void set_doom_scenario_path(str filePath)|

Sets path to additional scenario file (wad format).
Default value: ""


### setDoomMap

|C++|void setDoomMap(std::string map)|
|:--|:--|
|Lua|void setDoomMap(string map)|
|Java|void setDoomMap(String map)|
|Python|void set_doom_map(str map)|

Sets the map name to be used.
Default value: "map01", if set to empty "map01" will be used.

         
### setDoomSkill

|C++|void setDoomSkill(int skill)|
|:--|:--|
|Lua|void setDoomSkill(number skill)|
|Java|void setDoomSkill(int skill)|
|Python|void setDoomSkill(int skill)|

Sets Doom game difficulty level which is called skill in Doom.
The higher is the skill the harder the game becomes.
Skill level affects monsters' aggressiveness, monsters' speed, weapon damage, ammunition quantities etc.
Takes effect from the next episode.
Default value: 3

1 - VERY EASY, “I'm Too Young to Die” in Doom.
2 - EASY, “Hey, Not Too Rough" in Doom.
3 - NORMAL, “Hurt Me Plenty” in Doom.
4 - HARD, “Ultra-Violence” in Doom.
5 - VERY HARD, “Nightmare!” in Doom.


### setDoomConfigPath

|C++|void setDoomConfigPath(std::string filePath)|
|:--|:--|
|Lua|void setDoomConfigPath(string filePath)|
|Java|void setDoomConfigPath(String filePath)|
|Python|void set_doom_config_path(str filePath)|

Sets path for ViZDoom engine configuration file.
The file is responsible for configuration of Doom engine itself.
If it doesn't exist, it will be created after vizdoom executable is run.
This method is not needed for most of the tasks and is added for convenience of users with hacking tendencies.
Default value: "", if leave empty "_vizdoom.ini" will be used.


### getSeed

|C++|unsigned int getSeed()|
|:--|:--|
|Lua|number getSeed()|
|Java|unsigned int getSeed()|
|Python|int getSeed()|

Return ViZDoom's seed.


### setSeed

|C++|void setSeed(unsigned int seed)|
|:--|:--|
|Lua|void setSeed(number seed)|
|Java|void setSeed(unsigned int seed)|
|Python|void set_seed(int seed)|

Sets the seed of the ViZDoom's RNG that generates seeds (initial state) for episodes.


### getEpisodeStartTime

|C++|unsigned int getEpisodeStartTime()|
|:--|:--|
|Lua|number getEpisodeStartTime()|
|Java|unsigned int getEpisodeStartTime()|
|Python|int get_episode_start_time()|

Returns start delay of every episode in tics.


### setEpisodeStartTime

|C++|void setEpisodeStartTime(unsigned int tics)|
|:--|:--|
|Lua|void setEpisodeStartTime(number tics)|
|Java|setEpisodeStartTime(unsigned int tics)|
|Python|void set_episode_start_time(int tics)|

Sets start delay of every episode in tics.
Every episode will effectively start (from the user's perspective) after given number of tics.


### getEpisodeTimeout

|C++|unsigned int getEpisodeTimeout()|
|:--|:--|
|Lua|number getEpisodeTimeout()|
|Java|unsigned int getEpisodeTimeout()|
|Python|int get_episode_timeout()|

Returns the number of tics after which the episode will be finished.


### setEpisodeTimeout

|C++|void setEpisodeTimeout(unsigned int tics)|
|:--|:--|
|Lua|void setEpisodeTimeout(number tics)|
|Java|void setEpisodeTimeout(unsigned int tics)|
|Python|void set_episode_timeout(int tics)|

Sets the number of tics after which the episode will be finished. 0 will result in no timeout.


## Output/rendering setting methods
------------------------------------------------------------------------------------------------------------

### setScreenResolution(ScreenResolution resolution);

|C++|void setScreenResolution(ScreenResolution resolution)|
|:--|:--|
|Lua|void setScreenResolution(ScreenResolution resolution)|
|Java|void setScreenResolution(ScreenResolution resolution)|
|Python|void set_screen_resolution(ScreenResolution resolution)|

Sets the screen resolution.
Supported resolutions are part of ScreenResolution enumeration (e.g. RES_320X240, RES_1920X1080).
The buffers as well as content of ViZDoom's display window will be affected.

See also: ScreenResolution

### getScreenFormat()

|C++|ScreenFormat getScreenFormat()|
|:--|:--|
|Lua|ScreenFormat getScreenFormat()|
|Java|ScreenFormat getScreenFormat()|
|Python|ScreenFormat get_screen_format()|

Returns the format of the screen buffer and automap buffer.

### setScreenFormat(ScreenFormat format);

|C++|void setScreenFormat(ScreenFormat format)|
|:--|:--|
|Lua|void setScreenFormat(ScreenFormat format)|
|Java|void setScreenFormat(ScreenFormat format)|
|Python|void set_screen_format(ScreenFormat format)|

Sets the format of the screen buffer and automap buffer.
Supported formats are defined in ScreenFormat enumeration type (e.g. CRCGCB, CRCGCBDB, RGB24, GRAY8).
The format change affects only the buffers so it will not have any effect on the content of ViZDoom's display window.

See also: ScreenFormat


### setDepthBufferEnabled(bool enabled);

|C++|void setDepthBufferEnabled(bool enabled)|
|:--|:--|
|Lua|void setDepthBufferEnabled(bool enabled)|
|Java|void setDepthBufferEnabled(bool enabled)|
|Python|void set_depth_buffer_enabled(bool enabled)|

TODO: expand
Enables rendering of depth buffer.

See also: buffers.py example.


### setLabelsBufferEnabled(bool enabled);

|C++|void setLabelsBufferEnabled(bool enabled)|
|:--|:--|
|Lua|void setLabelsBufferEnabled(bool enabled)|
|Java|void setLabelsBufferEnabled(bool enabled)|
|Python|void set_labels_buffer_enabled(bool enabled)|

TODO: expand
Enables rendering of labels buffer.

See also: buffers.py example.


### setAutomapBufferEnabled(bool enabled);

|C++|void setAutomapBufferEnabled(bool enabled)|
|:--|:--|
|Lua|void setAutomapBufferEnabled(bool enabled)|
|Java|void setAutomapBufferEnabled(bool enabled)|
|Python|void set_automap_buffer_enabled(bool enabled)|

TODO: expand
Enables rendering of automap buffer.

See also: buffers.py example.


### setRenderHud(bool hud);

|C++|void setRenderHud(bool hud)|
|:--|:--|
|Lua|void setRenderHud(bool hud)|
|Java|void setRenderHud(bool hud)|
|Python|void set_render_hud(bool hud)|

Determine if game's hud will be rendered in game.


### setRenderWeapon

|C++|void setRenderWeapon(bool weapon)|
|:--|:--|
|Lua|void setRenderWeapon(bool weapon)|
|Java|void setRenderWeapon(bool weapon)|
|Python|void set_render_weapon(bool weapon)|

Determine if weapon held by player will be rendered in game.


### setRenderCrosshair

|C++|void setRenderCrosshair(bool crosshair)|
|:--|:--|
|Lua|void setRenderCrosshair(bool crosshair)|
|Java|void setRenderCrosshair(bool crosshair)|
|Python|void set_render_crosshair(bool crosshair)|

Determine if crosshair will be rendered in game.


### setRenderDecals

|C++|void setRenderDecals(bool decals)|
|:--|:--|
|Lua|void setRenderDecals(bool decals)|
|Java|void setRenderDecals(bool decals)|
|Python|void set_render_decals(bool decals)|

Determine if decals (marks on the walls) will be rendered in game.


### setRenderParticles

|C++|void setRenderParticles(bool particles)|
|:--|:--|
|Lua|void setRenderParticles(bool particles)|
|Java|void setRenderParticles(bool particles)|
|Python|void set_render_particles(bool particles)|

Determine if particles will be rendered in game.


### setWindowVisible

|C++|void setWindowVisible(bool visibility)|
|:--|:--|
|Lua|void setWindowVisible(bool visibility)|
|Java|void setWindowVisible(bool visibility)|
|Python|void set_window_visible(bool visibility)|

Determines if ViZDoom's window will be visible.
ViZDoom with window disabled can be used on Linux system without X Server.


### setConsoleEnabled

|C++|void setConsoleEnabled(bool console)|
|:--|:--|
|Lua|void setConsoleEnabled(bool console)|
|Java|void setConsoleEnabled(bool console)|
|Python|void set_console_enabled(bool console)|

Determines if ViZDoom's console output will be enabled.


### setSoundEnabled

|C++|void setSoundEnabled(bool sound)|
|:--|:--|
|Lua|void setSoundEnabled(bool sound)|
|Java|void setSoundEnabled(bool sound)|
|Python|void set_sound_enabled(bool sound)|

Determines if ViZDoom's sound will be played.


### getScreenWidth();

|C++|int getScreenWidth()|
|:--|:--|
|Lua|int getScreenWidth()|
|Java|int getScreenWidth()|
|Python|int get_screen_width()|

Returns game's screen width.


### getScreenHeight

|C++|int getScreenHeight()|
|:--|:--|
|Lua|int getScreenHeight()|
|Java|int getScreenHeight()|
|Python|int get_screen_height()|

Returns game's screen height.


### getScreenChannels

|C++|int getScreenChannels()|
|:--|:--|
|Lua|int getScreenChannels()|
|Java|int getScreenChannels()|
|Python|int get_screen_channels()|

Returns number of channels in game's screen and map buffer.

         
### getScreenPitch

|C++|size_t getScreenPitch()|
|:--|:--|
|Lua|size_t getScreenPitch()|
|Java|size_t getScreenPitch()|
|Python|size_t get_screen_pitch()|

Returns size in bytes of one row in game's screen and map buffer.
         
### getScreenSize

|C++|size_t getScreenSize()|
|:--|:--|
|Lua|number getScreenSize()|
|Java|int getScreenSize()|
|Python|int get_screen_size()|

Returns size in bytes of game's screen buffer.
