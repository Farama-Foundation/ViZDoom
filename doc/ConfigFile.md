# Configuration files
Instead of configuring the ViZDoom in code, you can load it from configuration file(s). Each file is read sequentially, so multiple entries with the same key will overwrite previous entries.

## <a name="format"></a> Format
Each entry in a configraution file is a pair of **key** and **value** separated by an equal sign (**"="**). The file format should also abide the following rules:

* one entry per line (except for list parameters),
* case insensitive
* lines starting with **#** are ignored,
* underscores in **keys** are ignored (*episode_timeout* is equivalent to *episodetimeout*),
* string values should **not** be surrounded with apostrophes or quotation marks.

A violation of any of these rules will result in ignoring **only** the line with the error and sending a warning message to stderr (""WARNING! Loading config from: ...").

### <a name="list"></a> List of values
**available_buttons** and **available_game_variables** are special parameters, which use multiple values and instead of a single value they expect a list of values separated by whitespaces and enclosed within braces ("{" and "}"). The list can stretch throughout multiple lines as long as all values are separated from each other by whitespaces.

### <a name="append"></a> Appending values
Each list assignment (**KEY = { VALUES }**)clears values specified for this key before (in other configuration files or in the code). That is why the **append operator(*KEY += { VALUES })** is available. This way you can more easily combine multiple configuration files and tinker in code.

### <a name="config_keys"></a> Supported configuration keys:
* `automapBufferEnabled/automap_buffer_enabled`
* `automapMode/set_automap_mode`
* `automapRenderTextures/automap_render_textures`
* `automapRotate/automap_rotate`
* `availableButtons/available_buttons` (list)
* `availableGameVariables/available_game_variables` (list)
* `consoleEnabled/console_enabled`
* `deathPenalty/death_penalty`
* `depthBufferEnabled/depth_buffer_enabled`
* `DoomConfigPath/doom_config_path`
* `DoomGamePath/doom_game_path`
* `DoomMap/doom_map`
* `DoomScenarioPath/set_doom_scenario_path`
* `DoomSkill/doom_skill`
* `episodeStartTime/episode_start_time`
* `episodeTimeout/episode_timeout`
* `gameArgs/game_args`
* `labelsBufferEnabled/labels_buffer_enabled`
* `livingReward/living_reward`
* `mode`
* `renderCrosshair/render_crosshair`
* `renderDecals/render_decals`
* `renderEffectsSprites/render_effects_sprites`
* `renderHud/render_hud`
* `renderMessages/render_messages`
* `renderMinimalHud/render_minimal_hud`
* `renderParticles/render_particles`
* `renderWeapon/render_weapon`
* `screenFormat/screen_format`
* `screenResolution/screen_resolution`
* `seed`
* `soundEnabled/sound_enabled`
* `ticrate`
* `ViZDoomPath/vizdoom_path`
* `windowVisible/window_visible`


### <a name="sample_config"></a>Sample configuration file content:

```ini
vizdoom_path = ../../bin/vizdoom
#doom_game_path = ../../scenarios/doom2.wad
doom_game_path = ../../scenarios/freedoom2.wad
doom_scenario_path = ../../scenarios/basic.wad
doom_map = map01

# Rewards
living_reward = -1

# Rendering options
screen_resolution = RES_320X240
screen_format = CRCGCB
render_hud = True
render_crosshair = false
render_weapon = true
render_decals = false
render_particles = false
window_visible = true

# make episodes start after 14 tics (after unholstering the gun)
episode_start_time = 14

# make episodes finish after 300 actions (tics)
episode_timeout = 300

# Available buttons
available_buttons = 
	{ 
		MOVE_LEFT 
		MOVE_RIGHT 
		ATTACK 
	}

# Game variables that will be in the state
available_game_variables = { AMMO2}

# Default mode - game is controlled from the code
mode = PLAYER

# Difficulty of gameplay ranging from 1 (baby) to 5 (nightmare)
doom_skill = 5

```

Other examples of configuration files can be found [here](../scenarios)
