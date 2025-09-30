# Configuration files

Instead of configuring the ViZDoom in code, you can load it from the configuration file(s). Each file is read sequentially, so multiple entries with the same key will overwrite previous entries.

## Format
Each entry in a configraution file is a pair of **key** and **value** separated by an equal sign (**`=`**). The file format should also abide the following rules:

* one entry per line (except for list parameters),
* case insensitive
* lines starting with **#** are ignored,
* underscores in **keys** are ignored (*episode_timeout* is equivalent to *episodetimeout*),
* string values should **not** be surrounded with apostrophes or quotation marks.

A violation of any of these rules will result in ignoring **only** the line with the error and sending a warning message to stderr ("WARNING! Loading config from: ...").

### List of values
**available_buttons** and **available_game_variables** are special parameters, which use multiple values and instead of a single value they expect a list of values separated by whitespaces and enclosed within braces (`{` and `}`). The list can stretch throughout multiple lines as long as all values are separated from each other by whitespaces.

### Appending values
Each list assignment (**`KEY = { VALUES }`**)clears values specified for this key before (in other configuration files or in the code). That is why the **append operator (`KEY += { VALUES }`)** is available. This way you can more easily combine multiple configuration files and tinker in code.

### Supported configuration keys:
* `audioBufferEnabled/audio_buffer_enabled`
* `audioBufferSize/audio_buffer_size`
* `audioSamplingRate/audio_samping_rate`
* `automapBufferEnabled/automap_buffer_enabled`
* `automapMode/automap_mode`
* `automapRenderTextures/automap_render_textures`
* `automapRotate/automap_rotate`
* `availableButtons/available_buttons` (list of values)
* `availableGameVariables/available_game_variables` (list of values)
* `consoleEnabled/console_enabled`
* `deathPenalty/death_penalty`
* `deathReward/death_reward`
* `depthBufferEnabled/depth_buffer_enabled`
* `killReward/kill_reward`
* `DoomConfigPath/doom_config_path`
* `DoomGamePath/doom_game_path`
* `DoomMap/doom_map`
* `DoomScenarioPath/doom_scenario_path`
* `DoomSkill/doom_skill`
* `episodeStartTime/episode_start_time`
* `episodeTimeout/episode_timeout`
* `fragReward/frags_reward`
* `gameArgs/game_args`
* `healthReward/health_reward`
* `hitReward/hit_reward`
* `hitTakenPenalty/hit_taken_penalty`
* `hitTakenReward/hit_taken_reward`
* `itemReward/item_reward`
* `killReward/kill_reward`
* `labelsBufferEnabled/labels_buffer_enabled`
* `livingReward/living_reward`
* `mode`
* `notificationsBufferEnabled/notifications_buffer_enabled`
* `notificationsBufferSize/notifications_buffer_size`
* `objectsInfoEnabled/objects_info_enabled`
* `renderAllFrames/render_all_frames`
* `renderCorpses/render_corpses`
* `renderCrosshair/render_crosshair`
* `renderDecals/render_decals`
* `renderEffectsSprites/render_effects_sprites`
* `renderScreenFlashes/render_screen_flashes`
* `renderHud/render_hud`
* `renderMessages/render_messages`
* `renderMinimalHud/render_minimal_hud`
* `renderParticles/render_particles`
* `renderWeapon/render_weapon`
* `screenFormat/screen_format`
* `screenResolution/screen_resolution`
* `sectorsInfoEnabled/sectors_info_enabled`
* `seed`
* `soundEnabled/sound_enabled`
* `ticrate`
* `ViZDoomPath/vizdoom_path`
* `windowVisible/window_visible`

Config keys are also mentioned for related methods in the documentation for `DoomGame` class.


### Sample configuration file content:

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
render_hud = true
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
available_game_variables = { AMMO2 }

# Default mode - the game is controlled from the code
mode = PLAYER

# Difficulty of gameplay ranging from 1 (baby) to 5 (nightmare)
doom_skill = 5
```

Other examples of configuration files can be found in [scenarios](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios) directory.
