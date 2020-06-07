using CxxWrap

wrap_modules("../../bin/libvizdoomjl")

const vz = ViZDoomWrapper

game = vz.DoomGame()

vz.set_doom_scenario_path(game, "../../scenarios/basic.wad")

vz.set_doom_map(game, "map01")

# Sets resolution. Default is 320X240
vz.set_screen_resolution(game, vz.RES_640X480)

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
vz.set_screen_format(game, vz.RGB24)

# Enables depth buffer.
vz.set_depth_buffer_enabled(game, true)

# Enables labeling of in vz objects labeling.
vz.set_labels_buffer_enabled(game, true)

# Enables buffer with top down map of the current episode/level.
vz.set_automap_buffer_enabled(game, true)


# Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
vz.set_render_hud(game, false)
vz.set_render_minimal_hud(game, false)  # If hud is enabled
vz.set_render_crosshair(game, false)
vz.set_render_weapon(game, true)
vz.set_render_decals(game, false)  # Bullet holes and blood on the walls
vz.set_render_particles(game, false)
vz.set_render_effects_sprites(game, false)  # Smoke and blood
vz.set_render_messages(game, false)  # In-vz messages
vz.set_render_corpses(game, false)
vz.set_render_screen_flashes(game, true)  # Effect upon taking damage or picking up items


# Adds buttons that will be allowed. 
vz.add_available_button(game, vz.MOVE_LEFT)
vz.add_available_button(game, vz.MOVE_RIGHT)
vz.add_available_button(game, vz.ATTACK)

# Adds vz variables that will be included in state.
vz.add_available_game_variable(game, vz.AMMO2)

# Causes episodes to finish after 200 tics (actions)
vz.set_episode_timeout(game, 200)

# Makes episodes start after 10 tics (~after raising the weapon)
vz.set_episode_start_time(game, 10)

# Makes the window appear (turned on by default)
vz.set_window_visible(game, true)

# Turns on the sound. (turned off by default)
vz.set_sound_enabled(game, true)

# Sets the living reward (for each move) to -1
vz.set_living_reward(game, -1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
vz.set_mode(game, vz.PLAYER)

vz.init(game)

actions = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
episodes = 10
sleep_time = 1.0 / vz.DEFAULT_TICRATE

for i in 1:episodes
    println("Episode #$i")
    vz.new_episode(game)

    while !vz.is_episode_finished(game)
        state = vz.get_state(game)
        r = vz.make_action(game, rand(actions))
        println("Reward $r")
        sleep(sleep_time)
    end
end
