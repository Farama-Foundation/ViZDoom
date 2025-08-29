# DoomGame
DoomGame is the main object of the ViZDoom library, representing a single instance of the Doom game and providing the interface for a single agent/player to interact with the game.
The object allows sending actions to the game, getting the game state, etc.

```{eval-rst}
.. autoclass:: vizdoom.DoomGame
```


## Flow control methods

```{eval-rst}
.. autofunction:: vizdoom.DoomGame.init
.. autofunction:: vizdoom.DoomGame.close
.. autofunction:: vizdoom.DoomGame.new_episode
.. autofunction:: vizdoom.DoomGame.replay_episode
.. autofunction:: vizdoom.DoomGame.is_running
.. autofunction:: vizdoom.DoomGame.is_multiplayer_game
.. autofunction:: vizdoom.DoomGame.is_recording_episode
.. autofunction:: vizdoom.DoomGame.is_replaying_episode
.. autofunction:: vizdoom.DoomGame.set_action
.. autofunction:: vizdoom.DoomGame.advance_action
.. autofunction:: vizdoom.DoomGame.make_action
.. autofunction:: vizdoom.DoomGame.is_new_episode
.. autofunction:: vizdoom.DoomGame.is_episode_finished
.. autofunction:: vizdoom.DoomGame.is_episode_timeout_reached
.. autofunction:: vizdoom.DoomGame.is_player_dead
.. autofunction:: vizdoom.DoomGame.respawn_player
.. autofunction:: vizdoom.DoomGame.send_game_command
.. autofunction:: vizdoom.DoomGame.get_state
.. autofunction:: vizdoom.DoomGame.get_server_state
.. autofunction:: vizdoom.DoomGame.get_last_action
.. autofunction:: vizdoom.DoomGame.get_episode_time
.. autofunction:: vizdoom.DoomGame.save
.. autofunction:: vizdoom.DoomGame.load
```

## Buttons settings methods

```{eval-rst}
.. autofunction:: vizdoom.DoomGame.get_available_buttons
.. autofunction:: vizdoom.DoomGame.set_available_buttons
.. autofunction:: vizdoom.DoomGame.add_available_button
.. autofunction:: vizdoom.DoomGame.clear_available_buttons
.. autofunction:: vizdoom.DoomGame.get_available_buttons_size
.. autofunction:: vizdoom.DoomGame.set_button_max_value
.. autofunction:: vizdoom.DoomGame.get_button_max_value
.. autofunction:: vizdoom.DoomGame.get_button
```

## GameVariables methods

```{eval-rst}
.. autofunction:: vizdoom.DoomGame.get_available_game_variables
.. autofunction:: vizdoom.DoomGame.set_available_game_variables
.. autofunction:: vizdoom.DoomGame.add_available_game_variable
.. autofunction:: vizdoom.DoomGame.clear_available_game_variables
.. autofunction:: vizdoom.DoomGame.get_available_game_variables_size
.. autofunction:: vizdoom.DoomGame.get_game_variable
```

## Game arguments methods

```{eval-rst}
.. autofunction:: vizdoom.DoomGame.set_game_args
.. autofunction:: vizdoom.DoomGame.add_game_args
.. autofunction:: vizdoom.DoomGame.clear_game_args
.. autofunction:: vizdoom.DoomGame.get_game_args
```

## Reward methods

```{eval-rst}
.. autofunction:: vizdoom.DoomGame.get_living_reward
.. autofunction:: vizdoom.DoomGame.set_living_reward
.. autofunction:: vizdoom.DoomGame.get_death_penalty
.. autofunction:: vizdoom.DoomGame.set_death_penalty
.. autofunction:: vizdoom.DoomGame.get_death_reward
.. autofunction:: vizdoom.DoomGame.set_death_reward
.. autofunction:: vizdoom.DoomGame.get_map_exit_reward
.. autofunction:: vizdoom.DoomGame.set_map_exit_reward
.. autofunction:: vizdoom.DoomGame.get_kill_reward
.. autofunction:: vizdoom.DoomGame.set_kill_reward
.. autofunction:: vizdoom.DoomGame.get_item_reward
.. autofunction:: vizdoom.DoomGame.set_item_reward
.. autofunction:: vizdoom.DoomGame.get_secret_reward
.. autofunction:: vizdoom.DoomGame.set_secret_reward
.. autofunction:: vizdoom.DoomGame.get_frag_reward
.. autofunction:: vizdoom.DoomGame.set_frag_reward
.. autofunction:: vizdoom.DoomGame.get_hit_reward
.. autofunction:: vizdoom.DoomGame.set_hit_reward
.. autofunction:: vizdoom.DoomGame.get_hit_taken_reward
.. autofunction:: vizdoom.DoomGame.set_hit_taken_reward
.. autofunction:: vizdoom.DoomGame.get_hit_taken_penalty
.. autofunction:: vizdoom.DoomGame.set_hit_taken_penalty
.. autofunction:: vizdoom.DoomGame.get_damage_made_reward
.. autofunction:: vizdoom.DoomGame.set_damage_made_reward
.. autofunction:: vizdoom.DoomGame.get_damage_taken_reward
.. autofunction:: vizdoom.DoomGame.set_damage_taken_reward
.. autofunction:: vizdoom.DoomGame.get_damage_taken_penalty
.. autofunction:: vizdoom.DoomGame.set_damage_taken_penalty
.. autofunction:: vizdoom.DoomGame.get_health_reward
.. autofunction:: vizdoom.DoomGame.set_health_reward
.. autofunction:: vizdoom.DoomGame.get_armor_reward
.. autofunction:: vizdoom.DoomGame.set_armor_reward
.. autofunction:: vizdoom.DoomGame.get_last_reward
.. autofunction:: vizdoom.DoomGame.get_total_reward
```

## General game setting methods

```{eval-rst}
.. autofunction:: vizdoom.DoomGame.load_config
.. autofunction:: vizdoom.DoomGame.get_mode
.. autofunction:: vizdoom.DoomGame.set_mode
.. autofunction:: vizdoom.DoomGame.get_ticrate
.. autofunction:: vizdoom.DoomGame.set_ticrate
.. autofunction:: vizdoom.DoomGame.set_vizdoom_path
.. autofunction:: vizdoom.DoomGame.set_doom_game_path
.. autofunction:: vizdoom.DoomGame.set_doom_scenario_path
.. autofunction:: vizdoom.DoomGame.set_doom_map
.. autofunction:: vizdoom.DoomGame.set_doom_skill
.. autofunction:: vizdoom.DoomGame.set_doom_config_path
.. autofunction:: vizdoom.DoomGame.get_seed
.. autofunction:: vizdoom.DoomGame.set_seed
.. autofunction:: vizdoom.DoomGame.get_episode_start_time
.. autofunction:: vizdoom.DoomGame.set_episode_start_time
.. autofunction:: vizdoom.DoomGame.get_episode_timeout
.. autofunction:: vizdoom.DoomGame.set_episode_timeout
```

## Output/rendering setting methods

```{eval-rst}
.. autofunction:: vizdoom.DoomGame.set_screen_resolution
.. autofunction:: vizdoom.DoomGame.get_screen_format
.. autofunction:: vizdoom.DoomGame.set_screen_format
.. autofunction:: vizdoom.DoomGame.is_depth_buffer_enabled
.. autofunction:: vizdoom.DoomGame.set_depth_buffer_enabled
.. autofunction:: vizdoom.DoomGame.is_labels_buffer_enabled
.. autofunction:: vizdoom.DoomGame.set_labels_buffer_enabled
.. autofunction:: vizdoom.DoomGame.is_automap_buffer_enabled
.. autofunction:: vizdoom.DoomGame.set_automap_buffer_enabled
.. autofunction:: vizdoom.DoomGame.set_automap_mode
.. autofunction:: vizdoom.DoomGame.set_automap_rotate
.. autofunction:: vizdoom.DoomGame.set_automap_render_textures
.. autofunction:: vizdoom.DoomGame.set_render_hud
.. autofunction:: vizdoom.DoomGame.set_render_minimal_hud
.. autofunction:: vizdoom.DoomGame.set_render_weapon
.. autofunction:: vizdoom.DoomGame.set_render_crosshair
.. autofunction:: vizdoom.DoomGame.set_render_decals
.. autofunction:: vizdoom.DoomGame.set_render_particles
.. autofunction:: vizdoom.DoomGame.set_render_effects_sprites
.. autofunction:: vizdoom.DoomGame.set_render_messages
.. autofunction:: vizdoom.DoomGame.set_render_corpses
.. autofunction:: vizdoom.DoomGame.set_render_screen_flashes
.. autofunction:: vizdoom.DoomGame.set_render_all_frames
.. autofunction:: vizdoom.DoomGame.set_window_visible
.. autofunction:: vizdoom.DoomGame.set_console_enabled
.. autofunction:: vizdoom.DoomGame.set_sound_enabled
.. autofunction:: vizdoom.DoomGame.get_screen_width
.. autofunction:: vizdoom.DoomGame.get_screen_height
.. autofunction:: vizdoom.DoomGame.get_screen_channels
.. autofunction:: vizdoom.DoomGame.get_screen_pitch
.. autofunction:: vizdoom.DoomGame.get_screen_size
.. autofunction:: vizdoom.DoomGame.is_objects_info_enabled
.. autofunction:: vizdoom.DoomGame.set_objects_info_enabled
.. autofunction:: vizdoom.DoomGame.is_sectors_info_enabled
.. autofunction:: vizdoom.DoomGame.set_sectors_info_enabled
.. autofunction:: vizdoom.DoomGame.is_audio_buffer_enabled
.. autofunction:: vizdoom.DoomGame.set_audio_buffer_enabled
.. autofunction:: vizdoom.DoomGame.get_audio_sampling_rate
.. autofunction:: vizdoom.DoomGame.set_audio_sampling_rate
.. autofunction:: vizdoom.DoomGame.get_audio_buffer_size
.. autofunction:: vizdoom.DoomGame.set_audio_buffer_size
.. autofunction:: vizdoom.DoomGame.is_notifications_buffer_enabled
.. autofunction:: vizdoom.DoomGame.set_notifications_buffer_enabled
.. autofunction:: vizdoom.DoomGame.get_notifications_buffer_size
.. autofunction:: vizdoom.DoomGame.set_notifications_buffer_size
```
