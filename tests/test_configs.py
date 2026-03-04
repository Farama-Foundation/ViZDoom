#!/usr/bin/env python3

# Tests ViZDoom load_config method and all the config files from the scenario directory.
# This test can be run as Python script or via PyTest

import os

import vizdoom as vzd


config_values = {
    # "ammo_reward": 0.1,
    "audio_buffer_enabled": True,
    "audio_buffer_size": 8,
    "audio_sampling_rate": vzd.SamplingRate.SR_44100,
    "automap_buffer_enabled": True,
    "automap_mode": vzd.AutomapMode.OBJECTS,
    "automap_render_textures": False,
    "automap_render_objects_as_sprites": True,
    "automap_rotate": True,
    "available_buttons": [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT],
    "available_game_variables": [vzd.GameVariable.AMMO2, vzd.GameVariable.HEALTH],
    "console_enabled": True,
    "damage_made_reward": 0.1,
    "damage_taken_penalty": 0.1,
    "damage_taken_reward": -0.1,
    "death_penalty": 100,
    "death_reward": -100,
    "depth_buffer_enabled": True,
    "doom_config_path": "test/path/config.cfg"
    if os.name != "nt"
    else "test\\path\\config.cfg",
    "doom_game_path": "test/path/doom.wad"
    if os.name != "nt"
    else "test\\path\\doom.wad",
    "doom_map": "map02",
    "doom_scenario_path": "test/path/scenario.cfg"
    if os.name != "nt"
    else "test\\path\\scenario.cfg",
    "doom_skill": 4,
    "episode_start_time": 2,
    "episode_timeout": 10,
    "frag_reward": 10,
    "game_args": "-fast -respawn",
    "health_reward": 0.1,
    "hit_reward": 0.5,
    "hit_taken_penalty": 0.5,
    "hit_taken_reward": -0.5,
    "item_reward": 1,
    "kill_reward": 5,
    "labels_buffer_enabled": True,
    "living_reward": 2,
    "mode": vzd.Mode.PLAYER,
    "notifications_buffer_enabled": True,
    "notifications_buffer_size": 8,
    "objects_info_enabled": True,
    "render_all_frames": True,
    "render_corpses": False,
    "render_crosshair": True,
    "render_decals": False,
    "render_effects_sprites": False,
    "render_hud": True,
    "render_messages": True,
    "render_minimal_hud": True,
    "render_particles": False,
    "render_screen_flashes": False,
    "render_weapon": False,
    "screen_format": vzd.ScreenFormat.BGR24,
    "screen_resolution": vzd.ScreenResolution.RES_640X480,
    "secret_reward": 50,
    "sectors_info_enabled": True,
    "seed": 1993,
    "sound_enabled": True,
    "ticrate": 70,
    "vizdoom_path": "test/path/vizdoom" if os.name != "nt" else "test\\path\\vizdoom",
    "window_visible": True,
}


def _create_config_string(config_values):
    test_config = ""
    for key, value in config_values.items():
        if isinstance(value, list):
            value = [v.name if hasattr(v, "name") else v for v in value]
            value = "{" + " ".join(value) + "}"
        else:
            if hasattr(value, "name"):
                value = value.name
        test_config += f"{key} = {value}\n"

    return test_config


def _test_if_config_set(game, config_values):
    for key, value in config_values.items():
        # Check if set method exists
        setter = f"set_{key}"
        assert hasattr(
            game, setter
        ), f"Config key {key} does not have a setter method ({setter})"

        if isinstance(value, bool):
            getter = f"is_{key}"
        else:
            getter = f"get_{key}"

        if not hasattr(game, getter):
            print(f"Skipping {key} as there is no getter method ({getter})")
            continue  # Skip if the getter does not exist

        getter_value = getattr(game, getter)()
        print(f"Testing {key}: {getter_value} == {value}")
        assert getter_value == value or str(getter_value) in str(
            value
        ), f"Config value for {key} does not match: expected {value}, got {getattr(game, getter)()}"


def _test_load_config(config_values_dict, remove_underscores):
    print("Testing loading all config keys from a config file ...")

    if remove_underscores:
        _config_values_dict = {
            key.replace("_", ""): value for key, value in config_values_dict.items()
        }
    else:
        _config_values_dict = config_values_dict.copy()

    # Create temporary config file
    with open("test_configs.cfg", "w") as f:
        f.write(_create_config_string(_config_values_dict))

    game = vzd.DoomGame()
    success = game.load_config("test_configs.cfg")
    assert success, "load_config should return True for valid config file"

    _test_if_config_set(game, config_values)

    os.remove("test_configs.cfg")


def _test_set_config(config_values_dict, remove_underscores):
    print("Testing setting all config keys ...")

    if remove_underscores:
        _config_values_dict = {
            key.replace("_", ""): value for key, value in config_values_dict.items()
        }
    else:
        _config_values_dict = config_values_dict.copy()

    # As dictionary
    game = vzd.DoomGame()
    success = game.set_config(_config_values_dict)
    assert success, "set_config should return True for valid config dict"
    _test_if_config_set(game, config_values_dict)

    # Create config string
    game = vzd.DoomGame()
    test_config_str = _create_config_string(_config_values_dict)
    success = game.set_config(test_config_str)
    assert success, "set_config should return True for valid config string"

    _test_if_config_set(game, config_values_dict)


def test_load_config():
    _test_load_config(config_values, False)
    _test_load_config(config_values, True)


def test_set_config():
    _test_set_config(config_values, False)
    _test_set_config(config_values, True)


def test_scenario_configs():
    print("Testing load_config() and default scenarios ...")

    for file in os.listdir(vzd.scenarios_path):
        print(f"Testing for config file: {vzd.scenarios_path}/{file}")
        if file.endswith(".cfg"):
            game = vzd.DoomGame()

            # Both should work
            success = game.load_config(os.path.join(vzd.scenarios_path, file))
            assert (
                success
            ), f"Failed to load config file: {os.path.join(vzd.scenarios_path, file)}"

            success = game.load_config(file)
            assert success, f"Failed to load config file: {file}"

            w = game.get_screen_width()
            h = game.get_screen_height()
            assert (
                w == 320 and h == 240
            ), f"Config file {file} is using non-default screen resolution: {w}x{h} instead 320x240."


if __name__ == "__main__":
    test_set_config()
    test_load_config()
    test_scenario_configs()
