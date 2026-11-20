# Tests counter, reward, and audio behavior when restoring a saved game.

import os

import numpy as np
import pytest

import vizdoom as vzd


@pytest.fixture
def game(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    game = vzd.DoomGame()
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_doom_skill(3)
    game.set_seed(7)
    game.set_episode_timeout(0)
    game.set_available_buttons([vzd.Button.MOVE_FORWARD])
    # Use the bundled map without monsters or scenario scripts changing the state.
    game.add_game_args("-nomonsters")
    try:
        game.init()
        yield game
    finally:
        game.close()


@pytest.mark.parametrize("reward_per_unit", [-2, 2])
@pytest.mark.parametrize(
    "variable,setting,setup,command,delta",
    [
        (
            vzd.GameVariable.HITCOUNT,
            "hit_reward",
            "summon DoomImp 0 42",
            "special Thing_Damage 42 7",
            1,
        ),
        (
            vzd.GameVariable.DAMAGECOUNT,
            "damage_made_reward",
            "summon DoomImp 0 42",
            "special Thing_Damage 42 7",
            7,
        ),
        (
            vzd.GameVariable.HITS_TAKEN,
            "hit_taken_reward",
            None,
            "special Thing_Damage 0 7",
            1,
        ),
        (
            vzd.GameVariable.DAMAGE_TAKEN,
            "damage_taken_reward",
            None,
            "special Thing_Damage 0 7",
            7,
        ),
    ],
)
def test_load_resets_combat_counters_and_reward_baselines(
    game, variable, setting, setup, command, delta, reward_per_unit
):
    getattr(game, f"set_{setting}")(reward_per_unit)
    expected_reward = reward_per_unit * delta
    if setup:
        game.send_game_command(setup)
        game.make_action([0])
    game.send_game_command(command)
    assert game.make_action([0]) == expected_reward
    game.save("counters.png")
    game.send_game_command(command)
    assert game.make_action([0]) == expected_reward
    assert game.get_game_variable(variable) == 2 * delta
    total_reward = game.get_total_reward()

    for _ in range(2):
        game.load("counters.png")
        assert game.get_game_variable(variable) == 0
        assert game.get_last_reward() == 0
        assert game.get_total_reward() == total_reward
        game.send_game_command(command)
        assert game.make_action([0]) == expected_reward
        assert game.get_game_variable(variable) == delta
        total_reward += expected_reward
        assert game.get_total_reward() == total_reward


def test_death_penalty_after_load(game):
    game.set_death_penalty(3)
    game.save("alive.png")
    game.send_game_command("special Thing_Damage 0 7")
    game.make_action([0])
    game.load("alive.png")
    assert not game.is_player_dead()
    assert game.get_game_variable(vzd.GameVariable.DEATHCOUNT) == 0
    assert game.get_last_reward() == 0
    game.send_game_command("kill")
    assert game.make_action([0]) == -3
    assert game.get_game_variable(vzd.GameVariable.DEATHCOUNT) == 1
    assert game.get_total_reward() == -3


def test_load_resets_audio_buffer_and_renderer(tmp_path):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic_audio.cfg"))
    game.set_window_visible(False)
    game.set_doom_config_path(str(tmp_path / "audio.ini"))
    game.set_audio_buffer_size(4)
    game.set_available_buttons([vzd.Button.MOVE_LEFT])
    game.add_game_args("+snd_efx 1")
    game.set_seed(0)
    try:
        game.init()
        checkpoint = str(tmp_path / "audio.zds")
        game.save(checkpoint)
        trajectories = []
        for tics in (4, 13, 25):
            game.make_action([1], tics)
            game.set_action([0])
            game.load(checkpoint)
            audio = game.get_state().audio_buffer
            assert audio is not None
            # Loading advances one tic. Older buffer entries must be cleared.
            samples_per_tic = int(game.get_audio_sampling_rate()) // game.get_ticrate()
            assert not np.any(audio[:-samples_per_tic])
            trajectory = [audio.copy()]
            for _ in range(3):
                game.make_action([0], 4)
                audio = game.get_state().audio_buffer
                assert audio is not None
                trajectory.append(audio.copy())
            # Require a signal above one-unit quantization/dithering noise.
            assert np.any(
                np.abs(trajectory[-1].astype(np.int32)) > 1
            ), "Saved sound channels must still play"
            trajectories.append(trajectory)
        for trajectory in trajectories[1:]:
            np.testing.assert_array_equal(trajectory, trajectories[0])
    finally:
        game.close()
