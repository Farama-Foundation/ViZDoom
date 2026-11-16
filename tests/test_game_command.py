# Tests console commands, argument handling, command length limits, and recording paths.

import os
import tempfile
from pathlib import Path

import pytest

import vizdoom as vzd


@pytest.fixture
def game_dir(monkeypatch):
    # Keep long demo filenames below Windows MAX_PATH, including the directory.
    # pytest's per-test directory names can push a 180-character filename over it.
    with tempfile.TemporaryDirectory(prefix="vzd-") as directory:
        with monkeypatch.context() as context:
            context.chdir(directory)
            yield Path(directory)


@pytest.fixture
def game(game_dir):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    # Nightmare difficulty blocks inventory commands such as give and take.
    game.set_doom_skill(3)
    game.set_window_visible(False)
    game.set_episode_timeout(20)
    try:
        game.init()
        yield game
    finally:
        game.close()


@pytest.mark.parametrize("command", ["give Clip 7", 'give "Clip" 7'])
def test_command_with_arguments(game, command):
    ammo = game.get_game_variable(vzd.GameVariable.AMMO2)

    game.send_game_command(command)
    game.advance_action()

    assert game.get_game_variable(vzd.GameVariable.AMMO2) == ammo + 7


@pytest.mark.parametrize("combine_commands", [False, True])
def test_commands_execute_in_order(game, combine_commands):
    commands = ["take ammo", "give Clip 7", "take Clip 2"]
    if combine_commands:
        commands = ["; ".join(commands)]

    for command in commands:
        game.send_game_command(command)
    game.advance_action()

    assert game.get_game_variable(vzd.GameVariable.AMMO2) == 5


@pytest.mark.parametrize("command", ["", "   ", "vizdoom_nonexistent_test_command"])
def test_empty_or_unknown_command_keeps_game_usable(game, command):
    ammo = game.get_game_variable(vzd.GameVariable.AMMO2)
    episode_time = game.get_episode_time()

    game.send_game_command(command)
    game.advance_action()

    assert game.get_state() is not None
    assert game.get_episode_time() == episode_time + 1
    assert game.get_game_variable(vzd.GameVariable.AMMO2) == ammo

    game.send_game_command("give Clip 7")
    game.advance_action()
    assert game.get_game_variable(vzd.GameVariable.AMMO2) == ammo + 7


@pytest.mark.parametrize("length", [127, 128, 129, 1023])
def test_command_length_boundary(game, length):
    # Keep the effect at the end so truncation cannot silently pass this test.
    command = " " * (length - len("kill")) + "kill"
    game.send_game_command(command)
    game.advance_action()
    assert game.is_player_dead()


@pytest.mark.parametrize("length", [1024, 1025])
@pytest.mark.parametrize("method", ["send_game_command", "new_episode", "save", "load"])
def test_oversized_command_raises_and_game_recovers(game, method, length):
    with pytest.raises(vzd.ViZDoomErrorException, match="maximum length of 1023 bytes"):
        getattr(game, method)("x" * length)

    game.new_episode()
    game.advance_action()
    assert game.get_state() is not None


@pytest.mark.parametrize("path_length", [112, 113, 119, 120, 180])
def test_long_recording_and_replay_path(game, game_dir, path_length):
    path = "a" * (path_length - len(".lmp")) + ".lmp"
    game.new_episode(path)
    game.make_action([0, 0, 1], 20)
    reward = game.get_total_reward()
    game.new_episode()
    assert (game_dir / path).stat().st_size > 0

    game.replay_episode(path)
    assert game.is_replaying_episode()
    for _ in range(30):
        if game.is_episode_finished():
            break
        game.advance_action()
    assert game.is_episode_finished()
    assert game.get_total_reward() == reward
