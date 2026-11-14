import os

import pytest

import vizdoom as vzd


@pytest.fixture
def game(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    game.set_episode_timeout(20)
    try:
        game.init()
        yield game
    finally:
        game.close()


@pytest.mark.parametrize("length", [127, 128, 129, 4095])
def test_command_length_boundary(game, length):
    # Keep the effect at the end so truncation cannot silently pass this test.
    command = " " * (length - len("kill")) + "kill"
    game.send_game_command(command)
    game.advance_action()
    assert game.is_player_dead()


@pytest.mark.parametrize("length", [4096, 4097])
@pytest.mark.parametrize("method", ["send_game_command", "new_episode", "save", "load"])
def test_oversized_command_raises_and_game_recovers(game, method, length):
    with pytest.raises(vzd.ViZDoomErrorException, match="maximum length of 4095 bytes"):
        getattr(game, method)("x" * length)

    game.new_episode()
    game.advance_action()
    assert game.get_state() is not None


@pytest.mark.parametrize("path_length", [112, 113, 119, 120, 180])
def test_long_recording_and_replay_path(game, tmp_path, path_length):
    path = "a" * (path_length - len(".lmp")) + ".lmp"
    game.new_episode(path)
    game.make_action([0, 0, 1], 20)
    reward = game.get_total_reward()
    game.new_episode()
    assert (tmp_path / path).stat().st_size > 0

    game.replay_episode(path)
    assert game.is_replaying_episode()
    for _ in range(30):
        if game.is_episode_finished():
            break
        game.advance_action()
    assert game.is_episode_finished()
    assert game.get_total_reward() == reward
