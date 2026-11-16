# Tests recording paths, replay consistency, independent recordings, and finalization.

import os

import numpy as np
import pytest

import vizdoom as vzd


@pytest.fixture
def game(tmp_path, monkeypatch):
    # Short relative paths keep these tests independent of the command length limit.
    monkeypatch.chdir(tmp_path)
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    game.set_seed(123)
    game.set_episode_start_time(1)
    game.set_episode_timeout(40)
    game.set_available_game_variables(
        [
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
        ]
    )
    try:
        game.init()
        yield game
    finally:
        game.close()


@pytest.mark.parametrize("forward_slashes", [False, True])
@pytest.mark.parametrize("with_extension", [False, True])
def test_record_and_replay_path_with_spaces(
    game, tmp_path, forward_slashes, with_extension
):
    demo_dir = tmp_path / "my demos"
    demo_dir.mkdir()
    path = os.path.join("my demos", "center.lmp")
    if forward_slashes:
        path = path.replace("\\", "/")

    game.set_episode_timeout(20)
    game.new_episode(path)
    game.make_action([0, 0, 1], 20)
    reward = game.get_total_reward()
    game.new_episode()
    assert (demo_dir / "center.lmp").stat().st_size > 0

    game.replay_episode(path if with_extension else path[:-4])
    assert game.is_replaying_episode()
    for _ in range(30):
        if game.is_episode_finished():
            break
        game.advance_action()
    assert game.is_episode_finished()
    assert game.get_total_reward() == reward


def test_replay_preserves_actions_variables_and_rewards(game):
    actions = [[1, 0, 0]] * 8 + [[0, 1, 0]] * 8 + [[0, 0, 1]] * 8
    history = []
    game.new_episode("actions.lmp")
    assert game.is_recording_episode()
    assert not game.is_replaying_episode()
    for action in actions:
        if game.is_episode_finished():
            break
        state = game.get_state()
        variables = state.game_variables.copy()
        reward = game.make_action(action)
        history.append((state.tic, variables, action, reward))
    total_reward = game.get_total_reward()
    game.new_episode()

    game.replay_episode("actions.lmp")
    assert game.is_replaying_episode()
    assert not game.is_recording_episode()
    for tic, variables, action, reward in history:
        state = game.get_state()
        assert state is not None
        assert state.tic == tic
        np.testing.assert_array_equal(state.game_variables, variables)
        game.advance_action()
        assert game.get_last_action() == action
        assert game.get_last_reward() == reward
    assert game.get_total_reward() == total_reward


def test_consecutive_recordings_are_independent(game, tmp_path):
    recordings = [("left.lmp", [1, 0, 0]), ("right.lmp", [0, 1, 0])]
    for name, action in recordings:
        game.new_episode(name)
        assert game.is_recording_episode()
        game.make_action(action, 5)
    game.new_episode()
    assert not game.is_recording_episode()
    contents = [(tmp_path / name).read_bytes() for name, _ in recordings]
    assert all(contents)
    assert contents[0] != contents[1]

    game.make_action([0, 0, 0], 5)
    for (name, action), content in zip(recordings, contents):
        assert (tmp_path / name).read_bytes() == content
        game.replay_episode(name)
        assert game.is_replaying_episode()
        game.advance_action()
        assert game.get_last_action() == action


@pytest.mark.parametrize("with_extension", [False, True])
def test_close_finalizes_recording(game, tmp_path, with_extension):
    game.new_episode("closed.lmp" if with_extension else "closed")
    game.make_action([1, 0, 0], 5)
    game.close()
    assert (tmp_path / "closed.lmp").stat().st_size > 0

    game.init()
    game.replay_episode("closed.lmp")
    assert game.is_replaying_episode()
    game.advance_action()
    assert game.get_last_action() == [1, 0, 0]
