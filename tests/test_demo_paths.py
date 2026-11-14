import os

import pytest

import vizdoom as vzd


@pytest.mark.parametrize("forward_slashes", [False, True])
@pytest.mark.parametrize("with_extension", [False, True])
def test_record_and_replay_path_with_spaces(
    tmp_path, monkeypatch, forward_slashes, with_extension
):
    # Relative paths keep this regression independent of the command length limit.
    monkeypatch.chdir(tmp_path)
    demo_dir = tmp_path / "my demos"
    demo_dir.mkdir()
    path = os.path.join("my demos", "center.lmp")
    if forward_slashes:
        path = path.replace("\\", "/")

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    game.set_episode_timeout(20)
    try:
        game.init()
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
    finally:
        game.close()
