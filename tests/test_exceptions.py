import pytest

import vizdoom as vzd


def test_vizdoom_is_not_running_exception():
    print("Testing ViZDoomIsNotRunningException...")

    game = vzd.DoomGame()
    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.advance_action()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.make_action([0, 1, 0])

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.get_state()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.get_server_state()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.get_last_action()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.get_last_reward()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.get_total_reward()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.new_episode()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.replay_episode("demo.replay")

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.is_new_episode()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.is_episode_finished()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.is_episode_timeout_reached()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.is_player_dead()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.respawn_player()

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.get_button(vzd.Button.ATTACK)

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.get_game_variable(vzd.GameVariable.HEALTH)

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.send_game_command("give ammo")

    with pytest.raises(vzd.ViZDoomIsNotRunningException):
        game.save("non_existent_file.save")


def test_file_does_not_exist_exception():
    print("Testing FileDoesNotExistException...")

    game = vzd.DoomGame()
    with pytest.raises(vzd.FileDoesNotExistException):
        game.load_config("non_existent_file.cfg")

    game = vzd.DoomGame()
    game.set_doom_scenario_path("non_existent_file.wad")
    with pytest.raises(vzd.FileDoesNotExistException):
        game.init()

    game = vzd.DoomGame()
    game.set_doom_game_path("non_existent_file.wad")
    with pytest.raises(vzd.FileDoesNotExistException):
        game.init()


def test_vizdoom_no_sound_exception():
    print("Testing ViZDoomNoOpenALSoundException...")

    # Testing no sound device available with audio buffer enabled (should raise)
    game = vzd.DoomGame()
    game.set_audio_buffer_enabled(True)
    game.set_window_visible(False)
    # game.add_game_args("+snd_backend null")
    # with pytest.raises(vzd.ViZDoomNoOpenALSoundException):
    #     game.init()

    # Testing no sound device available with audio buffer disabled (should not raise)
    game.close()
    game.set_audio_buffer_enabled(False)
    game.init()
    state = game.get_state()
    assert state is not None
    assert state.audio_buffer is None
    game.close()

    # Testing with sound device available and audio buffer enabled (should not raise)
    game.set_audio_buffer_enabled(True)
    game.init()
    state = game.get_state()
    assert state is not None
    assert state.audio_buffer is not None
    game.close()


if __name__ == "__main__":
    test_vizdoom_is_not_running_exception()
    test_file_does_not_exist_exception()
    test_vizdoom_no_sound_exception()
