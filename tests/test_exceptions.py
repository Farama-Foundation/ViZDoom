# Tests exceptions for unavailable games, missing files, and unsupported sound settings.

import struct
import subprocess
import sys
import textwrap

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
        game.get_instance_id()

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


def test_engine_startup_error(tmp_path):
    decorate = b"actor BadActor 16000 { Stop }\n"
    wad_path = tmp_path / "bad_decorate.wad"
    wad_path.write_bytes(
        struct.pack("<4sII", b"PWAD", 1, 12 + len(decorate))
        + decorate
        + struct.pack("<II8s", 12, len(decorate), b"DECORATE")
    )

    # Isolate native crashes so a regression fails this test, not the test runner.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import pytest
                import vizdoom as vzd

                game = vzd.DoomGame()
                game.set_window_visible(False)
                game.set_sound_enabled(False)
                game.set_doom_scenario_path(sys.argv[1])
                try:
                    for _ in range(2):
                        with pytest.raises((vzd.ViZDoomErrorException,
                                            vzd.ViZDoomUnexpectedExitException)):
                            game.init()
                        assert not game.is_running()
                        game.close()
                        game.close()

                    # Failed initialization must leave the game reusable.
                    game.set_doom_scenario_path("")
                    game.init()
                    assert game.is_running()
                    assert game.get_state() is not None
                    game.advance_action()
                finally:
                    game.close()
                """
            ),
            str(wad_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("game_args", ["-nosound", "+snd_backend null"])
def test_vizdoom_no_sound_exception(game_args):
    print("Testing ViZDoomNoOpenALSoundException...")

    # Testing no sound device available with audio buffer enabled (should raise)
    game = vzd.DoomGame()
    game.set_audio_buffer_enabled(True)
    game.set_window_visible(False)
    # -nosound keeps snd_backend set to openal but uses the null renderer,
    # just as when OpenAL fails to initialize and falls back to no sound.
    game.add_game_args(game_args)
    try:
        with pytest.raises(vzd.ViZDoomNoOpenALSoundException):
            game.init()
        assert not game.is_running()
    finally:
        game.close()

    # Testing no sound device available with audio buffer disabled (should not raise)
    game.set_audio_buffer_enabled(False)
    try:
        game.init()
        state = game.get_state()
        assert state is not None
        assert state.audio_buffer is None
    finally:
        game.close()
