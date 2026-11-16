# Tests unsigned engine seeds and reproducible seeds across episodes and recordings.

import os

import numpy as np
import pytest

import vizdoom as vzd


def demo_seed(path):
    data = path.read_bytes()
    # The ZDHD payload starts with four version bytes and a NUL-terminated map name.
    header = data.index(b"ZDHD") + 8
    offset = data.index(b"\0", header + 4) + 1
    return int.from_bytes(data[offset : offset + 4], "big")


def make_game():
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    return game


@pytest.mark.parametrize("seed", [0, 2**31 - 1, 2**31, 2**32 - 1])
@pytest.mark.parametrize("command", ["viz_set_seed", "rngseed set"])
def test_engine_seed_commands_preserve_unsigned_values(
    tmp_path, monkeypatch, seed, command
):
    monkeypatch.chdir(tmp_path)
    game = make_game()
    try:
        game.init()
        game.send_game_command(f"{command} {seed}")
        if command == "rngseed set":
            # This command applies the seed on the next map, after a demo header
            # would be written by recordmap, so load that map first.
            game.send_game_command("map map01")
            game.advance_action(5)
        # Use recordmap directly so new_episode does not draw another seed.
        game.send_game_command("recordmap seed.lmp map01")
        game.advance_action(5)
    finally:
        game.close()
    assert demo_seed(tmp_path / "seed.lmp") == seed


def test_startup_and_episode_seeds_match_unsigned_rng(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    game = make_game()
    game.set_seed(0)
    # Startup recording exercises the -rngseed argument before new_episode.
    game.add_game_args("-record startup.lmp")
    paths = [tmp_path / "startup.lmp"]
    try:
        game.init()
        game.advance_action(5)
        for episode in range(8):
            name = f"episode{episode}.lmp"
            game.new_episode(name)
            game.advance_action(5)
            paths.append(tmp_path / name)
    finally:
        game.close()

    # Both the controller and RandomState use MT19937 with the same integer seed.
    expected = np.random.RandomState(0).randint(
        0, 2**32, size=len(paths), dtype=np.uint32
    )
    assert [demo_seed(path) for path in paths] == expected.tolist()
