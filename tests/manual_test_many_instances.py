#!/usr/bin/env python3

import os

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Manager, Process, cpu_count
from random import choice, random
from time import sleep, time

import numpy as np

import vizdoom as vzd


# For singleplayer games threads can also be used.
# from threading import Thread

# Config
episodes = 200
timelimit = 1  # minutes
players = 8  # number of players
win_x = 100
win_y = 100

skip = 4
mode = vzd.Mode.PLAYER  # or Mode.ASYNC_PLAYER
ticrate = 2 * vzd.DEFAULT_TICRATE  # for Mode.ASYNC_PLAYER
random_sleep = False
random_sleep_time_base = 0.001
random_sleep_time_var = 0.005
const_sleep_time = 0.000
sleep_between_episodes = 1  # seconds
sleep_between_episodes = 0.2
window = False
resolution = vzd.ScreenResolution.RES_320X240

args = ""
console = False
config = os.path.join(vzd.scenarios_path, "cig.cfg")


def get_all_players_frags_from_game_vars(game):
    """Return list of (player_name, frags) for every player visible to this client."""
    player_count = int(game.get_game_variable(vzd.GameVariable.PLAYER_COUNT))
    players_frags_from_game_vars = []

    for player_idx in range(1, player_count + 1):
        variable = getattr(vzd.GameVariable, f"PLAYER{player_idx}_FRAGCOUNT")
        # Player names are set as Player0, Player1, ... so subtract 1 from the slot index.
        players_frags_from_game_vars.append(int(game.get_game_variable(variable)))

    server_state = game.get_server_state()
    num_players = server_state.player_count
    players_frags_from_server_state = server_state.players_frags[:num_players]
    # assert num_players == players
    # assert np.array_equal(players_frags_from_game_vars, players_frags_from_server_state)
    if num_players != players:
        print(
            f"WARNING: Player count mismatch between game vars and server state: {num_players} != {players}"
        )
    if not np.array_equal(
        players_frags_from_game_vars, players_frags_from_server_state
    ):
        print(
            f"WARNING: Frag mismatch between game vars and server state: {players_frags_from_game_vars} != {players_frags_from_server_state}"
        )

    return players_frags_from_game_vars, players_frags_from_server_state


def check_frag_agreement(frag_log, episodes):
    """Check that all players reported the same frags for all episodes."""
    print("\n\n\nChecking frag agreement...")
    for episode_idx in range(episodes):
        episode_frags = [
            (player, frags[0])
            for (ep_idx, player, frags) in frag_log
            if ep_idx == episode_idx
        ]

        # assert len(episode_frags) == players, f"Missing frag reports for episode {episode_idx}"
        if len(episode_frags) != players:
            print(f"Missing frag reports for episode {episode_idx}")

        first_frags = episode_frags[0]
        for frags in episode_frags[1:]:
            assert (
                frags[1] == first_frags[1]
            ), f"Frag mismatch in episode {episode_idx}: {frags[0]} {frags[1]} != {first_frags[0]} {first_frags[1]}"
            diff = np.array(frags[1]) - np.array(first_frags[1])
            sum_diff = np.sum(np.abs(diff))
            assert (
                sum_diff <= 1
            ), f"Frag mismatch more than 1 in episode {episode_idx}: {frags[0]} {frags[1]} != {first_frags[0]} {first_frags[1]}"
            if sum_diff > 1:
                print(
                    f"Frag mismatch more than 1 in episode {episode_idx}: {frags[0]} {frags[1]} != {first_frags[0]} {first_frags[1]}"
                )


def setup_player():
    game = vzd.DoomGame()

    game.load_config(config)
    game.set_mode(mode)
    game.add_game_args(args)
    game.set_screen_resolution(resolution)
    game.set_console_enabled(console)
    game.set_window_visible(window)
    game.set_ticrate(ticrate)

    actions = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
    ]

    return game, actions


def player_action(game, player_sleep_time, actions, player_skip):
    if random_sleep:
        sleep(random() * random_sleep_time_var + random_sleep_time_base)
    elif player_sleep_time > 0:
        sleep(player_sleep_time)

    game.make_action(choice(actions), player_skip)

    if game.is_player_dead():
        game.respawn_player()


def player_host(p, frag_log):
    game, actions = setup_player()

    # Setup multiplayer deathmatch game for {p} players that will time out after {timelimit} minutes
    game.add_game_args(
        f"-host {p} -netmode 0 -deathmatch +timelimit {timelimit} +sv_spawnfarthest 1"
    )
    # Use additional arguments to set player name, color and window position
    game.add_game_args(f"+name Player0 +colorset 0 +win_x {win_x} +win_y {win_y}")
    # Add additional arguments
    game.add_game_args(args)

    game.init()

    action_count = 0
    player_sleep_time = const_sleep_time
    player_skip = skip

    for episode_idx in range(episodes):
        print(f"Episode #{episode_idx + 1}")
        episode_start_time = None

        while not game.is_episode_finished():
            if episode_start_time is None:
                episode_start_time = time()

            state = game.get_state()
            assert state is not None

            (
                players_frags_from_game_vars,
                players_frags_from_server_state,
            ) = get_all_players_frags_from_game_vars(game)

            if state.number % 100 == 0:
                print(
                    "Player0 (host):",
                    state.number,
                    action_count,
                    game.get_episode_time(),
                    players_frags_from_game_vars,
                    players_frags_from_server_state,
                )

            player_action(game, player_sleep_time, actions, player_skip)
            action_count += 1

        (
            players_frags_from_game_vars,
            players_frags_from_server_state,
        ) = get_all_players_frags_from_game_vars(game)
        server_state = game.get_server_state()
        print(
            f"Player0 (host): Episode {episode_idx} finished! Frags: {players_frags_from_game_vars} {players_frags_from_server_state}, Time: {time() - episode_start_time:.2f}s, Tics: {server_state.tic}"
        )
        frag_log.append(
            (
                episode_idx,
                f"Player{i}",
                (
                    players_frags_from_game_vars,
                    players_frags_from_server_state,
                    server_state,
                ),
            )
        )

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.
        sleep(random() * sleep_between_episodes)
        game.new_episode()

    game.close()


def player_join(p, frag_log):
    game, actions = setup_player()

    # Join existing game
    game.add_game_args("-join 127.0.0.1")
    # Use additional arguments to set player name, color and window position
    game.add_game_args(
        f"+name Player{p} +colorset 0 +win_x {win_x + p % 4 * game.get_screen_width()} +win_y {win_y + p // 4 * game.get_screen_height()} "
    )
    # Add additional arguments
    game.add_game_args(args)

    game.init()

    action_count = 0
    player_sleep_time = const_sleep_time
    player_skip = skip

    for episode_idx in range(episodes):

        while not game.is_episode_finished():
            state = game.get_state()
            assert state is not None

            (
                players_frags_from_game_vars,
                players_frags_from_server_state,
            ) = get_all_players_frags_from_game_vars(game)

            if state.number % 100 == 0:
                print(
                    f"Player{p}:",
                    state.number,
                    action_count,
                    game.get_episode_time(),
                    players_frags_from_game_vars,
                    players_frags_from_server_state,
                )
            player_action(game, player_sleep_time, actions, player_skip)
            action_count += 1

        (
            players_frags_from_game_vars,
            players_frags_from_server_state,
        ) = get_all_players_frags_from_game_vars(game)
        server_state = game.get_server_state()
        print(
            f"Player{p}: Episode {episode_idx} finished! Frags: {players_frags_from_game_vars}, {players_frags_from_server_state}, Tics: {server_state.tic}"
        )
        frag_log.append(
            (
                episode_idx,
                f"Player{i}",
                (
                    players_frags_from_game_vars,
                    players_frags_from_server_state,
                    server_state,
                ),
            )
        )

        sleep(random() * sleep_between_episodes)
        game.new_episode()

    game.close()


if __name__ == "__main__":
    print("Players:", players)
    print("CPUS:", cpu_count())

    with Manager() as manager:
        frag_log = manager.list()

        processes = []
        for i in range(1, players):
            p_join = Process(target=player_join, args=(i, frag_log))
            p_join.start()
            processes.append(p_join)

        player_host(players, frag_log)

        for process in processes:
            process.join()

        check_frag_agreement(list(frag_log), episodes)

    print("Done")
