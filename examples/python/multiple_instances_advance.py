#!/usr/bin/env python

from __future__ import print_function

from random import choice, random
from time import sleep, time
from vizdoom import *

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import cpu_count, Process

# For singleplayer games threads can also be used.
# from threading import Thread

# Config
episodes = 1
timelimit = 1   # minutes
players = 8    # number of players

skip = 4
mode = Mode.PLAYER
random_sleep = True
const_sleep_time = 0.005
window = False
resolution = ScreenResolution.RES_320X240

args =""
console = False
config = "../../scenarios/multi_duel.cfg"

def player_host(p):
    game = DoomGame()

    game.load_config(config)
    game.add_game_args("-host " + str(p) + " -netmode 0 -deathmatch +timelimit " + str(timelimit) + " +sv_spawnfarthest 1")
    game.add_game_args("+name Player0 +colorset 0")
    game.set_mode(mode)
    game.add_game_args(args)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_console_enabled(console)
    game.set_window_visible(window)

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    action_count = 0
    sleep_time = const_sleep_time

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        episode_start_time = None

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            state = game.get_state()

            if episode_start_time is None:
                episode_start_time = time()

            game.make_action(choice(actions), skip)
            action_count += 1

            if random_sleep:
                sleep(random() * 0.005 + 0.001)
            elif sleep_time > 0:
                sleep(sleep_time)

            if state:
                print("Player0:", state.number, action_count, game.get_episode_time())

        print("Player0 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

        print("Host: Episode finished!")

        player_count = int(game.get_game_variable(GameVariable.PLAYER_COUNT))
        for i in range(1, player_count + 1):
            print("Host: Player" + str(i) + ":", game.get_game_variable(eval("GameVariable.PLAYER" + str(i) + "_FRAGCOUNT")))

        print("Host: Episode processing time:", time() - episode_start_time)

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.
        game.new_episode()

    game.close()


def player_join(p):
    game = DoomGame()

    game.load_config(config)
    game.add_game_args("-join 127.0.0.1")
    game.add_game_args("+name Player" + str(p) + " +colorset " + str(p))
    game.set_mode(mode)
    game.add_game_args(args)
    game.set_screen_resolution(resolution)
    game.set_console_enabled(console)
    game.set_window_visible(window)

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    action_count = 0
    sleep_time = const_sleep_time

    for i in range(episodes):

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            state = game.get_state()

            game.make_action(choice(actions), skip)
            action_count += 1
            print("Player" + str(p) + ":", state.number, action_count, game.get_episode_time())

            if random_sleep:
                sleep(random() * 0.005 + 0.001)
            elif sleep_time > 0:
                sleep(sleep_time)

        print("Player" + str(p) + " frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
        game.new_episode()

    game.close()


if __name__ == '__main__':
    print("Players:", players)
    print("CPUS:", cpu_count())

    processes = []
    for i in range(1, players):
        p_join = Process(target=player_join, args=(i,))
        p_join.start()
        processes.append(p_join)

    player_host(players)

    print("Done")

