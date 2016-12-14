#!/usr/bin/env python

from __future__ import print_function

from random import choice, random
from time import sleep
from vizdoom import *
# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Process

# For singleplayer games threads can also be used.
# from threading import Thread

# Config
episodes = 10
timelimit = 1   # min
players = 2     # number of players
skip = 1
mode = Mode.PLAYER
random_sleep = False
window = False

args =""
console = False
args = "+viz_debug 1 +viz_debug_instances 1"
console = True

#config = "../config/multi_duel.cfg"
config = "../config/cig.cfg"


def player_host(p):
    game = DoomGame()

    game.load_config(config)
    game.add_game_args("-host " + str(p) + " -deathmatch +timelimit " + str(timelimit) + " +sv_spawnfarthest 1")
    game.add_game_args("+name Player0 +colorset 0")
    game.set_mode(mode)
    game.add_game_args(args)
    game.set_console_enabled(console)
    game.set_window_visible(window)

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    action_count = 0

    for i in range(episodes):

        print("Episode #" + str(i + 1))

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            state = game.get_state()
            game.make_action(choice(actions), skip)
            action_count += 1

            if random_sleep:
                sleep(random()/10)

            print("Player0:", state.number, action_count, game.get_episode_time())

        print("Player0 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
        print("Episode finished!")
        for i in range(p):
            if i == 0:
                print("Host: Player" + str(i) + " frags:", game.get_game_variable(GameVariable.PLAYER1_FRAGCOUNT))
            if i == 1:
                print("Host: Player" + str(i) + " frags:", game.get_game_variable(GameVariable.PLAYER2_FRAGCOUNT))
            if i == 2:
                print("Host: Player" + str(i) + " frags:", game.get_game_variable(GameVariable.PLAYER3_FRAGCOUNT))
            if i == 3:
                print("Host: Player" + str(i) + " frags:", game.get_game_variable(GameVariable.PLAYER4_FRAGCOUNT))
            if i == 4:
                print("Host: Player" + str(i) + " frags:", game.get_game_variable(GameVariable.PLAYER5_FRAGCOUNT))
            if i == 5:
                print("Host: Player" + str(i) + " frags:", game.get_game_variable(GameVariable.PLAYER6_FRAGCOUNT))
            if i == 6:
                print("Host: Player" + str(i) + " frags:", game.get_game_variable(GameVariable.PLAYER7_FRAGCOUNT))
            if i == 7:
                print("Host: Player" + str(i) + " frags:", game.get_game_variable(GameVariable.PLAYER8_FRAGCOUNT))

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
    game.set_console_enabled(console)
    game.set_window_visible(window)

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    action_count = 0
    sleep_time = 0.01 * p

    for i in range(episodes):

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            state = game.get_state()

            game.make_action(choice(actions), skip)
            action_count += 1
            print("Player" + str(p) + ":", state.number, action_count, game.get_episode_time())

            if random_sleep:
                sleep(random()/10)
            elif sleep_time > 0:
                sleep(sleep_time)

        print("Player" + str(p) + " frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
        game.new_episode()

    game.close()


if __name__ == '__main__':
    processes = []

    # p_host = Process(target=player_host, args=(players,))
    # p_host.start()
    # processes.append(p_host)

    for i in range(players - 1):
        p_join = Process(target=player_join, args=(i + 1,))
        p_join.start()
        processes.append(p_join)

    player_host(players)

    print("Done")

