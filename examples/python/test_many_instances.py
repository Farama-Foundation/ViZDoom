#!/usr/bin/env python3

from __future__ import print_function
from multiprocessing import Process
from random import choice
import vizdoom as vzd
from argparse import ArgumentParser


DEFAULT_CONFIG = "../../scenarios/basic.cfg"


def play(process, instances, config_file):
    games = []
    for i in range(instances):
        game = vzd.DoomGame()
        game.load_config(config_file)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_window_visible(False)
        game.init()
        games.append(game)
        print("Process {}: Game {} started...".format(process, i))

    actions = [[True, False, False],
               [False, True, False],
               [False, False, True]]

    for g in games:
        g.new_episode()
        while not g.is_episode_finished():
            g.make_action(choice(actions))

    for g in games:
        g.close()


if __name__ == '__main__':
    parser = ArgumentParser("Test for N instances in P processes.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")
    parser.add_argument("-i", "--instances", default=128, help="Number of instances per process.")
    parser.add_argument("-p", "--processes", default=4, help="Number of processes.")
    args = parser.parse_args()

    processes= []
    for p in range(args.processes):
        p = Process(target=play, args=[p, args.instances, args.config])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
