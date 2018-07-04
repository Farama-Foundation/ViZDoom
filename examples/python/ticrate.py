#!/usr/bin/env python3

from __future__ import print_function
from multiprocessing import Process
from random import choice
import vizdoom as vzd
from argparse import ArgumentParser

DEFAULT_CONFIG = "../../scenarios/basic.cfg"


def play(config_file, ticrate=35):
    game = vzd.DoomGame()

    game.load_config(config_file)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    game.set_ticrate(ticrate)

    game.init()

    actions = [[True, False, False],
               [False, True, False],
               [False, False, True]]
    episodes = 10

    for i in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            game.make_action(choice(actions))

    game.close()


if __name__ == '__main__':
    if __name__ == "__main__":
        parser = ArgumentParser("ViZDoom example showing how to change the ticrate for asynchronous mode.")
        parser.add_argument(dest="config",
                            default=DEFAULT_CONFIG,
                            nargs="?",
                            help="Path to the configuration file of the scenario."
                                 " Please see "
                                 "../../scenarios/*cfg for more scenarios.")
        parser.add_argument("-t", "--ticrates",
                            default=[17,35,70],
                            nargs="+",
                            help="List of ticrates to show.")
        args = parser.parse_args()

        processes= []
        for ticrate in args.ticrates:
            p = Process(target=play, args=[args.config, ticrate])
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
