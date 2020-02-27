#!/usr/bin/env python3

import vizdoom as vzd
from vizdoom import *
import numpy as np
import imageio
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from random import randint
import os
import cv2
from tqdm import tqdm


def get_random_color():
    return np.random.randint(0, 255, 3, dtype=np.int32)


def draw_bounding_box(buffer, x, y, width, height, color):
    for i in range(width):
        buffer[y, x + i, :] = color
        buffer[y + height, x + i, :] = color

    for i in range(height):
        buffer[y + i, x, :] = color
        buffer[y + i, x + width, :] = color


name_to_color_map = dict({0: [0, 0, 0]})
id_to_color_map = dict({0: [128, 128, 128]})

ammo_color = [0, 0, 255]
weapon_color = [0, 0, 128]
medikit_color = [0, 255, 0]
armor_color = [0, 128, 0]
fog_color = (255, 255, 255)

random_monster_color = lambda: [randint(100, 255), 0, randint(0, 40)]

name_to_color_map['DoomPlayer'] = [128, 128, 128]
name_to_color_map['ClipBox'] = ammo_color
name_to_color_map['RocketBox'] = ammo_color
name_to_color_map['CellPack'] = ammo_color
name_to_color_map['RocketLauncher'] = weapon_color
name_to_color_map['Stimpack'] = medikit_color
name_to_color_map['Medikit'] = medikit_color
name_to_color_map['HealthBonus'] = medikit_color
name_to_color_map['ArmorBonus'] = armor_color
name_to_color_map['GreenArmor'] = armor_color
name_to_color_map['BlueArmor'] = armor_color
name_to_color_map['Chainsaw'] = weapon_color
name_to_color_map['PlasmaRifle'] = weapon_color
name_to_color_map['Chaingun'] = weapon_color
name_to_color_map['ShellBox'] = ammo_color
name_to_color_map['SuperShotgun'] = weapon_color
name_to_color_map['TeleportFog'] = fog_color
name_to_color_map['Zombieman'] = random_monster_color()
name_to_color_map['ShotgunGuy'] = random_monster_color()
name_to_color_map['HellKnight'] = random_monster_color()
name_to_color_map['MarineChainsawVzd'] = random_monster_color()
name_to_color_map['BaronBall'] = random_monster_color()
name_to_color_map['Demon'] = random_monster_color()
name_to_color_map['ChaingunGuy'] = random_monster_color()
name_to_color_map['Blood'] = [0, 0, 0]
name_to_color_map['Clip'] = ammo_color
name_to_color_map['Shotgun'] = weapon_color

wall_id = 0
floor_id = 1

wall_color = [128, 40, 40]
floor_color = [40, 40, 128]

def transform_labels(labels,
                     buffer,
                     disco=False,
                     colorful_name=False,
                     colorful_object=False,
                     bounding_boxes=False):
    rgb_buffer = np.stack([buffer] * 3, axis=2)

    # Walls and floor
    if disco:
        rgb_buffer[buffer == wall_id] = get_random_color()
        rgb_buffer[buffer == floor_id] = get_random_color()
    else:
        rgb_buffer[buffer == wall_id] = wall_color
        rgb_buffer[buffer == floor_id] = floor_color

    if not (disco or colorful_name or colorful_object or bounding_boxes):
        return rgb_buffer
    for l in labels:
        if disco:
            color = get_random_color()
        elif colorful_name:
            name = l.object_name
            if name not in name_to_color_map:
                name_to_color_map[name] = get_random_color()
            color = name_to_color_map[name]
        elif colorful_object:
            if l.object_name == "DoomPlayer":
                color = name_to_color_map[l.object_name]
            else:
                if l.object_id not in id_to_color_map:
                    id_to_color_map[l.object_id] = get_random_color()
                color = id_to_color_map[l.object_id]
        else:
            color = [l.value] * 3

        rgb_buffer[buffer == l.value, :] = color

        if bounding_boxes:
            draw_bounding_box(rgb_buffer, l.x, l.y, l.width, l.height, color)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_buffer, l.object_name, (l.x, l.y - 3), font, 0.3, [int(c) for c in color], 1, cv2.LINE_AA)



    return rgb_buffer


if __name__ == "__main__":
    available_scenarios = [cfg[0:-4] for cfg in vzd.configs]
    parser = ArgumentParser("Creates gif with ViZDoom buffers", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--output_dir", "-o", default="gifs")
    parser.add_argument("--scenario", "-s", default="deadly_corridor", choices=available_scenarios)
    parser.add_argument("--fps", "-fps", type=float, default=35)
    parser.add_argument("--drop", "-d", type=int, default=4, help="Drop every n frames")
    parser.add_argument("--speedup", type=float, default=1, help="Speedup compared to gameplay fps.")
    parser.add_argument("--timeout", "-t", type=float, default=10, help="Set timeout in seconds.")
    parser.add_argument("--bounding-boxes", "-bb", action="store_true", default=False,
                        help="Add bounding boxes to labels buffer.")
    parser.add_argument("--dump-images", "-di", action="store_true", help="Dumps all frames to images directory.")
    coloring_group = parser.add_mutually_exclusive_group()
    coloring_group.add_argument("--disco", action="store_true", default=False, help="Stayin alive!")
    coloring_group.add_argument("--color-labels-name", "-cln", action="store_true", default=False,
                                help="Use colors for labels (by name).")
    coloring_group.add_argument("--color-labels-object", "-clo", action="store_true", default=False,
                                help="Use colors for labels (by object.id).")
    args = parser.parse_args()

    images = []

    game = DoomGame()
    CONC_AXIS = 1
    game.load_config(vzd.__path__[0] + "/scenarios/{}.cfg".format(args.scenario))

    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_320X240)

    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)

    # game.set_automap_rotate(False)
    game.set_automap_render_textures(True)

    game.set_render_hud(False)
    game.set_render_minimal_hud(False)

    game.set_mode(Mode.SPECTATOR)
    game.set_episode_timeout(int(35 * args.timeout))
    game.init()
    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()

        picture = state.screen_buffer
        labels_buffer = state.labels_buffer
        if labels_buffer is not None:
            labels_buffer = transform_labels(
                state.labels,
                labels_buffer,
                disco=args.disco,
                colorful_name=args.color_labels_name,
                colorful_object=args.color_labels_object,
                bounding_boxes=args.bounding_boxes)
            picture = np.concatenate([picture, labels_buffer], axis=CONC_AXIS)

        depthbuffer = state.depth_buffer
        if depthbuffer is not None:
            depthbuffer = np.stack([depthbuffer] * 3, axis=2)
            picture = np.concatenate([picture, depthbuffer], axis=CONC_AXIS)

        automap = state.automap_buffer
        if automap is not None:
            picture = np.concatenate([picture, automap], axis=CONC_AXIS)

        game.advance_action()
        images.append(picture)

    game.close()

    if args.dump_images:
        img_dir = "images"
        if not os.path.exists(img_dir):
            print("Creating directory: {}".format(img_dir))
            os.makedirs(img_dir)
        for i, img in tqdm(enumerate(images), desc="Dumping images", leave=False, total=len(images)):
            cv2.imwrite("{}/{}_frame_{}.png".format(img_dir, args.scenario, i), img[:, :, [2, 1, 0]])

    images = np.array(images)[::args.drop]

    if not os.path.exists(args.output_dir):
        print("Creating directory: {}".format(args.output_dir))
        os.makedirs(args.output_dir)
    print("Saving the gif ...")
    imageio.mimsave('{}/{}_{}fps.gif'.format(args.output_dir, args.scenario, args.fps), images,
                    duration=1 / args.fps * args.drop / args.speedup)
