#!/usr/bin/env python3

#####################################################################
# This script captures map sectors information once and renders a 3D
# sector map (floor/ceiling edges and blocking walls), then exits.
#####################################################################

import os
from argparse import ArgumentParser

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "basic.cfg")

DEFAULT_OUTPUT = "sectors_3d.png"
FLOOR_COLOR = "#2ca25f"
CEILING_COLOR = "#ef6548"
OBJECT_COLOR = "#3288bd"
PLAYER_COLOR = "#ffd92f"


def draw_sectors_3d(ax, sectors, objects):
    """Draw floor/ceiling edges, blocking walls, and map objects."""
    floor_heights = [s.floor_height for s in sectors]
    min_floor = min(floor_heights)
    max_floor = max(floor_heights)
    norm = mcolors.Normalize(
        vmin=min_floor, vmax=max_floor if max_floor > min_floor else min_floor + 1.0
    )
    colormap = plt.get_cmap("viridis")
    xs, ys, zs = [], [], []

    for sector in sectors:
        color = colormap(norm(sector.floor_height))
        fz = sector.floor_height
        cz = sector.ceiling_height
        assert cz >= fz, "Ceiling height must be greater than or equal to floor height."
        zs.extend([fz, cz])

        for line in sector.lines:
            x = [line.x1, line.x2]
            y = [line.y1, line.y2]
            xs.extend(x)
            ys.extend(y)

            # Floor and ceiling are marked with fixed colors.
            ax.plot(x, y, [fz, fz], color=FLOOR_COLOR, linewidth=1.4, alpha=0.95)
            ax.plot(
                x,
                y,
                [cz, cz],
                color=CEILING_COLOR,
                linewidth=1.2,
                alpha=0.95,
            )

            # Draw vertical walls only for blocking edges.
            if line.is_blocking:
                wall = [
                    (line.x1, line.y1, fz),
                    (line.x2, line.y2, fz),
                    (line.x2, line.y2, cz),
                    (line.x1, line.y1, cz),
                ]
                ax.add_collection3d(
                    Poly3DCollection(
                        [wall],
                        facecolors=[color],
                        edgecolors=[color],
                        linewidths=0.3,
                        alpha=0.22,
                    )
                )

    for obj in objects:
        is_player = obj.name == "DoomPlayer"
        marker_color = PLAYER_COLOR if is_player else OBJECT_COLOR
        marker_size = 45 if is_player else 20
        x, y, z = obj.position_x, obj.position_y, obj.position_z
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ax.scatter(
            [x],
            [y],
            [z],
            color=marker_color,
            s=marker_size,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.95,
        )

    ax.set_title("ViZDoom sectors 3D map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(alpha=0.2)
    ax.view_init(elev=35, azim=-60)
    if xs and ys and zs:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
        half = max_range / 2.0
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        z_center = (z_min + z_max) / 2.0
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_zlim(z_center - half, z_center + half)
        ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.legend(
        handles=[
            Line2D([0], [0], color=FLOOR_COLOR, lw=2, label="Floor"),
            Line2D([0], [0], color=CEILING_COLOR, lw=2, label="Ceiling"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=OBJECT_COLOR,
                markeredgecolor="black",
                markersize=7,
                label="Object",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PLAYER_COLOR,
                markeredgecolor="black",
                markersize=8,
                label="Player",
            ),
        ],
        loc="upper right",
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        "ViZDoom example showing how to render sectors information as a 3D map."
    )
    parser.add_argument(
        dest="config",
        default=DEFAULT_CONFIG,
        nargs="?",
        help="Path to the configuration file of the scenario."
        " Please see "
        "../../scenarios/*cfg for more scenarios.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    args = parser.parse_args()

    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_window_visible(False)
    game.set_render_hud(False)
    game.set_sectors_info_enabled(True)
    game.set_objects_info_enabled(True)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)

    game.init()
    game.new_episode()
    state = game.get_state()
    if state is None or state.sectors is None or len(state.sectors) == 0:
        game.close()
        raise RuntimeError(
            "Sectors info is unavailable. Ensure '+viz_nocheat' is not enabled and "
            "the scenario map defines sectors."
        )
    if state.objects is None:
        game.close()
        raise RuntimeError(
            "Objects info is unavailable. Ensure '+viz_nocheat' is not enabled."
        )

    print("Sectors:", len(state.sectors))
    print("Objects:", len(state.objects))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_sectors_3d(ax, state.sectors, state.objects)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print("Saved plot:", args.output)

    plt.close(fig)

    game.close()
