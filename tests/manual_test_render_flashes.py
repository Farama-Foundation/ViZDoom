#!/usr/bin/env python3

# This test should be run manually.
# It shows flashes in the native game window and in screen_buffer.

import os
from argparse import ArgumentParser

import pygame

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "predict_position.cfg")
DEFAULT_INTERVAL_SECONDS = 2.0
MAX_PICKUP_TICS = 35
TESTBLEND_AMOUNT = 0.65


def start_pickup(game):
    game.send_game_command("take health 1")
    game.send_game_command("summon HealthBonus")


def set_testblend(game, color, flashing):
    amount = TESTBLEND_AMOUNT if flashing else 0
    game.send_game_command(f"testblend {color} {amount}")
    print(f"testblend {color}: {'flash' if flashing else 'break'}")


def draw_buffer(window, state, effect, status, render_flashes):
    screen_buffer = state.screen_buffer
    screen_size = (screen_buffer.shape[1], screen_buffer.shape[0])
    buffer_surface = pygame.image.frombytes(screen_buffer.tobytes(), screen_size, "RGB")
    window.blit(buffer_surface, (0, 0))
    pygame.display.set_caption(
        f"ViZDoom screen buffer - {effect} {status} - render flashes "
        f"{'on' if render_flashes else 'off'}"
    )
    pygame.display.flip()


def main():
    parser = ArgumentParser(
        description=(
            "Periodically produce flashes and show the native game window "
            "and Python screen buffer."
        )
    )
    parser.add_argument(
        "render_flashes",
        choices=("on", "off"),
        help="Whether player flashes are rendered.",
    )
    parser.add_argument(
        "--testblend",
        metavar="COLOR",
        help=(
            "Use testblend with COLOR instead of pickups. testblend is not "
            "controlled by render_screen_flashes."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to a ViZDoom scenario configuration file.",
    )
    parser.add_argument(
        "--interval-seconds",
        default=DEFAULT_INTERVAL_SECONDS,
        type=float,
        help="Seconds between flashes and breaks.",
    )
    args = parser.parse_args()

    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")

    render_flashes = args.render_flashes == "on"

    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_episode_timeout(0)
    game.set_render_all_frames(True)
    game.set_render_screen_flashes(render_flashes)
    game.clear_available_buttons()
    game.add_available_button(vzd.Button.MOVE_FORWARD)

    pygame_initialized = False
    try:
        game.init()

        pygame.init()
        pygame_initialized = True

        window = pygame.display.set_mode(
            (game.get_screen_width(), game.get_screen_height())
        )
        clock = pygame.time.Clock()

        print("The native ViZDoom window shows the engine output.")
        print("The Pygame window shows the Python screen_buffer.")
        print(f"render_screen_flashes is {args.render_flashes}.")

        use_testblend = args.testblend is not None
        effect = f"testblend {args.testblend}" if use_testblend else "pickup"
        start_x = round(game.get_game_variable(vzd.GameVariable.POSITION_X))
        start_y = round(game.get_game_variable(vzd.GameVariable.POSITION_Y))
        collecting = False
        saw_health_drop = False
        pickup_tics = 0
        health_before_pickup = game.get_game_variable(vzd.GameVariable.HEALTH)
        last_pickup = pygame.time.get_ticks() - args.interval_seconds * 1000
        flashing = use_testblend
        last_testblend_change = pygame.time.get_ticks()
        if use_testblend:
            set_testblend(game, args.testblend, flashing)
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if use_testblend:
                            flashing = not flashing
                            last_testblend_change = pygame.time.get_ticks()
                            set_testblend(game, args.testblend, flashing)
                        else:
                            last_pickup = (
                                pygame.time.get_ticks() - args.interval_seconds * 1000
                            )

            if not running:
                break

            now = pygame.time.get_ticks()
            if (
                use_testblend
                and now - last_testblend_change >= args.interval_seconds * 1000
            ):
                flashing = not flashing
                last_testblend_change = now
                set_testblend(game, args.testblend, flashing)
            elif (
                not use_testblend
                and not collecting
                and now - last_pickup >= args.interval_seconds * 1000
            ):
                health_before_pickup = game.get_game_variable(vzd.GameVariable.HEALTH)
                start_pickup(game)
                collecting = True
                saw_health_drop = False
                pickup_tics = 0

            if game.is_episode_finished():
                game.new_episode()
                collecting = False
                last_pickup = now
                if use_testblend:
                    set_testblend(game, args.testblend, flashing)

            game.make_action([int(collecting and not use_testblend)])
            if collecting and not use_testblend:
                pickup_tics += 1

                health = game.get_game_variable(vzd.GameVariable.HEALTH)
                saw_health_drop = saw_health_drop or health < health_before_pickup
                if saw_health_drop and health >= health_before_pickup:
                    collecting = False
                    last_pickup = now
                    game.send_game_command(f"warp {start_x} {start_y}")
                    print("HealthBonus picked up")
                elif pickup_tics >= MAX_PICKUP_TICS:
                    raise RuntimeError("HealthBonus was not picked up")

            state = game.get_state()
            if state is not None:
                if use_testblend:
                    status = "flash" if flashing else "break"
                else:
                    status = "approaching" if collecting else "break"
                draw_buffer(
                    window,
                    state,
                    effect,
                    status,
                    render_flashes,
                )

            clock.tick(vzd.DEFAULT_TICRATE)
    finally:
        if pygame_initialized:
            pygame.quit()
        game.close()


if __name__ == "__main__":
    main()
