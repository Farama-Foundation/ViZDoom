#!/usr/bin/env python3

# This test should be run manually.
# It shows the native game window and a second window containing screen_buffer

import os
from argparse import ArgumentParser

import pygame

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "basic.cfg")
DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_SCALE = 2
FLASH_AMOUNT = 0.65
TURN_DEGREES_PER_TIC = 1.0
STATUS_BAR_HEIGHT = 24


def set_flash(game, color, flashing):
    amount = FLASH_AMOUNT if flashing else 0
    game.send_game_command(f"testblend {color} {amount}")
    print(f"Flash {color}: {'on' if flashing else 'off'}")


def draw_buffer(window, state, color, render_flashes, flashing, scale):
    screen_buffer = state.screen_buffer
    screen_size = (screen_buffer.shape[1], screen_buffer.shape[0])
    buffer_surface = pygame.image.frombytes(screen_buffer.tobytes(), screen_size, "RGB")
    buffer_size = (screen_size[0] * scale, screen_size[1] * scale)
    window.blit(pygame.transform.scale(buffer_surface, buffer_size), (0, 0))

    if not flashing:
        status_color = (65, 65, 65)
    elif render_flashes:
        status_color = (20, 90, 20)
    else:
        status_color = (110, 25, 25)

    pygame.draw.rect(
        window,
        status_color,
        (0, buffer_size[1], buffer_size[0], STATUS_BAR_HEIGHT),
    )
    pygame.display.set_caption(
        f"ViZDoom screen buffer - {color} flash "
        f"{'active' if flashing else 'break'} - render flashes "
        f"{'on' if render_flashes else 'off'}"
    )
    pygame.display.flip()


def main():
    parser = ArgumentParser(
        description=(
            "Periodically flash ViZDoom's native window and show the Python "
            "screen buffer in a second window."
        )
    )
    parser.add_argument("color", help="Flash color accepted by the testblend command.")
    parser.add_argument(
        "render_flashes",
        choices=("on", "off"),
        help="Whether render_screen_flashes is enabled for the screen buffer.",
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
        help="Seconds between switching the flash on and off.",
    )
    parser.add_argument(
        "--scale",
        default=DEFAULT_SCALE,
        type=int,
        help="Integer scale of the screen-buffer window.",
    )
    args = parser.parse_args()

    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")
    if args.scale <= 0:
        parser.error("--scale must be greater than zero")

    render_flashes = args.render_flashes == "on"

    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_episode_timeout(0)
    game.set_render_all_frames(True)
    game.set_render_screen_flashes(render_flashes)
    game.clear_available_buttons()
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, TURN_DEGREES_PER_TIC)

    pygame_initialized = False
    try:
        game.init()

        pygame.init()
        pygame_initialized = True

        window_width = game.get_screen_width() * args.scale
        window_height = game.get_screen_height() * args.scale + STATUS_BAR_HEIGHT
        window = pygame.display.set_mode((window_width, window_height))
        clock = pygame.time.Clock()

        print("The native ViZDoom window shows the engine output.")
        print("The Pygame window shows the Python screen_buffer.")
        print(f"render_screen_flashes is {args.render_flashes}.")

        flashing = True
        last_change = pygame.time.get_ticks()
        running = True
        game.set_action([TURN_DEGREES_PER_TIC])
        set_flash(game, args.color, flashing)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        flashing = not flashing
                        last_change = pygame.time.get_ticks()
                        set_flash(game, args.color, flashing)

            if not running:
                break

            now = pygame.time.get_ticks()
            if now - last_change >= args.interval_seconds * 1000:
                flashing = not flashing
                last_change = now
                set_flash(game, args.color, flashing)

            if game.is_episode_finished():
                game.new_episode()
                game.set_action([TURN_DEGREES_PER_TIC])
                set_flash(game, args.color, flashing)

            game.advance_action()
            state = game.get_state()
            if state is not None:
                draw_buffer(
                    window,
                    state,
                    args.color,
                    render_flashes,
                    flashing,
                    args.scale,
                )

            clock.tick(vzd.DEFAULT_TICRATE)
    finally:
        if pygame_initialized:
            pygame.quit()
        game.close()


if __name__ == "__main__":
    main()
