import itertools
import warnings
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.utils import EzPickle

import vizdoom.vizdoom as vzd


# A fixed set of colors for each potential label
# for rendering an image.
# 256 is not nearly enough for all IDs, but we limit
# ourselves here to avoid hogging too much memory.
LABEL_COLORS = (
    np.random.default_rng(42).uniform(25, 256, size=(256, 3)).astype(np.uint8)
)


class VizdoomEnv(gym.Env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": vzd.DEFAULT_TICRATE,
    }

    def __init__(
        self,
        level,
        frame_skip=1,
        max_buttons_pressed=1,
        render_mode: Optional[str] = None,
    ):
        """
        Base class for Gymnasium interface for ViZDoom. Thanks to https://github.com/shakenes/vizdoomgym
        Child classes are defined in gym_env_defns.py,

        Arguments:
            level (str): The path to the config file to load. Most settings should be set by this config file.
            frame_skip (int): The number of frames the will be advanced per action. 1 = take action on every frame. Default: 1.
            max_buttons_pressed (int): Defines the number of binary buttons that can be selected at once. Default: 1.
                                       Should be >= 0. If < 0 a RuntimeError is raised.
                                       If == 0, the binary action space becomes ``MultiDiscrete([2] * num_binary_buttons)``
                                       and [0, ``num_binary_buttons``] number of binary buttons can be selected.
                                       If > 0, the binary action space becomes ``Discrete(n)``
                                       and ``n`` actions can be selected.
                                       ``n`` is equal to number of possible buttons combinations
                                       with the number of buttons pressed < ``max_buttons_pressed``.
            render_mode(Optional[str]): The render mode to use could be either "human" or "rgb_array"

        This environment forces the game window to be hidden. Use :meth:`render` function to see the game.

        Observations are dictionaries with different number of entries, depending on if depth/label buffers were
        enabled in the config file (CHANNELS == 1 if GRAY8, else 3):

        - "screen": The screen image buffer (always available) as `np.ndarray` in shape ``(HEIGHT, WIDTH, CHANNELS)``.
        - "depth": The depth image as ``np.ndarray`` in shape ``(HEIGHT, WIDTH, 1)``, if enabled by the config file.
        - "labels": The label image buffer as `np.ndarray` in shape ``(HEIGHT, WIDTH, 1)``, if enabled by the config file.
          For info on labels, access `env.state.labels` variable.
        - "automap": The automap image buffer as ``np.ndarray`` in shape ``(HEIGHT, WIDTH, CHANNELS)``, if enabled by the config file.
        - "gamevariables": All game variables, in the order specified by the config file.

        Action space can be a single one of binary/continuous action space, or a ``Dict`` containing both:

        - "binary": Is ``MultiDiscrete([2] * num_binary_buttons)`` if :attr:`max_buttons_pressed` == 0
          or ``Discrete(num_binary_buttons + 1)`` if :attr:`max_buttons_pressed` >= 1.
        - "continuous": Is ``Box(float32.min, float32.max, (num_delta_buttons,), float32)``.
        """
        EzPickle.__init__(self, level, frame_skip, max_buttons_pressed, render_mode)
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # init game
        self.game = vzd.DoomGame()
        self.game.load_config(level)
        self.game.set_window_visible(False)

        screen_format = self.game.get_screen_format()
        if (
            screen_format != vzd.ScreenFormat.RGB24
            and screen_format != vzd.ScreenFormat.GRAY8
        ):
            warnings.warn(
                f"Detected screen format {screen_format.name}. Only RGB24 and GRAY8 are supported in the Gymnasium"
                f" wrapper. Forcing RGB24."
            )
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        self.state = None
        self.clock = None
        self.window_surface = None
        self.isopen = True
        self.channels = 3
        if screen_format == vzd.ScreenFormat.GRAY8:
            self.channels = 1

        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()
        self.automap = self.game.is_automap_buffer_enabled()

        # parse buttons defined by config file
        self.__parse_available_buttons()

        # check for valid max_buttons_pressed
        if max_buttons_pressed > self.num_binary_buttons > 0:
            warnings.warn(
                f"max_buttons_pressed={max_buttons_pressed} "
                f"> number of binary buttons defined={self.num_binary_buttons}. "
                f"Clipping max_buttons_pressed to {self.num_binary_buttons}."
            )
            max_buttons_pressed = self.num_binary_buttons
        elif max_buttons_pressed < 0:
            raise RuntimeError(
                f"max_buttons_pressed={max_buttons_pressed} < 0. Should be >= 0. "
            )

        # specify action space(s)
        self.max_buttons_pressed = max_buttons_pressed
        self.action_space = self.__get_action_space()

        # specify observation space(s)
        self.observation_space = self.__get_observation_space()

        self.game.init()

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call `reset` before using `step` method."

        env_action = self.__build_env_action(action)
        reward = self.game.make_action(env_action, self.frame_skip)
        self.state = self.game.get_state()
        terminated = self.game.is_episode_finished()
        truncated = False  # Truncation to be handled by the TimeLimit wrapper
        if self.render_mode == "human":
            self.render()
        return self.__collect_observations(), reward, terminated, truncated, {}

    def __parse_binary_buttons(self, env_action, agent_action):
        if self.num_binary_buttons != 0:
            if self.num_delta_buttons != 0:
                agent_action = agent_action["binary"]

            if np.issubdtype(type(agent_action), np.integer):
                agent_action = self.button_map[agent_action]

            # binary actions offset by number of delta buttons
            env_action[self.num_delta_buttons :] = agent_action

    def __parse_delta_buttons(self, env_action, agent_action):
        if self.num_delta_buttons != 0:
            if self.num_binary_buttons != 0:
                agent_action = agent_action["continuous"]

            # delta buttons have a direct mapping since they're reorganized to be prior to any binary buttons
            env_action[0 : self.num_delta_buttons] = agent_action

    def __build_env_action(self, agent_action):
        # encode users action as environment action
        env_action = np.array(
            [0 for _ in range(self.num_delta_buttons + self.num_binary_buttons)],
            dtype=np.float32,
        )
        self.__parse_delta_buttons(env_action, agent_action)
        self.__parse_binary_buttons(env_action, agent_action)
        return env_action

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.state = self.game.get_state()

        return self.__collect_observations(), {}

    def __collect_observations(self):
        observation = {}
        if self.state is not None:
            observation["screen"] = self.state.screen_buffer
            if self.channels == 1:
                observation["screen"] = self.state.screen_buffer[..., None]
            if self.depth:
                observation["depth"] = self.state.depth_buffer[..., None]
            if self.labels:
                observation["labels"] = self.state.labels_buffer[..., None]
            if self.automap:
                observation["automap"] = self.state.automap_buffer
                if self.channels == 1:
                    observation["automap"] = self.state.automap_buffer[..., None]
            if self.num_game_variables > 0:
                observation["gamevariables"] = self.state.game_variables.astype(
                    np.float32
                )
        else:
            # there is no state in the terminal step, so a zero observation is returned instead
            for space_key, space_item in self.observation_space.spaces.items():
                observation[space_key] = np.zeros(
                    space_item.shape, dtype=space_item.dtype
                )

        return observation

    def __build_human_render_image(self):
        """Stack all available buffers into one for human consumption"""
        game_state = self.game.get_state()
        valid_buffers = game_state is not None

        if not valid_buffers:
            # Return a blank image
            num_enabled_buffers = 1 + self.depth + self.labels + self.automap
            img = np.zeros(
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width() * num_enabled_buffers,
                    3,
                ),
                dtype=np.uint8,
            )
            return img

        image_list = [game_state.screen_buffer]
        if self.channels == 1:
            image_list = [
                np.repeat(game_state.screen_buffer[..., None], repeats=3, axis=2)
            ]

        if self.depth:
            image_list.append(
                np.repeat(game_state.depth_buffer[..., None], repeats=3, axis=2)
            )

        if self.labels:
            # Give each label a fixed color.
            # We need to connect each pixel in labels_buffer to the corresponding
            # id via `value``
            labels_rgb = np.zeros_like(image_list[0])
            labels_buffer = game_state.labels_buffer
            for label in game_state.labels:
                color = LABEL_COLORS[label.object_id % 256]
                labels_rgb[labels_buffer == label.value] = color
            image_list.append(labels_rgb)

        if self.automap:
            automap_buffer = game_state.automap_buffer
            if self.channels == 1:
                automap_buffer = np.repeat(automap_buffer[..., None], repeats=3, axis=2)
            image_list.append(automap_buffer)

        return np.concatenate(image_list, axis=1)

    def render(self):
        if self.clock is None:
            self.clock = pygame.time.Clock()
        render_image = self.__build_human_render_image()
        if self.render_mode == "rgb_array":
            return render_image
        elif self.render_mode == "human":
            # Transpose image (pygame wants (width, height, channels), we have (height, width, channels))
            render_image = render_image.transpose(1, 0, 2)
            if self.window_surface is None:
                pygame.init()
                pygame.display.set_caption("ViZDoom")
                self.window_surface = pygame.display.set_mode(render_image.shape[:2])

            surf = pygame.surfarray.make_surface(render_image)
            self.window_surface.blit(surf, (0, 0))
            pygame.display.update()
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return self.isopen

    def close(self):
        if self.window_surface:
            pygame.quit()
            self.isopen = False

    def __parse_available_buttons(self):
        """
        Parses the currently available game buttons,
        reorganizes all delta buttons to be prior to any binary buttons
        sets ``num_delta_buttons``, ``num_binary_buttons``
        """
        delta_buttons = []
        binary_buttons = []
        for button in self.game.get_available_buttons():
            if vzd.is_delta_button(button) and button not in delta_buttons:
                delta_buttons.append(button)
            else:
                binary_buttons.append(button)
        # force all delta buttons to be first before any binary buttons
        self.game.set_available_buttons(delta_buttons + binary_buttons)
        self.num_delta_buttons = len(delta_buttons)
        self.num_binary_buttons = len(binary_buttons)
        if delta_buttons == binary_buttons == 0:
            raise RuntimeError(
                "No game buttons defined. Must specify game buttons using `available_buttons` in the "
                "config file."
            )

    def __get_binary_action_space(self):
        """
        Return binary action space: either ``Discrete(n)``/``MultiDiscrete([2] * num_binary_buttons)``
        """
        if self.max_buttons_pressed == 0:
            button_space = gym.spaces.MultiDiscrete(
                [
                    2,
                ]
                * self.num_binary_buttons
            )
        else:
            self.button_map = [
                np.array(list(action))
                for action in itertools.product((0, 1), repeat=self.num_binary_buttons)
                if (self.max_buttons_pressed >= sum(action) >= 0)
            ]
            button_space = gym.spaces.Discrete(len(self.button_map))
        return button_space

    def __get_continuous_action_space(self):
        """
        Returns continuous action space: Box(float32.min, float32.max, (num_delta_buttons,), float32)
        """
        return gym.spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            (self.num_delta_buttons,),
            dtype=np.float32,
        )

    def __get_action_space(self):
        """
        Returns action space:
            if both binary and delta buttons defined in the config file, action space will be:
              ``Dict("binary": MultiDiscrete|Discrete, "continuous", Box)``
            else:
              action space will be only one of the following ``MultiDiscrete``|``Discrete``|``Box``
        """
        if self.num_delta_buttons == 0:
            return self.__get_binary_action_space()
        elif self.num_binary_buttons == 0:
            return self.__get_continuous_action_space()
        else:
            return gym.spaces.Dict(
                {
                    "binary": self.__get_binary_action_space(),
                    "continuous": self.__get_continuous_action_space(),
                }
            )

    def __get_observation_space(self):
        """
        Returns observation space: Dict with Box entry for each activated buffer:
          "screen", "depth", "labels", "automap", "gamevariables"
        """
        spaces = {
            "screen": gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.channels,
                ),
                dtype=np.uint8,
            )
        }

        if self.depth:
            spaces["depth"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )

        if self.labels:
            spaces["labels"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )

        if self.automap:
            spaces["automap"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    # "automap" buffer uses same number of channels
                    # as the main screen buffer,
                    self.channels,
                ),
                dtype=np.uint8,
            )

        self.num_game_variables = self.game.get_available_game_variables_size()
        if self.num_game_variables > 0:
            spaces["gamevariables"] = gym.spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (self.num_game_variables,),
                dtype=np.float32,
            )

        return gym.spaces.Dict(spaces)
