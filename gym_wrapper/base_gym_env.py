from typing import Optional
import warnings

import gym
import numpy as np
import pygame
import vizdoom.vizdoom as vzd

# A fixed set of colors for each potential label
# for rendering an image.
# 256 is not nearly enough for all IDs, but we limit
# ourselves here to avoid hogging too much memory.
LABEL_COLORS = np.random.default_rng(42).uniform(25, 256, size=(256, 3)).astype(np.uint8)


class VizdoomEnv(gym.Env):
    def __init__(
        self,
        level,
        frame_skip=1,
    ):
        """
        Base class for Gym interface for ViZDoom. Thanks to https://github.com/shakenes/vizdoomgym
        Child classes are defined in vizdoom_env_definitions.py,

        Arguments:
            level (str): path to the config file to load. Most settings should be set by this config file.
            frame_skip (int): how many frames should be advanced per action. 1 = take action on every frame. Default: 1.

        This environment forces window to be hidden. Use `render()` function to see the game.

        Observations are dictionaries with different amount of entries, depending on if depth/label buffers were
        enabled in the config file:
          "rgb"           = the RGB image (always available) in shape (HEIGHT, WIDTH, CHANNELS)
          "depth"         = the depth image in shape (HEIGHT, WIDTH), if enabled by the config file,
          "labels"        = the label image buffer in shape (HEIGHT, WIDTH), if enabled by the config file. For info on labels, access `env.state.labels` variable.
          "automap"       = the automap image buffer in shape (HEIGHT, WIDTH, CHANNELS), if enabled by the config file
          "gamevariables" = all game variables, in the order specified by the config file

        Action space is always a Discrete one, one choice for each button (only one button can be pressed down at a time).
        """
        self.frame_skip = frame_skip

        # init game
        self.game = vzd.DoomGame()
        self.game.load_config(level)
        self.game.set_window_visible(False)

        screen_format = self.game.get_screen_format()
        if screen_format != vzd.ScreenFormat.RGB24:
            warnings.warn(f"Detected screen format {screen_format.name}. Only RGB24 is supported in the Gym wrapper. Forcing RGB24.")
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        self.game.init()
        self.state = None
        self.window_surface = None
        self.isopen = True

        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()
        self.automap = self.game.is_automap_buffer_enabled()

        allowed_buttons = []
        for button in self.game.get_available_buttons():
            if "DELTA" in button.name:
                warnings.warn(f"Removing button {button.name}. DELTA buttons are currently not supported in Gym wrapper. Use binary buttons instead.")
            else:
                allowed_buttons.append(button)
        self.game.set_available_buttons(allowed_buttons)
        self.action_space = gym.spaces.Discrete(len(allowed_buttons))

        # specify observation space(s)
        spaces = {
            "rgb": gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    3,
                ),
                dtype=np.uint8,
            )
        }

        if self.depth:
            spaces["depth"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                ),
                dtype=np.uint8,
            )

        if self.labels:
            spaces["labels"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                ),
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
                    # as the main screen buffer
                    3,
                ),
                dtype=np.uint8,
            )

        self.num_game_variables = self.game.get_available_game_variables_size()
        if self.num_game_variables > 0:
            spaces["gamevariables"] = gym.spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (self.num_game_variables,),
                dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call `reset` before using `step` method."

        # convert action to vizdoom action space (one hot)
        act = [0 for _ in range(self.action_space.n)]
        act[action] = 1

        reward = self.game.make_action(act, self.frame_skip)
        self.state = self.game.get_state()
        done = self.game.is_episode_finished()

        return self.__collect_observations(), reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.state = self.game.get_state()

        if not return_info:
            return self.__collect_observations()
        else:
            return self.__collect_observations(), {}

    def __collect_observations(self):
        observation = {}
        if self.state is not None:
            observation["rgb"] = self.state.screen_buffer
            if self.depth:
                observation["depth"] = self.state.depth_buffer
            if self.labels:
                observation["labels"] = self.state.labels_buffer
            if self.automap:
                observation["automap"] = self.state.automap_buffer
            if self.num_game_variables > 0:
                observation["gamevariables"] = self.state.game_variables.astype(np.float32)
        else:
            # there is no state in the terminal step, so a zero observation is returned instead
            for space_key, space_item in self.observation_space.spaces.items():
                observation[space_key] = np.zeros(space_item.shape, dtype=space_item.dtype)

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

        if self.depth:
            image_list.append(np.repeat(game_state.depth_buffer[..., None], repeats=3, axis=2))

        if self.labels:
            # Give each label a fixed color.
            # We need to connect each pixel in labels_buffer to the corresponding
            # id via `value``
            labels_rgb = np.zeros_like(game_state.screen_buffer)
            labels_buffer = game_state.labels_buffer
            for label in game_state.labels:
                color = LABEL_COLORS[label.object_id % 256]
                labels_rgb[labels_buffer == label.value] = color
            image_list.append(labels_rgb)

        if self.automap:
            image_list.append(game_state.automap_buffer)

        return np.concatenate(image_list, axis=1)

    def render(self, mode="human"):
        render_image = self.__build_human_render_image()
        if mode == "rgb_array":
            return render_image
        elif mode == "human":
            # Transpose image (pygame wants (width, height, channels), we have (height, width, channels))
            render_image = render_image.transpose(1, 0, 2)
            if self.window_surface is None:
                pygame.init()
                pygame.display.set_caption("ViZDoom")
                self.window_surface = pygame.display.set_mode(render_image.shape[:2])

            surf = pygame.surfarray.make_surface(render_image)
            self.window_surface.blit(surf, (0, 0))
            pygame.display.update()
        else:
            return self.isopen

    def close(self):
        if self.window_surface:
            pygame.quit()
            self.isopen = False
