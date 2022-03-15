from typing import Optional
import warnings

import gym
import numpy as np
import pygame
import vizdoom.vizdoom as vzd


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
          "rgb"           = the RGB image (always available), in the format specified by the config file
          "depth"         = the depth image, if enabled by the config file
          "labels"        = the label image buffer, if enabled by the config file. For info on labels, access `env.state.labels` variable.
          "gamevariables" = all game variables, in the order specified by the config file

        Action space is always a Discrete one, one choice for each button (only one button can be pressed down at a time).
        """
        self.frame_skip = frame_skip

        # init game
        self.game = vzd.DoomGame()
        self.game.load_config(level)
        self.game.set_window_visible(False)
        self.game.init()
        self.state = None
        self.window_surface = None
        self.isopen = True

        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()

        allowed_buttons = []
        for button in self.game.get_available_buttons():
            if "DELTA" in button.name:
                warnings.warn(f"Removing button {button.name}. DELTA buttons are not supported. Use binary buttons instead.")
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
                    self.game.get_screen_channels(),
                ),
                # TODO not always true, but in most cases, yes...
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
        # convert action to vizdoom action space (one hot)
        act = [0 for _ in range(self.action_space.n)]
        act[action] = 1

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

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
            # TODO ensure that this works with GRAY8 images
            observation["rgb"] = np.transpose(self.state.screen_buffer, (1, 2, 0))
            if self.depth:
                observation["depth"] = self.state.depth_buffer
            if self.labels:
                observation["labels"] = self.state.labels_buffer
            if self.num_game_variables > 0:
                observation["gamevariables"] = self.state.game_variables.astype(np.float32)
        else:
            # there is no state in the terminal step, so a zero observation is returned instead
            for space_key, space_item in self.observation_space.items():
                observation[space_key] = np.zeros(space_item.shape, dtype=space_item.dtype)

        return observation

    def render(self, mode="human"):
        game_state = self.game.get_state()
        if game_state is None:
            img = np.zeros(
                (
                    self.game.get_screen_channels(),
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                ),
                dtype=np.uint8,
            )
        else:
            img = game_state.screen_buffer
        img = np.transpose(img, [2, 1, 0])

        if self.window_surface is None:
            pygame.init()
            pygame.display.set_caption("Vizdoom")
            if mode == "human":
                self.window_surface = pygame.display.set_mode(img.shape[:2])
            else:  # rgb_array
                self.window_surface = pygame.Surface(img.shape[:2])

        surf = pygame.surfarray.make_surface(img)
        self.window_surface.blit(surf, (0, 0))

        if mode == "human":
            pygame.display.update()

        if mode == "rgb_array":
            return img
        else:
            return self.isopen

    def close(self):
        if self.window_surface:
            pygame.quit()
            self.isopen = False
