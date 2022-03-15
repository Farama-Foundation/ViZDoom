from typing import List, Optional
import warnings

import gym
import numpy as np
import pygame
import vizdoom.vizdoom as vzd
from gym import spaces


class VizdoomEnv(gym.Env):
    def __init__(
        self,
        level,
        frame_skip=1,
        depth=False,
        labels=False,
        position=False,
        health=False,
    ):
        """
        Base class for Gym interface for ViZDoom. Thanks to https://github.com/shakenes/vizdoomgym
        Child classes are defined in vizdoom_env_definitions.py,
        depth buffer and 'labels' will render the object labels and return it in the observation.
        Note that the observation will be a list with the screen buffer as the first element.
        """

        # parse keyword arguments
        self.frame_skip = frame_skip
        self.depth = depth
        self.labels = labels
        self.position = position
        self.health = health

        # init game
        self.game = vzd.DoomGame()
        self.game.load_config(level)
        self.game.set_window_visible(False)
        self.game.set_depth_buffer_enabled(self.depth)
        self.game.set_labels_buffer_enabled(self.labels)
        self.game.clear_available_game_variables()
        if self.position:
            self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
            self.game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
            self.game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
            self.game.add_available_game_variable(vzd.GameVariable.ANGLE)
        if self.health:
            self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.init()
        self.state = None
        self.window_surface = None
        self.isopen = True

        allowed_buttons = []
        for button in self.game.get_available_buttons():
            if "DELTA" in button.name:
                warnings.warn(f"Removing button {button.name}. DELTA buttons are not supported. Use binary buttons instead.")
            else:
                allowed_buttons.append(button)
        self.game.set_available_buttons(allowed_buttons)
        self.action_space = spaces.Discrete(len(allowed_buttons))

        # specify observation space(s)
        list_spaces: List[gym.Space] = [
            spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.game.get_screen_channels(),
                ),
                dtype=np.uint8,
            )
        ]
        if self.depth:
            list_spaces.append(
                spaces.Box(
                    0,
                    255,
                    (
                        self.game.get_screen_height(),
                        self.game.get_screen_width(),
                    ),
                    dtype=np.uint8,
                )
            )
        if self.labels:
            list_spaces.append(
                spaces.Box(
                    0,
                    255,
                    (
                        self.game.get_screen_height(),
                        self.game.get_screen_width(),
                    ),
                    dtype=np.uint8,
                )
            )
        if self.position:
            list_spaces.append(
                spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, (4,))
            )
        if self.health:
            list_spaces.append(spaces.Box(0, np.finfo(np.float32).max, (1,)))
        if len(list_spaces) == 1:
            self.observation_space = list_spaces[0]
        else:
            self.observation_space = spaces.Tuple(list_spaces)

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
        observation = []
        if self.state is not None:
            observation.append(np.transpose(self.state.screen_buffer, (1, 2, 0)))
            if self.depth:
                observation.append(self.state.depth_buffer)
            if self.labels:
                observation.append(self.state.labels_buffer)
            if self.position:
                observation.append(
                    np.array(
                        [self.state.game_variables[i] for i in range(4)],
                        dtype=np.float32,
                    )
                )
                if self.health:
                    observation.append(
                        np.array([self.state.game_variables[4]], dtype=np.float32)
                    )
            elif self.health:
                observation.append(
                    np.array([self.state.game_variables[0]], dtype=np.float32)
                )
        else:
            # there is no state in the terminal step, so a zero observation is returned instead
            if isinstance(self.observation_space, gym.spaces.box.Box):
                # Box isn't iterable
                obs_space = [self.observation_space]
            else:
                obs_space = self.observation_space

            for space in obs_space:
                observation.append(np.zeros(space.shape, dtype=space.dtype))

        # if there is only one observation, return obs as array to sustain compatibility
        if len(observation) == 1:
            observation = observation[0]
        return observation

    def render(self, mode="human"):
        game_state = self.game.get_state()
        if game_state is None:
            img = np.zeros(
                (
                    self.game.get_screen_channels(),
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                )
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
