import os
from typing import Any, Optional

from gymnasium.utils import EzPickle

from vizdoom import scenarios_path
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv


class VizdoomScenarioEnv(VizdoomEnv, EzPickle):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""

    def __init__(
        self,
        scenario_config_file: str,
        frame_skip: int = 1,
        max_buttons_pressed: int = 0,
        render_mode: Optional[str] = None,
        treat_episode_timeout_as_truncation: bool = True,
        use_multi_binary_action_space: bool = True,
        **kwargs: Any,
    ):
        EzPickle.__init__(
            self,
            scenario_config_file,
            frame_skip,
            max_buttons_pressed,
            render_mode,
            treat_episode_timeout_as_truncation,
            use_multi_binary_action_space,
            **kwargs,
        )
        super().__init__(
            config_file=os.path.join(scenarios_path, scenario_config_file),
            frame_skip=frame_skip,
            max_buttons_pressed=max_buttons_pressed,
            render_mode=render_mode,
            treat_episode_timeout_as_truncation=treat_episode_timeout_as_truncation,
            use_multi_binary_action_space=use_multi_binary_action_space,
            **kwargs,
        )
