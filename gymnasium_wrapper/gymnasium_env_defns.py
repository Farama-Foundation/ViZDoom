import os

from vizdoom import scenarios_path
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv


class VizdoomScenarioEnv(VizdoomEnv):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""

    def __init__(
        self, scenario_file, frame_skip=1, max_buttons_pressed=1, render_mode=None
    ):
        super().__init__(
            os.path.join(scenarios_path, scenario_file),
            frame_skip,
            max_buttons_pressed,
            render_mode,
        )
