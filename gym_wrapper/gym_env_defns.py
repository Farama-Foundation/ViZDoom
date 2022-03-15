import os
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv
from vizdoom import scenarios_path


class VizdoomScenarioEnv(VizdoomEnv):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""
    def __init__(
        self, scenario_file, frame_skip=1
    ):
        super(VizdoomScenarioEnv, self).__init__(
           os.path.join(scenarios_path, scenario_file), frame_skip
        )
