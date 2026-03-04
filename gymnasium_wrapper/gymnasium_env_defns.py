"""
Copyright (C) 2023 - 2026 Farama Foundation, and the respective contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

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
