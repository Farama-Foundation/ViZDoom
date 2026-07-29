from typing import Any, Dict

from pettingzoo.utils.wrappers import BaseParallelWrapper

from vizdoom.pettingzoo_wrapper.base_pettingzoo_env import INTERNAL_INFO_KEYS


def _strip(infos: Dict[str, Any]) -> Dict[str, Any]:
    for info in infos.values():
        if isinstance(info, dict):
            for key in INTERNAL_INFO_KEYS:
                info.pop(key, None)
    return infos


class InternalInfoFilter(BaseParallelWrapper):
    """
    Keys like _hidden_reset and reset_info (in INTERNAL_INFO_KEYS) are not necessary for pettingzoo
    """

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return obs, _strip(infos)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, rewards, terminations, truncations, _strip(infos)
