import math

import numpy as np
import wandb
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ParallelEnv


def _has_hidden_reset(infos):
    if not isinstance(infos, dict):
        return False
    for info in infos.values():
        if isinstance(info, dict) and info.get("_hidden_reset"):
            return True
    return False


class VideoLoggerParallelWrapper(wrappers.BaseParallelWrapper):
    def __init__(
            self,
            env: ParallelEnv,
            *,
            every_n_episodes: int = 50,
            fps: int = 35,
            max_frames: int = 1000,
    ):
        super().__init__(env)
        self.every_n = int(every_n_episodes)
        self.fps = int(fps)
        self.max_frames = int(max_frames)

        self._ep_idx = 0
        self._recording = False
        self._frames = []  # list of tiled HWC uint8 frames

    # ---- helpers ----
    def _maybe_enable(self):
        self._ep_idx += 1
        self._recording = (self.every_n > 0) and (self._ep_idx % self.every_n == 0)
        self._frames.clear()

    def _tile_ahwc(self, per_agent_ahwc: np.ndarray) -> np.ndarray:
        # per_agent_ahwc: (A,H,W,C)
        A, H, W, C = per_agent_ahwc.shape
        cols = int(math.ceil(math.sqrt(A)))
        rows = int(math.ceil(A / cols))
        canvas = np.zeros((rows * H, cols * W, C), dtype=per_agent_ahwc.dtype)
        for i in range(A):
            r, c = divmod(i, cols)
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = per_agent_ahwc[i]
        return canvas

    def _push_frame_from_obs(self, obs_dict):
        if not self._recording or not obs_dict:
            return
        # stack in a stable order (use self.agents at this step)
        agent_list = list(self.agents)
        frames = []
        for a in agent_list:
            x = obs_dict[a]  # AHWC, uint8 or float
            if x.dtype.kind == "f":
                x = np.clip(x, 0.0, 1.0)
                x = (x * 255.0).astype(np.uint8)
            else:
                x = x.astype(np.uint8, copy=False)
            frames.append(x)
        ahwc = np.stack(frames, axis=0)  # (A,H,W,C)
        tiled = self._tile_ahwc(ahwc)
        self._frames.append(tiled)
        if len(self._frames) > self.max_frames:
            self._frames.pop(0)

    def _finalize(self):
        if not self._recording or not self._frames:
            return
        arr = np.stack(self._frames, axis=0)  # THWC
        arr = np.transpose(arr, (0, 3, 1, 2))  # TCHW for wandb.Video
        try:
            wandb.log({f"videos/episode_{self._ep_idx}": wandb.Video(arr, fps=self.fps, format="mp4")})
        except Exception as e:
            print(f"[VideoLogger] wandb log failed: {e}")
        self._frames.clear()

    # ---- PZ ParallelEnv API ----
    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # sync agents list and episode counter
        self.agents = self.env.agents[:]
        self._maybe_enable()
        self._push_frame_from_obs(obs)
        return obs, info

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        self.agents = self.env.agents[:]

        if _has_hidden_reset(info):
            self._finalize()
            self._maybe_enable()

        self._push_frame_from_obs(obs)

        # end of episode if any agent done
        any_done = any(term.values()) or any(trunc.values())
        if any_done:
            self._finalize()
        return obs, rewards, term, trunc, info
