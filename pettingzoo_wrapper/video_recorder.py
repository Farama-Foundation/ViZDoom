import math
import os
import tempfile

import imageio.v2 as imageio
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

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype.kind == "f":
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255.0).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8, copy=False)

        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] > 3:
            frame = frame[:, :, :3]
        return np.ascontiguousarray(frame)

    def _tile_ahwc(self, per_agent_ahwc: np.ndarray) -> np.ndarray:
        # per_agent_ahwc: (A,H,W,C)
        A, H, W, C = per_agent_ahwc.shape
        cols = int(math.ceil(math.sqrt(A)))
        rows = int(math.ceil(A / cols))
        canvas = np.zeros((rows * H, cols * W, C), dtype=per_agent_ahwc.dtype)
        for i in range(A):
            r, c = divmod(i, cols)
            canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = per_agent_ahwc[i]
        return canvas

    def _push_frame(self, frame: np.ndarray | None):
        if not self._recording or frame is None:
            return
        self._frames.append(self._normalize_frame(frame))
        if len(self._frames) > self.max_frames:
            self._frames.pop(0)

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
        self._push_frame(tiled)

    def _capture_frame(self, obs_dict):
        if getattr(self.env, "render_mode", None) == "rgb_array":
            frame = self.env.render()
            if frame is not None:
                self._push_frame(frame)
                return
        self._push_frame_from_obs(obs_dict)

    def _write_video(self, path: str):
        writer = imageio.get_writer(
            path,
            fps=float(self.fps),
            quality=8,
            macro_block_size=1,
        )
        try:
            for frame in self._frames:
                writer.append_data(frame)
        finally:
            writer.close()

    def _upload_to_wandb(self, path: str):
        if getattr(wandb, "run", None) is None:
            return

        key = f"videos/episode_{self._ep_idx:06d}"
        try:
            wandb.log({key: wandb.Video(path, format="mp4")})
        except Exception as e:
            print(f"[VideoLogger] wandb log failed: {e}")

    def _finalize(self):
        if not self._recording or not self._frames:
            return
        if not (getattr(wandb, "run", None) is not None):
            self._frames.clear()
            return
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                temp_path = tmp.name
            self._write_video(temp_path)
            self._upload_to_wandb(temp_path)
        except Exception as e:
            print(f"[VideoLogger] finalize failed: {e}")
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"[VideoLogger] cleanup failed: {e}")
        self._frames.clear()

    # ---- PZ ParallelEnv API ----
    @property
    def state_space(self):
        return self.env.state_space

    def state(self):
        return self.env.state()

    def state_observation(self, agent: str):
        return self.env.state_observation(agent)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if self._frames:
            self._finalize()
        obs, info = self.env.reset(seed=seed, options=options)
        # sync agents list and episode counter
        self.agents = self.env.agents[:]
        self._maybe_enable()
        self._capture_frame(obs)
        return obs, info

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        self.agents = self.env.agents[:]

        if _has_hidden_reset(info):
            self._finalize()
            self._maybe_enable()

        self._capture_frame(obs)

        # end of episode if any agent done
        any_done = any(term.values()) or any(trunc.values())
        if any_done:
            self._finalize()
        return obs, rewards, term, trunc, info

    def close(self):
        if self._frames:
            self._finalize()
        return self.env.close()
