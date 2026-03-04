from __future__ import annotations

import time
from typing import Tuple, Dict

import numpy as np
import vizdoom as vzd


def parse_hw(res: str) -> Tuple[int, int]:
    w, h = res.lower().split("x")
    return int(w), int(h)


def get_screen_resolution(resolution: str) -> vzd.ScreenResolution:
    try:
        return getattr(vzd.ScreenResolution, f"RES_{resolution}")
    except AttributeError as e:
        raise ValueError(f"Invalid resolution: {resolution}")


def get_flat_game_vars(state, available_game_vars) -> Dict[str, float]:
    """Return game variables as flat scalars suitable for info (no nested dict)."""
    if state is None or state.game_variables is None:
        return {}
    game_variables = state.game_variables
    out: Dict[str, float] = {}
    n = min(len(available_game_vars), len(game_variables))
    for i in range(n):
        name = available_game_vars[i].name
        val = game_variables[i]
        out[name] = float(val)
    return out


def read_frame(state, resolution) -> np.ndarray:
    if state is not None and state.screen_buffer is not None:
        sb = state.screen_buffer  # (C,H,W) or (H,W)
        return np.transpose(sb, (1, 2, 0)) if sb.ndim == 3 else sb[..., None]
    # fallback to zeros if no frame
    h, w = parse_hw(resolution)[1], parse_hw(resolution)[0]
    return np.zeros((h, w, 3), dtype=np.uint8)


def discover_buttons(config_path: str) -> Tuple[int, int]:
    game = vzd.DoomGame()
    game.load_config(config_path)
    game.set_window_visible(False)
    delta, binary = [], []
    for b in game.get_available_buttons():
        if vzd.is_delta_button(b) and b not in delta:
            delta.append(b)
        else:
            binary.append(b)
    game.close()
    return len(delta), len(binary)


def wait_for_child_init(idx: int, pipe, timeout_sec: float = 90.0):
    start_time = time.time()
    status_msgs = []
    last_msg_time_start = start_time

    while True:
        elapsed = time.time() - start_time
        last_msg_time_end = time.time() - last_msg_time_start

        # Warn if no progress
        if elapsed > timeout_sec:
            raise TimeoutError(
                f"Agent {idx} init timeout after {timeout_sec}s"
                f"Status msg: {status_msgs}"
                f"Last message {last_msg_time_end:.1f}s ago"
            )

        # Log progress
        if elapsed > 30 and elapsed % 10 < 1.1:
            print(f"Agent {idx} still initializing. {elapsed:.0f}s elapsed so far)")

        # Avoid recv() block, let data arrive, wait up to 1s for data
        if pipe.poll(timeout=1.0):
            try:
                msg = pipe.recv()
                status_msgs.append(msg)
                last_msg_time_start = time.time()

                if msg.get("status") == "ready":
                    return True
                elif msg.get("status") == "init_failed":
                    raise RuntimeError(f"Agent {idx} init failed: {msg.get('error', 'unknown')}")
            except (EOFError, BrokenPipeError) as e:
                # Child process died
                raise RuntimeError(f"Agent {idx} process died init. Status msg: {status_msgs}") from e
            except Exception as e:
                raise RuntimeError(f"error from {idx}: {e}")


def sync_agent_init(pipes_parent, procs):
    for i, pipe in enumerate(pipes_parent):
        try:
            # Host first and then wait for all children sequentially
            role = "host" if i == 0 else f"peer {i}"
            print(f"Waiting for agent {i} ({role}) to init")
            wait_for_child_init(i, pipe, timeout_sec=90.0)  # 90s = 45s connection timeout + 30s buffer + max 15s init
            print(f"Agent {i} ({role}) ready")
        except Exception as e:
            print(f"Agent {i} init failed: {e}")
            # Cleanup due to failed init
            for j, p in enumerate(procs):
                if p.is_alive():
                    try:
                        p.terminate()
                    except:
                        pass

            time.sleep(0.5)  # Wait for termination

            for j, p in enumerate(procs):
                if p.is_alive():
                    try:
                        p.kill()
                    except:
                        pass
                try:
                    p.join(timeout=1.0)
                except:
                    pass

            raise RuntimeError(".") from e
