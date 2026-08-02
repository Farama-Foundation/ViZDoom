from __future__ import annotations

import fcntl
import os
import random
import socket
import tempfile
import time
from contextlib import contextmanager

import numpy as np

import vizdoom as vzd


def parse_hw(res: str) -> tuple[int, int]:
    w, h = res.lower().split("x")
    return int(w), int(h)


def get_screen_resolution(resolution: str) -> vzd.ScreenResolution:
    try:
        return getattr(vzd.ScreenResolution, f"RES_{resolution}")
    except AttributeError as e:
        raise ValueError(f"Invalid resolution: {resolution}. Error: {e}")


def get_live_game_vars(game, available_game_vars) -> dict[str, float]:
    """
    Return game variables read straight from the engine, as flat scalars.
    Prefer this over `get_flat_game_vars` as `get_state()` returns None
    once the episode is finished, which drops all variable from terminal step info.
    """
    return {
        variable.name: float(game.get_game_variable(variable))
        for variable in available_game_vars
    }


def get_flat_game_vars(state, available_game_vars) -> dict[str, float]:
    """Return game variables as flat scalars suitable for info (no nested dict).
    Returns {} on finished episode (see get_live_game_vars).
    """
    if state is None or state.game_variables is None:
        return {}
    game_variables = state.game_variables
    out: dict[str, float] = {}
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


def discover_buttons(config_path: str) -> tuple[vzd.Button, ...]:
    game = vzd.DoomGame()
    game.load_config(config_path)
    game.set_window_visible(False)
    buttons = tuple(game.get_available_buttons())
    game.close()
    return buttons


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
                    raise RuntimeError(
                        f"Agent {idx} init failed: {msg.get('error', 'unknown')}"
                    )
            except (EOFError, BrokenPipeError) as e:
                # Child process died
                raise RuntimeError(
                    f"Agent {idx} process died init. Status msg: {status_msgs}"
                ) from e
            except Exception as e:
                raise RuntimeError(f"error from {idx}: {e}")


def sync_agent_init(pipes_parent, procs):
    for i, pipe in enumerate(pipes_parent):
        try:
            # Host first and then wait for all children sequentially
            role = "host" if i == 0 else f"peer {i}"
            print(f"Waiting for agent {i} ({role}) to init")
            wait_for_child_init(
                i, pipe, timeout_sec=90.0
            )  # 90s = 45s connection timeout + 30s buffer + max 15s init
            print(f"Agent {i} ({role}) ready")
        except Exception as e:
            print(f"Agent {i} init failed: {e}")
            # Cleanup due to failed init
            for j, p in enumerate(procs):
                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass

            time.sleep(0.5)  # Wait for termination

            for j, p in enumerate(procs):
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass

            raise RuntimeError(".") from e


def is_udp_port_available(host_address: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host_address, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _port_lock_path(port: int) -> str:
    return os.path.join(tempfile.gettempdir(), f"vizdoom_udp_port_{port}.lock")


def _init_lock_path(slot: int) -> str:
    return os.path.join(tempfile.gettempdir(), f"vizdoom_multi_init_{slot}.lock")


_MIN_UDP_PORT = 1024
_MAX_UDP_PORT = 65535
_MAX_PORT_PROBES = 4096


def _candidate_ports(first: int, floor: int, span: int, increment: int):
    if increment > 1:
        for i in range(span // increment + 1):
            yield floor + (first - floor + i * increment) % span
    for i in range(span):
        yield floor + (first - floor + i) % span


@contextmanager
def reserve_udp_port(
    host_address: str,
    start_port: int,
    increment: int = 1,
    floor_port: int | None = None,
):
    increment = max(1, int(increment))
    floor = _MIN_UDP_PORT if floor_port is None else max(_MIN_UDP_PORT, int(floor_port))
    span = _MAX_UDP_PORT - floor + 1
    if span < 1:
        raise ValueError(f"No usable UDP ports at or above {floor}")
    first = floor + (int(start_port) - floor) % span

    probes = 0
    for port in _candidate_ports(first, floor, span, increment):
        if probes >= _MAX_PORT_PROBES:
            break
        probes += 1
        lock_fd = os.open(_port_lock_path(port), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue

            if is_udp_port_available(host_address, port):
                try:
                    yield port
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                return
        finally:
            os.close(lock_fd)

    raise RuntimeError(
        f"Could not reserve an available UDP port after {probes} probes "
        f"(hint={start_port}, searched {floor}-{_MAX_UDP_PORT})"
    )


@contextmanager
def reserve_init_slot(max_parallel: int = 20, timeout_sec: float = 10.0):
    if max_parallel <= 1:
        yield None
        return

    deadline = time.time() + timeout_sec
    slot_indices = list(range(int(max_parallel)))
    random.shuffle(slot_indices)

    while time.time() < deadline:
        for slot in slot_indices:
            lock_fd = os.open(_init_lock_path(slot), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue

                try:
                    yield slot
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                return
            finally:
                os.close(lock_fd)

        time.sleep(0.1)

    raise TimeoutError(
        f"Could not acquire a ViZDoom multiplayer init slot within {timeout_sec:.1f}s"
    )
