from __future__ import annotations

import multiprocessing as mp
import queue
import random
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection

from vizdoom.pettingzoo_wrapper.base_env_common import (
    VizdoomParallelEnvBase,
    configure_doom_game,
)
from vizdoom.pettingzoo_wrapper.utils import (
    get_flat_game_vars,
    read_frame,
    reserve_init_slot,
    reserve_udp_port,
)


ctx = mp.get_context("fork")
_INIT_TIMEOUT = 90.0
_STEP_TIMEOUT = 30.0
_RESET_TIMEOUT = 30.0
_HIDDEN_RECOVERY_ATTEMPTS = 1
_INIT_STAGGER_SEC = 0.05
_MAX_PARALLEL_INIT = 20
_MAX_INIT_ATTEMPTS = 5
_INIT_PORT_STRIDE = 1000


@dataclass(frozen=True)
class _Task:
    cmd: str
    action: list[float] | None = None
    port: int | None = None
    tics: int = 1


class _AgentWorkerCrashed(RuntimeError):
    pass


def _agent_worker_thread(
    *,
    task_queue: queue.Queue[_Task],
    result_queue: queue.Queue[dict],
    step_barrier: threading.Barrier,
    config_path: str,
    resolution: str,
    timeout: int | None,
    num_agents: int,
    agent_id: int,
    is_host: bool,
    host_address: str,
    async_mode: bool,
    netmode: int,
    ticrate: int,
    seed: int | None,
    verbose: bool,
) -> None:
    game = None
    available_game_vars = []
    frames_advanced = 0

    def _close_game() -> None:
        nonlocal game
        if game is not None:
            try:
                game.close()
            except Exception:
                pass
            game = None

    try:
        while True:
            task = task_queue.get()
            if task.cmd == "close":
                break

            try:
                if task.cmd == "init":
                    _close_game()
                    if task.port is None:
                        raise ValueError("init task requires a port")
                    game = configure_doom_game(
                        config_path=config_path,
                        resolution=resolution,
                        ticrate=ticrate,
                        async_mode=async_mode,
                        timeout=timeout,
                        seed=seed,
                        is_host=is_host,
                        num_agents=num_agents,
                        host_address=host_address,
                        port=int(task.port),
                        netmode=netmode,
                        agent_idx=agent_id,
                    )
                    if not is_host:
                        time.sleep(0.5 + random.uniform(0.5, 1.0))
                    game.init()
                    available_game_vars = game.get_available_game_variables()
                    frames_advanced = 0
                    result_queue.put({"status": "ready"})
                    continue

                if game is None:
                    raise RuntimeError(
                        f"Agent {agent_id} received {task.cmd} before init"
                    )

                if task.cmd == "reset":
                    game.new_episode()
                    state = game.get_state()
                    info = {
                        "num_frames": 1,
                        "player_dead": bool(game.is_player_dead()),
                        "just_died": False,
                        "step": 0,
                    }
                    info.update(get_flat_game_vars(state, available_game_vars))
                    frames_advanced = 0
                    result_queue.put(
                        {
                            "obs": read_frame(state, resolution),
                            "reward": 0.0,
                            "terminated": False,
                            "truncated": False,
                            "info": info,
                        }
                    )
                    continue

                if task.cmd != "step":
                    raise ValueError(f"Unknown task {task.cmd}")

                action = task.action if task.action is not None else []
                tics = max(1, int(task.tics))
                was_dead_before = game.is_player_dead()
                game.set_action(action)
                for tic in range(tics):
                    update_state = tic == tics - 1
                    game.advance_action(1, update_state)
                    if not update_state:
                        # Keep synchronous multiplayer peers on the same tic
                        step_barrier.wait(timeout=_STEP_TIMEOUT)
                reward = float(game.get_last_reward())
                is_dead = game.is_player_dead()
                just_died = (not was_dead_before) and is_dead
                episode_finished = bool(game.is_episode_finished())
                truncated = episode_finished and bool(game.is_episode_timeout_reached())
                terminated = episode_finished and not truncated
                frames_advanced += tics
                state = game.get_state()
                info = {
                    "num_frames": 1,
                    "player_dead": is_dead,
                    "just_died": just_died,
                    "step": frames_advanced,
                }
                info.update(get_flat_game_vars(state, available_game_vars))
                result_queue.put(
                    {
                        "obs": read_frame(state, resolution),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    }
                )
            except Exception as exc:
                result_queue.put(
                    {"status": "crashed", "error": f"{type(exc).__name__}: {exc}"}
                )
                break
    finally:
        _close_game()


class _AgentWorkerCoordinator:
    def __init__(
        self,
        *,
        config_path: str,
        resolution: str,
        timeout: int | None,
        skip_frames: int,
        num_agents: int,
        host_address: str,
        port: int,
        slot_index: int,
        async_mode: bool,
        netmode: int,
        ticrate: int,
        seed: int | None,
        verbose: bool,
    ) -> None:
        self.config_path = config_path
        self.resolution = resolution
        self.timeout = timeout
        self.skip_frames = max(1, int(skip_frames))
        self.num_agents = int(num_agents)
        self.host_address = host_address
        self.base_port = int(port)
        self.slot_index = int(slot_index)
        self.async_mode = bool(async_mode)
        self.netmode = int(netmode)
        self.ticrate = int(ticrate)
        self.seed = seed
        self.verbose = bool(verbose)
        self._agent_worker_task_queues: list[queue.Queue[_Task]] = []
        self._agent_worker_result_queues: list[queue.Queue[dict]] = []
        self._agent_worker_threads: list[threading.Thread] = []
        self._step_barrier: threading.Barrier | None = None
        self._port = self.base_port
        self._init_attempts = 0
        self._initialize_agent_workers()

    def _close_agent_worker_threads(self) -> None:
        if self._step_barrier is not None:
            self._step_barrier.abort()
        for task_queue in self._agent_worker_task_queues:
            try:
                task_queue.put(_Task("close"))
            except Exception:
                pass
        for thread in self._agent_worker_threads:
            thread.join(timeout=1.0)
        self._agent_worker_task_queues.clear()
        self._agent_worker_result_queues.clear()
        self._agent_worker_threads.clear()
        self._step_barrier = None

    def _spawn_agent_worker_threads(self) -> None:
        self._step_barrier = threading.Barrier(self.num_agents)
        for agent_id in range(self.num_agents):
            task_queue: queue.Queue[_Task] = queue.Queue()
            result_queue: queue.Queue[dict] = queue.Queue()
            thread = threading.Thread(
                target=_agent_worker_thread,
                kwargs=dict(
                    task_queue=task_queue,
                    result_queue=result_queue,
                    step_barrier=self._step_barrier,
                    config_path=self.config_path,
                    resolution=self.resolution,
                    timeout=self.timeout,
                    num_agents=self.num_agents,
                    agent_id=agent_id,
                    is_host=(agent_id == 0),
                    host_address=self.host_address,
                    async_mode=self.async_mode,
                    netmode=self.netmode,
                    ticrate=self.ticrate,
                    seed=(None if self.seed is None else int(self.seed) + agent_id),
                    verbose=self.verbose,
                ),
                daemon=True,
            )
            thread.start()
            self._agent_worker_task_queues.append(task_queue)
            self._agent_worker_result_queues.append(result_queue)
            self._agent_worker_threads.append(thread)

    def _initialize_agent_workers(self) -> None:
        last_error: Exception | None = None
        for init_attempt in range(1, _MAX_INIT_ATTEMPTS + 1):
            self._close_agent_worker_threads()
            self._spawn_agent_worker_threads()
            requested_port = self.base_port + (init_attempt - 1) * _INIT_PORT_STRIDE
            try:
                with reserve_init_slot(max_parallel=_MAX_PARALLEL_INIT):
                    with reserve_udp_port(
                        self.host_address,
                        requested_port,
                        increment=_INIT_PORT_STRIDE,
                    ) as port:
                        self._port = int(port)
                        for agent_id, task_queue in enumerate(
                            self._agent_worker_task_queues
                        ):
                            task_queue.put(_Task("init", port=self._port))
                            if agent_id + 1 < self.num_agents:
                                time.sleep(_INIT_STAGGER_SEC)
                        self._await_startup()
                self._init_attempts = init_attempt
                return
            except Exception as exc:
                last_error = exc
                self._close_agent_worker_threads()
                if init_attempt < _MAX_INIT_ATTEMPTS:
                    time.sleep(min(1.0, 0.2 * init_attempt))
        raise RuntimeError(
            "Failed to initialize COMRAD agent workers after retries "
            f"(slot={self.slot_index}, seed={self.seed}, base_port={self.base_port}): {last_error}"
        )

    def _await_startup(self) -> None:
        deadline = time.time() + _INIT_TIMEOUT
        for agent_id, result_queue in enumerate(self._agent_worker_result_queues):
            timeout = max(0.1, deadline - time.time())
            try:
                result = result_queue.get(timeout=timeout)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Agent {agent_id} init timeout after {_INIT_TIMEOUT:.1f}s"
                ) from exc
            if result.get("status") == "ready":
                continue
            raise RuntimeError(
                f"Agent {agent_id} init failed: {result.get('error', 'unknown error')}"
            )

    def _dispatch(self, tasks: list[_Task], timeout: float) -> list[dict]:
        for task_queue, task in zip(self._agent_worker_task_queues, tasks):
            task_queue.put(task)

        results: list[dict] = []
        deadline = time.time() + timeout
        for agent_id, result_queue in enumerate(self._agent_worker_result_queues):
            remaining = max(0.1, deadline - time.time())
            try:
                result = result_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Agent {agent_id} {tasks[agent_id].cmd} timeout after {timeout:.1f}s"
                ) from exc
            if isinstance(result, dict) and result.get("status") == "crashed":
                raise _AgentWorkerCrashed(
                    result.get("error", f"agent {agent_id} crashed")
                )
            results.append(result)
        return results

    def reset(self) -> dict:
        tasks = [_Task("reset") for _ in range(self.num_agents)]
        results = self._dispatch(tasks, _RESET_TIMEOUT)
        observations = {}
        infos = {}
        for agent_id, result in enumerate(results):
            agent_name = f"agent_{agent_id}"
            observations[agent_name] = result["obs"]
            infos[agent_name] = result["info"]
        return {"observations": observations, "infos": infos}

    def step(self, actions: list[list[float]]) -> dict:
        results = self._dispatch(
            [_Task("step", action=action, tics=self.skip_frames) for action in actions],
            _STEP_TIMEOUT,
        )
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent_id, result in enumerate(results):
            agent_name = f"agent_{agent_id}"
            observations[agent_name] = result["obs"]
            rewards[agent_name] = float(result["reward"])
            terminated = bool(result.get("terminated", False))
            truncated = bool(result.get("truncated", False))
            terminations[agent_name] = terminated
            truncations[agent_name] = truncated
            info = result["info"]
            info["num_frames"] = self.skip_frames
            infos[agent_name] = info

        return {
            "observations": observations,
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
            "infos": infos,
        }

    def debug_status(self) -> dict:
        return {
            "slot_index": self.slot_index,
            "seed": self.seed,
            "base_port": self.base_port,
            "port": self._port,
            "init_attempts": self._init_attempts,
        }

    def close(self) -> None:
        self._close_agent_worker_threads()


def _agent_worker_process_main(conn: Connection, kwargs: dict) -> None:
    coordinator = None
    try:
        coordinator = _AgentWorkerCoordinator(**kwargs)
        conn.send({"status": "ready", **coordinator.debug_status()})
        while True:
            try:
                message = conn.recv()
            except EOFError:
                break
            cmd = message.get("cmd")
            if cmd == "close":
                conn.send({"status": "closed"})
                break
            if cmd == "reset":
                conn.send({"status": "ok", **coordinator.reset()})
                continue
            if cmd == "step":
                conn.send({"status": "ok", **coordinator.step(message["actions"])})
                continue
            raise ValueError(f"Unknown parent command: {cmd}")
    except Exception as exc:
        try:
            conn.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        if coordinator is not None:
            coordinator.close()
        try:
            conn.close()
        except Exception:
            pass


class VizdoomParallelEnv(VizdoomParallelEnvBase):
    def __init__(self, **kwargs) -> None:
        barrier_timeout = kwargs.pop("barrier_timeout", _STEP_TIMEOUT)
        self._barrier_timeout = (
            _STEP_TIMEOUT if barrier_timeout is None else float(barrier_timeout)
        )
        self._daemon = bool(kwargs.get("daemon", True))
        self._slot_index = int(kwargs.pop("slot_index", 0))
        super().__init__(**kwargs)
        self._parent_conn: Connection | None = None
        self._process: ctx.Process | None = None
        self._pending_reset_infos: dict[str, dict] | None = None
        self._pending_hidden_reset = False
        self._current_agent_worker_port = int(self.port)
        self._last_init_attempts = 0
        self._last_recovery_phase = "none"
        self._spawn_agent_worker_process()

    def _spawn_agent_worker_process(self) -> None:
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        process = ctx.Process(
            target=_agent_worker_process_main,
            kwargs=dict(
                conn=child_conn,
                kwargs=dict(
                    config_path=self.config_file,
                    resolution=self.resolution,
                    timeout=self._timeout,
                    skip_frames=self._skip_frames,
                    num_agents=self._num_agents,
                    host_address=self.host_address,
                    port=self.port,
                    slot_index=self._slot_index,
                    async_mode=self.async_mode,
                    netmode=self.netmode,
                    ticrate=self.ticrate,
                    seed=self._ext_seed,
                    verbose=self.verbose,
                ),
            ),
            daemon=self._daemon,
        )
        process.start()
        self._parent_conn = parent_conn
        self._process = process
        message = self._recv_message(timeout=_INIT_TIMEOUT)
        self._current_agent_worker_port = int(message.get("port", self.port))
        self._last_init_attempts = int(message.get("init_attempts", 0))

    def _recv_message(self, timeout: float) -> dict:
        if self._parent_conn is None:
            raise RuntimeError("Agent worker process is not available")
        if not self._parent_conn.poll(timeout):
            exitcode = None if self._process is None else self._process.exitcode
            raise TimeoutError(
                f"Agent worker process timeout after {timeout:.1f}s (exitcode={exitcode})"
            )
        message = self._parent_conn.recv()
        status = message.get("status")
        if status in {"ready", "ok", "closed"}:
            return message
        raise RuntimeError(message.get("error", "agent worker process failed"))

    def _shutdown_agent_worker_process(self) -> None:
        if self._parent_conn is not None:
            try:
                self._parent_conn.send({"cmd": "close"})
                if self._parent_conn.poll(timeout=1.0):
                    self._parent_conn.recv()
            except Exception:
                pass
            try:
                self._parent_conn.close()
            except Exception:
                pass
            self._parent_conn = None

        if self._process is not None:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
            self._process = None

    def _restart_agent_worker_process(self) -> None:
        self._shutdown_agent_worker_process()
        self._spawn_agent_worker_process()

    def _hidden_reset_after_recovery(self) -> None:
        assert self._parent_conn is not None
        self._parent_conn.send({"cmd": "reset"})
        message = self._recv_message(timeout=_RESET_TIMEOUT)
        observations = message["observations"]
        infos = message["infos"]
        self._last_frames = dict(observations)
        self.agents = self.possible_agents[:]
        self._pending_reset_infos = infos
        self._pending_hidden_reset = True

    def _attach_pending_reset_infos(self, infos: dict[str, dict]) -> None:
        if self._pending_reset_infos is None or not self._pending_hidden_reset:
            return

        pending_reset_infos = self._pending_reset_infos
        self._pending_reset_infos = None
        self._pending_hidden_reset = False

        for agent_name, info in infos.items():
            if not isinstance(info, dict):
                continue
            info["_hidden_reset"] = True
            reset_info = pending_reset_infos.get(agent_name)
            if reset_info is not None and "reset_info" not in info:
                info["reset_info"] = reset_info

    def _recover_after_error(self, phase: str, exc: BaseException) -> None:
        self._last_recovery_phase = phase
        if self.verbose:
            print(
                f"hidden recovery start slot={self._slot_index} phase={phase} "
                f"base_port={self.port} current_port={self._current_agent_worker_port} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
        self._restart_agent_worker_process()
        self._hidden_reset_after_recovery()
        if self.verbose:
            print(
                f"hidden recovery success slot={self._slot_index} phase={phase} "
                f"base_port={self.port} current_port={self._current_agent_worker_port}",
                flush=True,
            )

    def _format_backend_failure(self, phase: str, exc: BaseException) -> RuntimeError:
        return RuntimeError(
            f"COMRAD backend {phase} failed "
            f"(slot={self._slot_index}, base_port={self.port}, current_port={self._current_agent_worker_port}, "
            f"init_attempts={self._last_init_attempts}): {type(exc).__name__}: {exc}"
        )

    def reset(self, seed=None, options=None):
        if seed is not None and (
            self._ext_seed is None or int(seed) != int(self._ext_seed)
        ):
            self._ext_seed = int(seed)
            self._restart_agent_worker_process()

        def _do_reset():
            assert self._parent_conn is not None
            self._parent_conn.send({"cmd": "reset"})
            message = self._recv_message(timeout=_RESET_TIMEOUT)
            obs = message["observations"]
            infos = message["infos"]
            self._last_frames = dict(obs)
            self.agents = self.possible_agents[:]
            self._pending_reset_infos = None
            self._pending_hidden_reset = False
            return obs, infos

        try:
            return _do_reset()
        except Exception as exc:
            first_error = exc
            if _HIDDEN_RECOVERY_ATTEMPTS < 1:
                raise self._format_backend_failure(
                    "reset", first_error
                ) from first_error
            try:
                self._recover_after_error("reset", first_error)
                return _do_reset()
            except Exception as retry_exc:
                raise self._format_backend_failure("reset", retry_exc) from first_error

    def step(self, actions):
        flat_actions: list[list[float]] = []
        for agent in self.agents:
            agent_action = actions.get(agent, self._noop_action())
            encoded = self._encode_env_action(agent_action)
            if len(encoded) != self._act_len:
                raise ValueError(
                    f"Encoded action length {len(encoded)} != expected {self._act_len}"
                )
            flat_actions.append(encoded)

        def _do_step():
            assert self._parent_conn is not None
            self._parent_conn.send({"cmd": "step", "actions": flat_actions})
            message = self._recv_message(timeout=self._barrier_timeout)
            observations = message["observations"]
            rewards = message["rewards"]
            terminations = message["terminations"]
            truncations = message["truncations"]
            infos = message["infos"]
            self._last_frames = dict(observations)
            self._attach_pending_reset_infos(infos)
            return observations, rewards, terminations, truncations, infos

        try:
            return _do_step()
        except Exception as exc:
            first_error = exc
            if _HIDDEN_RECOVERY_ATTEMPTS < 1:
                raise self._format_backend_failure("step", first_error) from first_error
            try:
                self._recover_after_error("step", first_error)
                return _do_step()
            except Exception as retry_exc:
                raise self._format_backend_failure("step", retry_exc) from first_error

    def close(self):
        self._shutdown_agent_worker_process()

    def debug_status(self) -> dict[str, int | str]:
        return {
            "slot_index": self._slot_index,
            "base_port": int(self.port),
            "current_port": int(self._current_agent_worker_port),
            "seed": -1 if self._ext_seed is None else int(self._ext_seed),
            "init_attempts": int(self._last_init_attempts),
            "hidden_reset_pending": bool(self._pending_hidden_reset),
            "last_recovery_phase": self._last_recovery_phase,
        }
