from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import vizdoom as vzd

from .base_env_common import configure_doom_game, encode_env_action
from .bot_eval_duel import resolve_scenario_config, scenario_episode_timeout
from .bot_eval_policy import TorchRLPolicyAdapter, load_bot_eval_experiment
from .bot_eval_types import (
    BotEvalConfig,
    EpisodeResult,
    classify_outcome,
    summarize_tier,
)


POLICY_NAME = "Policy"
HUMAN_NAME = "Human"
HUMAN_BUTTONS = (
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT_RIGHT_DELTA,
    vzd.Button.LOOK_UP_DOWN_DELTA,
    vzd.Button.ATTACK,
)
HUMAN_CONTROL_COMMANDS = (
    "use_mouse 1",
    "mouse_capturemode 2",
    "m_yaw 1",
    "m_pitch 1",
    "freelook 1",
    'bind w "+forward"',
    'bind a "+moveleft"',
    'bind s "+back"',
    'bind d "+moveright"',
)
HUMAN_CONTROL_ARGS = " ".join(f"+{command}" for command in HUMAN_CONTROL_COMMANDS)


def _close_reloaded_experiment(experiment: Any) -> None:
    for environment_name in ("rollout_env", "test_env"):
        environment = getattr(experiment, environment_name, None)
        if environment is None:
            continue
        try:
            environment.close()
        except RuntimeError as exc:
            if "closed environment" not in str(exc):
                raise
    logger = getattr(experiment, "logger", None)
    if logger is not None:
        logger.finish()


def _screen_frame(game: vzd.DoomGame) -> np.ndarray:
    state = game.get_state()
    if state is None or state.screen_buffer is None:
        raise RuntimeError("Policy player has no screen state")
    return np.asarray(np.transpose(state.screen_buffer, (1, 2, 0)), dtype=np.uint8)


def _variable(game: vzd.DoomGame, name: str) -> float:
    return float(game.get_game_variable(getattr(vzd.GameVariable, name)))


def player_frags(server_state: Any) -> dict[str, int]:
    names = tuple(str(name) for name in server_state.players_names)
    frags = tuple(int(frag) for frag in server_state.players_frags)
    in_game = tuple(bool(active) for active in server_state.players_in_game)
    return {
        name: frags[index]
        for index, name in enumerate(names)
        if index < len(frags) and (index >= len(in_game) or in_game[index])
    }


def _policy_worker(
    connection,
    checkpoint: str,
    episodes: int,
    scenario_config: str,
    resolution: str,
    frame_skip: int,
    port: int,
    seed: int,
) -> None:
    try:
        experiment = load_bot_eval_experiment(checkpoint)
        group_name = next(iter(experiment.group_map))
        policy = TorchRLPolicyAdapter.from_experiment(
            experiment, group_name=group_name, agent_index=0, device="cpu"
        )
        _close_reloaded_experiment(experiment)
        connection.send(("loaded", None))

        for episode in range(episodes):
            episode_seed = seed + episode
            game = configure_doom_game(
                config_path=scenario_config,
                resolution=resolution,
                ticrate=vzd.DEFAULT_TICRATE,
                async_mode=False,
                timeout=None,
                seed=episode_seed,
                is_host=True,
                num_agents=2,
                host_address="127.0.0.1",
                port=port,
                netmode=0,
                agent_idx=0,
            )
            game.add_game_args(f"+name {POLICY_NAME} +colorset 0")
            try:
                connection.send(("ready", episode))
                game.init()
                policy.reset(episode_seed)
                action = [0.0] * game.get_available_buttons_size()
                policy_steps = 0
                engine_tics = 0

                while not game.is_episode_finished():
                    if game.is_player_dead():
                        game.respawn_player()
                    if engine_tics % frame_skip == 0:
                        action = encode_env_action(
                            policy.act(_screen_frame(game), deterministic=True),
                            game.get_available_buttons(),
                        )
                        policy_steps += 1
                    game.set_action(action)
                    game.advance_action(1)
                    engine_tics += 1

                connection.send(
                    (
                        "finished",
                        {
                            "episode": episode,
                            "policy_steps": policy_steps,
                            "engine_tics": int(game.get_episode_time()),
                            "frags": player_frags(game.get_server_state()),
                        },
                    )
                )
                message, payload = connection.recv()
                if message != "collected" or payload != episode:
                    raise RuntimeError(
                        f"Expected metrics acknowledgement for episode {episode}"
                    )
            finally:
                game.close()
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _receive(connection, expected: str, timeout: float = 120.0) -> Any:
    if not connection.poll(timeout):
        raise TimeoutError(f"Timed out waiting for policy process to report {expected}")
    message, payload = connection.recv()
    if message == "error":
        raise RuntimeError(f"Policy process failed: {payload}")
    if message != expected:
        raise RuntimeError(f"Expected policy message {expected!r}, got {message!r}")
    return payload


def _configure_human_controls(game: vzd.DoomGame) -> None:
    game.set_available_buttons(list(HUMAN_BUTTONS))
    game.add_game_args(HUMAN_CONTROL_ARGS)


def _respawn_human(game: vzd.DoomGame) -> None:
    game.respawn_player()
    for command in HUMAN_CONTROL_COMMANDS:
        game.send_game_command(command)


def _human_game(scenario_config: str, port: int) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(scenario_config)
    _configure_human_controls(game)
    game.set_window_visible(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.add_game_args(
        f"-join 127.0.0.1:{port} -netmode 0 +viz_connect_timeout 60 "
        f"+name {HUMAN_NAME} +colorset 3"
    )
    return game


def _episode_result(
    game: vzd.DoomGame,
    *,
    seed: int,
    elapsed: float,
    policy_status: dict[str, Any],
    episode_timeout: int | None,
) -> EpisodeResult:
    frags = policy_status["frags"]
    if HUMAN_NAME not in frags or POLICY_NAME not in frags:
        raise RuntimeError(f"Missing duel players in server state: {sorted(frags)}")
    human_frags = frags[HUMAN_NAME]
    policy_frags = frags[POLICY_NAME]
    engine_tics = int(policy_status["engine_tics"])
    return EpisodeResult(
        seed=seed,
        tier="policy",
        valid=True,
        learner_frags=human_frags,
        bot_frags=policy_frags,
        learner_deaths=int(_variable(game, "DEATHCOUNT")),
        learner_damage_made=_variable(game, "DAMAGECOUNT"),
        learner_damage_taken=_variable(game, "DAMAGE_TAKEN"),
        duration_seconds=elapsed,
        engine_tics=engine_tics,
        policy_steps=int(policy_status["policy_steps"]),
        timeout=episode_timeout is not None and engine_tics >= episode_timeout,
        outcome=classify_outcome(human_frags, policy_frags),
        bot_profile="checkpoint_policy",
    )


def summary_payload(
    results: Sequence[EpisodeResult], bootstrap_seed: int
) -> dict[str, Any]:
    summary = summarize_tier(
        "policy", results, bootstrap_seed=bootstrap_seed, bootstrap_samples=10_000
    ).to_dict()
    summary["human_frags_mean"] = summary.pop("learner_frags_mean")
    summary["policy_frags_mean"] = summary.pop("bot_frags_mean")
    summary["human_damage_made_mean"] = summary.pop("learner_damage_made_mean")
    summary["human_damage_taken_mean"] = summary.pop("learner_damage_taken_mean")
    summary["human_deaths_mean"] = summary.pop("learner_deaths_mean")
    return summary


def play_against_policy(
    checkpoint: str | Path,
    *,
    episodes: int,
    port: int = 5029,
    seed: int = 42,
    output: str | Path | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = BotEvalConfig(scenario="multi_duel")
    scenario_config = resolve_scenario_config(config)
    episode_timeout = scenario_episode_timeout(config, scenario_config)
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_policy_worker,
        args=(
            child_connection,
            str(checkpoint),
            episodes,
            str(scenario_config),
            "160X120",
            4,
            port,
            seed,
        ),
        daemon=False,
    )
    process.start()
    child_connection.close()
    results: list[EpisodeResult] = []

    try:
        _receive(parent_connection, "loaded")
        print(
            "Controls: W/A/S/D to move, mouse/trackpad to look, "
            "left click or Ctrl to fire, Esc to release input."
        )
        for episode in range(episodes):
            ready_episode = _receive(parent_connection, "ready")
            if ready_episode != episode:
                raise RuntimeError(
                    f"Policy prepared unexpected episode {ready_episode}"
                )
            game = _human_game(str(scenario_config), port)
            try:
                print(f"Episode {episode + 1}/{episodes}: connecting...")
                game.init()
                started = time.monotonic()
                deadline = started
                while not game.is_episode_finished():
                    if game.is_player_dead():
                        _respawn_human(game)
                    game.advance_action(1)
                    deadline += 1.0 / vzd.DEFAULT_TICRATE
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)

                policy_status = _receive(parent_connection, "finished")
                result = _episode_result(
                    game,
                    seed=seed + episode,
                    elapsed=time.monotonic() - started,
                    policy_status=policy_status,
                    episode_timeout=episode_timeout,
                )
                results.append(result)
                print(json.dumps(result.to_dict(), indent=2))
                parent_connection.send(("collected", episode))
            finally:
                game.close()
    finally:
        parent_connection.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()

    payload = {
        "checkpoint": str(checkpoint),
        "episodes": [result.to_dict() for result in results],
        "summary": summary_payload(results, bootstrap_seed=seed + 10_000),
    }
    print("Summary:")
    print(json.dumps(payload["summary"], indent=2))
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {output_path}")
    return payload
