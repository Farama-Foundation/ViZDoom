from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

import vizdoom as vzd

from .base_env_common import encode_env_action
from .bot_eval_types import (
    DEFAULT_BOT_PROFILES,
    BotEvalConfig,
    EpisodeResult,
    PolicyAdapter,
    classify_outcome,
)


@dataclass
class EpisodeRun:
    result: EpisodeResult
    frames: list[np.ndarray] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_profile_path() -> Path:
    packaged = Path(vzd.__file__).with_name("deathmatch_eval_bots.cfg")
    if packaged.is_file():
        return packaged.resolve()
    source = _repo_root() / "src" / "deathmatch_eval_bots.cfg"
    return source.resolve()


def resolve_scenario_config(config: BotEvalConfig) -> Path:
    if config.scenario_config:
        return Path(config.scenario_config).expanduser().resolve()
    if not config.scenario:
        raise ValueError(
            "BotEvalConfig.scenario is unset: pass the training scenario name "
            "(pettingzoo_learning does this via build_bot_eval_config) or "
            "scenario_config"
        )
    packaged = Path(vzd.__file__).with_name("scenarios") / f"{config.scenario}.cfg"
    if packaged.is_file():
        return packaged.resolve()
    return (_repo_root() / "scenarios" / f"{config.scenario}.cfg").resolve()


def _parse_scenario_config(path: Path) -> dict[str, str]:
    """Read `key = value` pairs of scenario cfg
    Block values like `available_buttons = { ... }` are skipped as its parsed by viz_game already
    """
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower().replace(" ", "")
        value = value.strip()
        if value.startswith("{"):
            continue
        entries[key] = value
    return entries


def _normalized_key(entries: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        candidate = name.lower().replace("_", "")
        for key, value in entries.items():
            if key.replace("_", "") == candidate:
                return value
    return None


def resolve_scenario_wad(config: BotEvalConfig, config_path: Path) -> Path | None:
    entries = _parse_scenario_config(config_path)
    wad = _normalized_key(entries, "doom_scenario_path")
    if not wad:
        return None
    candidate = Path(wad).expanduser()
    if candidate.is_absolute():
        return candidate
    for root in (config_path.parent, _repo_root() / "scenarios", _repo_root()):
        resolved = root / candidate
        if resolved.is_file():
            return resolved.resolve()
    return (config_path.parent / candidate).resolve()


def scenario_episode_timeout(config: BotEvalConfig, config_path: Path) -> int | None:
    if config.episode_timeout is not None:
        return int(config.episode_timeout)
    declared = _normalized_key(_parse_scenario_config(config_path), "episode_timeout")
    if declared is None:
        return None
    try:
        return int(float(declared))
    except ValueError:
        return None


def validate_scenario(config: BotEvalConfig) -> Path:
    config_path = resolve_scenario_config(config)
    if not config_path.is_file():
        raise ValueError(
            f"bot evaluation scenario config not found: {config_path}. "
            "Pass scenario_config to point at an explicit .cfg."
        )

    wad_path = resolve_scenario_wad(config, config_path)
    if wad_path is not None and not wad_path.is_file():
        raise ValueError(
            f"scenario '{config.scenario}' references a missing WAD: {wad_path}"
        )

    if config.num_bots < 1:
        raise ValueError("num_bots must be at least 1")

    if config.require_deathmatch:
        entries = _parse_scenario_config(config_path)
        game_args = (_normalized_key(entries, "game_args") or "").lower()
        if "-deathmatch" not in game_args and "-altdeath" not in game_args:
            raise ValueError(
                f"scenario '{config.scenario}' does not declare a deathmatch in "
                "game_args, so frag-difference metrics are meaningless. Add "
                "'-deathmatch' to the scenario cfg, or set "
                "BotEvalConfig(require_deathmatch=False) to run anyway."
            )
    return config_path


def _screen_frame(state) -> np.ndarray:
    if state is None or state.screen_buffer is None:
        raise RuntimeError("missing_screen_state")
    frame = np.asarray(state.screen_buffer)
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise RuntimeError(f"invalid_screen_shape:{tuple(frame.shape)}")
    return np.asarray(np.transpose(frame, (1, 2, 0)), dtype=np.uint8)


def _variable(game, name: str) -> float:
    return float(game.get_game_variable(getattr(vzd.GameVariable, name)))


class BotDuelEvaluator:
    """Evaluate one learned player against one or more configured bots."""

    def __init__(
        self,
        config: BotEvalConfig,
        game_factory: Callable[[], object] | None = None,
        profiles: Mapping[str, str] | None = None,
    ) -> None:
        self.scenario_config_path = validate_scenario(config)
        self.config = config
        self.game_factory = game_factory or vzd.DoomGame
        self.profiles = dict(profiles or DEFAULT_BOT_PROFILES)
        self.episode_timeout = scenario_episode_timeout(
            config, self.scenario_config_path
        )

    def _configure_game(self, game, seed: int) -> None:
        game.load_config(str(self.scenario_config_path))
        game.set_window_visible(False)
        game.set_screen_resolution(
            getattr(vzd.ScreenResolution, f"RES_{self.config.resolution}")
        )
        game.set_ticrate(int(self.config.ticrate))
        game.set_mode(vzd.Mode.PLAYER)
        if self.episode_timeout is not None:
            game.set_episode_timeout(int(self.episode_timeout))
        game.set_seed(int(seed))
        game.add_game_args(
            "-host 1 -deathmatch +timelimit 0 +sv_spawnfarthest 1 "
            "+sv_noautoaim 1 +sv_nocrouch 1 +sv_nofreelook 1 "
            f"+sv_forcerespawn 1 +viz_respawn_delay {int(self.config.respawn_delay)}"
        )
        game.add_game_args(f"+viz_bots_path {resolve_profile_path()}")
        game.add_game_args(f"+name {self.config.learner_name} +colorset 0")

    def _split_frags(self, game) -> tuple[int, int]:
        """Return (learner frags, best opposing frags) from the server state."""
        server_state = game.get_server_state()
        frags = tuple(int(value) for value in server_state.players_frags)
        names = tuple(
            str(value) for value in getattr(server_state, "players_names", ())
        )
        in_game = tuple(
            bool(value) for value in getattr(server_state, "players_in_game", ())
        )

        active = [
            index
            for index in range(len(frags))
            if not in_game or (index < len(in_game) and in_game[index])
        ]
        if len(active) < 1 + self.config.num_bots:
            raise RuntimeError(
                f"missing_terminal_player_frags:{len(active)}"
                f"_expected_{1 + self.config.num_bots}"
            )

        learner_index = next(
            (
                index
                for index in active
                if index < len(names) and names[index] == self.config.learner_name
            ),
            None,
        )
        if learner_index is None:
            raise RuntimeError(
                f"learner_not_found_in_server_state:{self.config.learner_name!r}"
                f"_names_{names!r}"
            )
        opponents = [frags[index] for index in active if index != learner_index]
        if not opponents:
            raise RuntimeError("missing_terminal_player_frags")
        return frags[learner_index], max(opponents)

    def run_episode(
        self,
        seed: int,
        tier: str,
        policy: PolicyAdapter,
        capture_video: bool = False,
        deterministic: bool = True,
    ) -> EpisodeRun:
        normalized_tier = str(tier).lower()
        profile = self.profiles.get(normalized_tier)
        if profile is None:
            raise ValueError(f"unknown bot tier: {tier}")
        started = time.monotonic()
        game = self.game_factory()
        frames: list[np.ndarray] | None = [] if capture_video else None
        steps = 0
        try:
            self._configure_game(game, seed)
            game.init()
            for _ in range(int(self.config.num_bots)):
                game.send_game_command(f"addbot {profile}")
            policy.reset(int(seed))

            while not game.is_episode_finished():
                state = game.get_state()
                frame = _screen_frame(state)
                if frames is not None:
                    frames.append(frame.copy())
                action = policy.act(frame, deterministic=deterministic)
                if game.is_player_dead():
                    game.respawn_player()
                else:
                    game.make_action(
                        encode_env_action(action, game.get_available_buttons()),
                        int(self.config.skip_frames),
                    )
                steps += 1

            learner_frags, bot_frags = self._split_frags(game)
            episode_tics = int(game.get_episode_time())
            result = EpisodeResult(
                seed=int(seed),
                tier=normalized_tier,
                bot_profile=profile,
                valid=True,
                learner_frags=learner_frags,
                bot_frags=bot_frags,
                learner_deaths=int(_variable(game, "DEATHCOUNT")),
                learner_damage_made=float(_variable(game, "DAMAGECOUNT")),
                learner_damage_taken=float(_variable(game, "DAMAGE_TAKEN")),
                duration_seconds=time.monotonic() - started,
                engine_tics=episode_tics,
                policy_steps=steps,
                timeout=(
                    self.episode_timeout is not None
                    and episode_tics >= int(self.episode_timeout)
                ),
                outcome=classify_outcome(learner_frags, bot_frags),
            )
            return EpisodeRun(result=result, frames=frames)
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            result = EpisodeResult(
                seed=int(seed),
                tier=normalized_tier,
                bot_profile=profile,
                valid=False,
                duration_seconds=time.monotonic() - started,
                policy_steps=steps,
                invalid_reason=reason[:200],
            )
            return EpisodeRun(result=result, frames=frames)
        finally:
            try:
                game.close()
            except Exception:
                pass
