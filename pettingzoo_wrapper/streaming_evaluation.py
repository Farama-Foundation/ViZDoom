"""Overwrite BenchMARL evaluation with metric only trajectories, as image will cause OOM for many agents."""

import time
import warnings

import torch
from benchmarl.utils import local_seed, seed_everything
from torchrl.envs.utils import ExplorationType, set_exploration_type


def metric_transition(transition, groups):
    """Copy logging fields before the environment can reuse their storage."""
    fields = ("reward", "episode_reward", "done", "terminated", "truncated", "info")
    keys = []
    for prefix in ((), ("next",)):
        keys.extend((*prefix, field) for field in fields)
        for group in groups:
            keys.extend((*prefix, group, field) for field in fields)
    return transition.select(*keys, strict=False).detach().to("cpu").clone()


def metric_rollout(env, policy, max_steps, groups, callback=None):
    """Stream steps; retain only rewards, done flags and numeric info on CPU."""
    policy_device = next(policy.parameters(), torch.empty(0)).device
    td = env.reset()
    if callback is not None:
        callback(env, td)
    steps = []
    for index in range(max_steps):
        td = td.to(policy_device)
        td.update(policy(td))
        if env.device is not None:
            td = td.to(env.device)
        else:
            td.clear_device_()
        transition = env.step(td)
        td = env.step_mdp(transition)
        steps.append(metric_transition(transition, groups))
        done = any(
            transition.get(
                ("next", *key) if isinstance(key, tuple) else ("next", key)
            ).any()
            for key in env.done_keys
        )
        del transition
        if callback is not None:
            callback(env, td)
            if done:
                callback = None
        if index == max_steps - 1 or (not env.batch_size and done):
            break
        if env.batch_size:
            td = env.maybe_reset(td)
    return torch.stack(steps, dim=len(env.batch_size))


@local_seed()
@torch.no_grad()
def streaming_evaluation(experiment):
    """Keep BenchMARL logging/seeding without full image rollout list."""
    config = experiment.config
    env = experiment.test_env
    if config.evaluation_static:
        seed_everything(experiment.seed)
        try:
            env.set_seed(experiment.seed)
        except NotImplementedError:
            warnings.warn("Static evaluation is not guaranteed: env cannot set seeds.")
    started = time.perf_counter()
    video_frames = None
    callback = None
    if config.render and experiment.task.has_render(env):
        video_frames = []

        def record_frame(env, td):
            frame = experiment.task.__class__.render_callback(experiment, env, td)
            if frame is not None:
                video_frames.append(frame.copy())

        callback = record_frame

    with set_exploration_type(
        ExplorationType.DETERMINISTIC
        if config.evaluation_deterministic_actions
        else ExplorationType.RANDOM
    ):
        if env.batch_size:
            rollout = metric_rollout(
                env,
                experiment.policy,
                experiment.max_steps,
                experiment.group_map,
                callback,
            )
            rollouts = list(rollout.unbind(0))
        else:
            rollouts = [
                metric_rollout(
                    env,
                    experiment.policy,
                    experiment.max_steps,
                    experiment.group_map,
                    callback if episode == 0 else None,
                )
                for episode in range(config.evaluation_episodes)
            ]
    experiment.logger.log(
        {"timers/evaluation_time": time.perf_counter() - started},
        step=experiment.n_iters_performed,
    )
    experiment.logger.log_evaluation(
        rollouts,
        video_frames=video_frames or None,
        step=experiment.n_iters_performed,
        total_frames=experiment.total_frames,
    )
    experiment._on_evaluation_end(rollouts)
