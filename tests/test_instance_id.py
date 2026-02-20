#!/usr/bin/env python3

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import vizdoom as vzd


def _init_and_get_instance_id():
    game = vzd.DoomGame()
    game.set_window_visible(False)
    game.init()
    try:
        return game.get_instance_id()
    finally:
        game.close()


def _process_init_and_get_instance_id(queue):
    try:
        instance_id = _init_and_get_instance_id()
        queue.put(("ok", instance_id))
    except BaseException as exc:
        queue.put(("error", repr(exc)))


def _thread_init_and_get_instance_id(_):
    return _init_and_get_instance_id()


def test_instance_id_unique_same_thread():
    games = []
    instance_ids = []

    try:
        for _ in range(6):
            game = vzd.DoomGame()
            game.set_window_visible(False)
            game.init()
            games.append(game)
            instance_ids.append(game.get_instance_id())
    finally:
        for game in games:
            game.close()

    assert len(instance_ids) == len(set(instance_ids))


def test_instance_id_unique_multiple_threads():
    with ThreadPoolExecutor(max_workers=6) as executor:
        instance_ids = list(executor.map(_thread_init_and_get_instance_id, range(6)))

    assert len(instance_ids) == len(set(instance_ids))


def test_instance_id_unique_multiple_processes():
    ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(ctx_name)
    queue = ctx.Queue()

    processes = [
        ctx.Process(target=_process_init_and_get_instance_id, args=(queue,))
        for _ in range(4)
    ]

    for process in processes:
        process.start()

    results = [queue.get(timeout=120) for _ in processes]

    for process in processes:
        process.join(timeout=120)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    for process in processes:
        assert process.exitcode == 0

    errors = [payload for status, payload in results if status == "error"]
    assert not errors, f"Process errors: {errors}"

    instance_ids = [payload for status, payload in results if status == "ok"]
    assert len(instance_ids) == len(set(instance_ids))
