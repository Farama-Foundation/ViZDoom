import glob
import multiprocessing
import os
import time

import vizdoom as vzd


DEFAULT_PROCESSES = 2


def _ipc_paths():
    return [
        "/private/tmp/boost_interprocess",
        "/tmp/boost_interprocess",
        "/dev/shm/boost_interprocess",
        "/dev/shm",
        "/tmp",
        "/dev/mqueue",
    ]


def now():
    results = set()
    for p in _ipc_paths():
        results.update(glob.glob(os.path.join(p, "ViZDoom*")))
    return results


def get_ipc_id(p):
    ne = os.path.basename(p)
    for pre in ("ViZDoomMQCtr", "ViZDoomMQDoom", "ViZDoomSM"):
        if ne.startswith(pre):
            return ne[len(pre) :]
    return ne


def game(q):
    g = vzd.DoomGame()
    # g.set_console_enabled(True)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.init()
    q.put(os.getpid())
    time.sleep(2)
    g.close()


if __name__ == "__main__":
    q = multiprocessing.Queue()

    bef = now()

    proc = [
        multiprocessing.Process(target=game, args=(q,))
        for _ in range(DEFAULT_PROCESSES)
    ]
    for p in proc:
        p.start()

    time.sleep(1)
    ipc_ids = {get_ipc_id(f) for f in now() - bef}

    r = [q.get(timeout=30) for _ in proc]

    for p in proc:
        p.join()

    pids = set(r)
    print("PIDs:", pids)
    print("IPC IDs:", ipc_ids)
    assert len(pids) == DEFAULT_PROCESSES
    assert len(ipc_ids) == DEFAULT_PROCESSES
