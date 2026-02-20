import glob
import multiprocessing
import os
import time

import vizdoom as vzd


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


def getid(p):
    ne = os.path.basename(p)
    for pre in ("ViZDoomMQCtr", "ViZDoomMQDoom", "ViZDoomSM"):
        if ne.startswith(pre):
            return ne[len(pre) :]
    return ne


def game(q):
    bef = now()
    g = vzd.DoomGame()
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.init()
    q.put((os.getpid(), {getid(f) for f in now() - bef}, None))
    time.sleep(5)
    g.close()


if __name__ == "__main__":
    q = multiprocessing.Queue()
    proc = [multiprocessing.Process(target=game, args=(q,)) for _ in range(10)]
    for p in proc:
        p.start()

    r = [q.get(timeout=30) for _ in proc]
    for p in proc:
        p.join()
    for pid, ids, err in r:
        print(f"PID {pid}: {', '.join(sorted(ids))}, len: {len(ids)}")
