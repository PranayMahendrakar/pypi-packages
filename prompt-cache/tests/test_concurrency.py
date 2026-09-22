"""Two processes, and several threads, writing to one cache file at once."""

from __future__ import annotations

import subprocess
import sys
import threading

from prompt_cache import Cache

WORKER = """
import sys
import prompt_cache

path, tag, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
cache = prompt_cache.Cache(path)
for index in range(count):
    cache.set("{0}-{1}".format(tag, index), {"tag": tag, "index": index})
for index in range(count):
    assert cache.get("{0}-{1}".format(tag, index)) == {"tag": tag, "index": index}
cache.close()
"""

PER_PROCESS = 60


def test_two_processes_writing_at_once_lose_nothing(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    target = str(tmp_path / "shared")

    processes = [
        subprocess.Popen(
            [sys.executable, str(worker), target, tag, str(PER_PROCESS)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for tag in ("alpha", "beta")
    ]
    results = [process.communicate() for process in processes]

    for process, (_out, err) in zip(processes, results):
        assert process.returncode == 0, err.decode("utf-8", "replace")
        assert err.decode("utf-8", "replace").strip() == ""

    cache = Cache(target)
    assert cache.stats().entries == 2 * PER_PROCESS
    assert cache.get("alpha-0") == {"tag": "alpha", "index": 0}
    assert cache.get("beta-59") == {"tag": "beta", "index": 59}


def test_many_threads_on_one_cache_object(cache_dir):
    cache = Cache(cache_dir)
    errors = []

    def work(worker):
        try:
            for index in range(25):
                prompt = "thread-{0}-{1}".format(worker, index)
                cache.set(prompt, index)
                assert cache.get(prompt) == index
        except BaseException as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert cache.stats().entries == 8 * 25


def test_separate_cache_objects_in_one_process_share_the_file(cache_dir):
    caches = [Cache(cache_dir) for _ in range(4)]
    errors = []

    def work(cache, worker):
        try:
            for index in range(25):
                cache.set("obj-{0}-{1}".format(worker, index), index)
        except BaseException as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(cache, worker)) for worker, cache in enumerate(caches)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert caches[0].stats().entries == 4 * 25
