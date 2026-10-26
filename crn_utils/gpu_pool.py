"""Run a worker function over a work list, spread across the machine's GPUs.

Replaces FireWorks/qlaunch: no job queue, just a persistent pool of one worker
process per GPU pulling from a plain Python work list.
"""

import functools
import multiprocessing as mp
import os

_gpu_queue = None


def _init_worker(gpu_queue, env):
    global _gpu_queue
    _gpu_queue = gpu_queue
    gpu_id = _gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.update(env)


def _run_with_error_capture(worker_fn, item):
    try:
        return worker_fn(item)
    except Exception as exc:  # noqa: BLE001 - surfaced per-item, not fatal to the pool
        return {"error": f"{type(exc).__name__}: {exc}", "item": repr(item)}


def map_over_gpus(worker_fn, work_items, n_gpus=8, env=None):
    """Yield worker_fn(item) for each item in work_items, using n_gpus persistent
    worker processes (one GPU each, via CUDA_VISIBLE_DEVICES). `spawn` is required
    (not `fork`) because CUDA contexts are not fork-safe. `env`: extra environment
    variables set in each worker before it imports anything CUDA-related.
    """
    env = dict(env or {})
    if not work_items:
        return

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in range(n_gpus):
        gpu_queue.put(gpu_id)

    bound_worker = functools.partial(_run_with_error_capture, worker_fn)
    with ctx.Pool(processes=n_gpus, initializer=_init_worker,
                  initargs=(gpu_queue, env)) as pool:
        for result in pool.imap_unordered(bound_worker, work_items):
            yield result


def worker_env(config):
    """Environment for the workers from config.yaml's `gpu` block:
    `deterministic: true` -> GPU4PYSCF_DETERMINISTIC=1 (bit-reproducible runs;
    needs the group's gpu4pyscf fork, see README)."""
    env = {}
    if config.get("gpu", {}).get("deterministic"):
        env["GPU4PYSCF_DETERMINISTIC"] = "1"
    return env
