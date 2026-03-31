"""
Profiler — the sole owner of tracemalloc, resource, and time in this codebase.

Usage:
    with Profiler() as p:
        result = do_work()
    print(p.result.time_s, p.result.peak_rss_bytes)
    hash_ = Profiler.hash_values(result)
"""
import hashlib
import resource
import time
import tracemalloc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ProfileResult:
    time_s: float
    peak_rss_bytes: int
    peak_py_bytes: int


class Profiler:
    def __init__(self) -> None:
        self._result: Optional[ProfileResult] = None
        self._t0: float = 0.0
        self._rss_before: int = 0

    def __enter__(self) -> "Profiler":
        self._rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.perf_counter() - self._t0
        _, peak_py = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self._result = ProfileResult(
            time_s=elapsed,
            peak_rss_bytes=max(rss_after, self._rss_before),
            peak_py_bytes=peak_py,
        )

    @property
    def result(self) -> ProfileResult:
        if self._result is None:
            raise RuntimeError("Profiler has not exited its context yet")
        return self._result

    @staticmethod
    def hash_values(values) -> str:
        """sha256 of float64 byte representation — used to verify replicate consistency."""
        return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()
