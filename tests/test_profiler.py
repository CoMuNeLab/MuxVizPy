import time
import numpy as np
import pytest
from runners.profiler import Profiler


def test_profiler_measures_elapsed_time():
    with Profiler() as p:
        time.sleep(0.05)
    assert p.result.time_s >= 0.04
    assert p.result.time_s < 0.5


def test_profiler_measures_memory_allocation():
    with Profiler() as p:
        data = np.zeros(10_000_000)  # ~80 MB
    assert p.result.peak_py_bytes > 1_000_000


def test_profiler_rss_is_non_negative():
    with Profiler() as p:
        _ = list(range(10000))
    assert p.result.peak_rss_bytes >= 0


def test_profiler_result_raises_before_exit():
    p = Profiler()
    with pytest.raises(RuntimeError):
        _ = p.result


def test_profiler_can_be_used_multiple_times_sequentially():
    with Profiler() as p1:
        time.sleep(0.01)
    with Profiler() as p2:
        time.sleep(0.02)
    assert p1.result.time_s < p2.result.time_s


def test_hash_values_is_deterministic():
    values = [1.0, 2.0, 3.0]
    h1 = Profiler.hash_values(values)
    h2 = Profiler.hash_values(values)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_hash_values_differs_for_different_arrays():
    assert Profiler.hash_values([1.0, 2.0]) != Profiler.hash_values([1.0, 3.0])
