"""
resource_tracker.py — Lightweight Linux Process Resource Monitor.

Measures peak RSS memory (MB) and average CPU utilisation (%) during an operation
via a non-intrusive background sampling thread.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional


class ResourceTracker:
    """
    Context manager that tracks peak RSS memory and average CPU utilisation
    for a single operation via a background sampling thread.

    Usage::

        with ResourceTracker() as rt:
            do_heavy_work()

        print(rt.peak_ram_mb, rt.cpu_percent, rt.elapsed_sec)

    Attributes (readable after __exit__)
    -------------------------------------
    peak_ram_mb   : float  — highest RSS observed during the operation (MB)
    cpu_percent   : float  — average CPU utilisation across all samples (%)
    elapsed_sec   : float  — wall-clock duration of the operation (s)
    """

    _SAMPLE_INTERVAL = 0.25  # seconds between RSS/CPU samples

    def __init__(self) -> None:
        self.peak_ram_mb: float = 0.0
        self.cpu_percent: float = 0.0
        self.elapsed_sec: float = 0.0
        self._start: float = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_samples: list[float] = []
        self._start_cpu: Optional[float] = None

    # ---- RSS helper (Linux /proc; falls back to resource module) ----
    @staticmethod
    def _rss_mb() -> float:
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        try:
            import resource as _res
            return _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0

    # ---- CPU helper (requires /proc/self/stat) ----
    @staticmethod
    def _read_cpu_jiffies() -> Optional[float]:
        try:
            with open("/proc/self/stat") as fh:
                parts = fh.read().split()
            # utime (index 13) + stime (index 14), in clock ticks
            hz = os.sysconf("SC_CLK_TCK")
            return (int(parts[13]) + int(parts[14])) / hz
        except Exception:
            return None

    def _sample_loop(self) -> None:
        prev_cpu = self._read_cpu_jiffies()
        prev_time = time.monotonic()
        while not self._stop_event.is_set():
            time.sleep(self._SAMPLE_INTERVAL)
            rss = self._rss_mb()
            if rss > self.peak_ram_mb:
                self.peak_ram_mb = rss
            now = time.monotonic()
            cur_cpu = self._read_cpu_jiffies()
            if prev_cpu is not None and cur_cpu is not None:
                dt_wall = now - prev_time or 1e-9
                dt_cpu = cur_cpu - prev_cpu
                self._cpu_samples.append(min((dt_cpu / dt_wall) * 100.0, 800.0))
                prev_cpu = cur_cpu
                prev_time = now

    def __enter__(self) -> "ResourceTracker":
        self.peak_ram_mb = self._rss_mb()   # baseline before work starts
        self._start = time.monotonic()
        self._start_cpu = self._read_cpu_jiffies()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="resource-tracker"
        )
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.elapsed_sec = time.monotonic() - self._start
        # Final RSS sample after work completes
        final_rss = self._rss_mb()
        if final_rss > self.peak_ram_mb:
            self.peak_ram_mb = final_rss
        if self._cpu_samples:
            self.cpu_percent = sum(self._cpu_samples) / len(self._cpu_samples)
        else:
            # Fallback for fast runs where sampling loop hasn't accumulated samples
            end_cpu = self._read_cpu_jiffies()
            if self._start_cpu is not None and end_cpu is not None and self.elapsed_sec > 0:
                dt_cpu = end_cpu - self._start_cpu
                self.cpu_percent = min((dt_cpu / self.elapsed_sec) * 100.0, 800.0)
            else:
                self.cpu_percent = 0.0
        self.peak_ram_mb = round(self.peak_ram_mb, 1)
        self.cpu_percent = round(self.cpu_percent, 1)
        self.elapsed_sec = round(self.elapsed_sec, 3)
