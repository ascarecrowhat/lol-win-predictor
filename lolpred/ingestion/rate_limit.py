"""Sliding-window rate limiting for the Riot API.

Riot enforces two independent layers:
  * app limits  - shared by every endpoint on the key (dev key: 20/1s and 100/120s)
  * method limits - per endpoint, advertised in the X-Method-Rate-Limit header

We enforce the app limits from config and *learn* the method limits from response
headers the first time we see each endpoint, so a production key needs no re-tuning.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowLimiter:
    """Blocks until a request fits inside every configured (count, seconds) window.

    Riot counts against *fixed* windows, not sliding ones, so a client that bursts
    its whole budget and then waits will straddle a bucket boundary and get 429s
    even though its own sliding window is satisfied. We therefore also pace
    requests evenly: at most one per span/count seconds, times a safety factor.
    Steady-state throughput is the same, minus the headroom, but without the
    bursts - and without the multi-minute Retry-After stalls they trigger.
    """

    def __init__(self, limits: list[tuple[int, int]], safety: float = 1.05):
        self._lock = threading.Lock()
        self._windows = [(count, span, deque()) for count, span in limits]
        self._min_interval = (
            max(span / count for count, span in limits) * safety if limits else 0.0
        )
        self._last_issued = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                wait = 0.0
                for count, span, stamps in self._windows:
                    while stamps and now - stamps[0] >= span:
                        stamps.popleft()
                    if len(stamps) >= count:
                        wait = max(wait, span - (now - stamps[0]) + 0.005)
                wait = max(wait, self._last_issued + self._min_interval - now)
                if wait <= 0.0:
                    for _, _, stamps in self._windows:
                        stamps.append(now)
                    self._last_issued = now
                    return
            time.sleep(min(wait, 5.0))

    def penalise(self, seconds: float) -> None:
        """After a 429: block everyone for `seconds` by backdating a full window."""
        deadline = time.monotonic() + seconds
        with self._lock:
            for count, span, stamps in self._windows:
                stamps.clear()
                # Fill the window with timestamps that only expire at the deadline.
                for _ in range(count):
                    stamps.append(deadline - span)


class LimiterRegistry:
    """One app-wide limiter plus a lazily created limiter per API method."""

    def __init__(self, app_limits: list[tuple[int, int]]):
        self.app = SlidingWindowLimiter(app_limits)
        self._methods: dict[str, SlidingWindowLimiter] = {}
        self._lock = threading.Lock()

    def acquire(self, method: str) -> None:
        self.app.acquire()
        with self._lock:
            limiter = self._methods.get(method)
        if limiter is not None:
            limiter.acquire()

    def learn_method_limits(self, method: str, header: str | None) -> None:
        if not header:
            return
        with self._lock:
            if method in self._methods:
                return
            limits = []
            for chunk in header.split(","):
                count, _, span = chunk.strip().partition(":")
                try:
                    limits.append((int(count), int(span)))
                except ValueError:
                    return
            if limits:
                self._methods[method] = SlidingWindowLimiter(limits)

    def penalise(self, method: str, seconds: float, scope: str) -> None:
        if scope == "application":
            self.app.penalise(seconds)
        else:
            with self._lock:
                limiter = self._methods.get(method)
            if limiter is not None:
                limiter.penalise(seconds)
            else:
                self.app.penalise(seconds)
