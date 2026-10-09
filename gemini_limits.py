"""Rate limiting and retries for Gemini requests.

* ``THROTTLE`` spaces requests so that at most *rpm* requests per minute are
  started (per API key / project).
* ``call_with_retry`` repeats a request after rate-limit (429) and overload
  (500/503) errors, waiting for the delay the API suggests ("retry in 53s")
  or an exponential backoff, and gives up with a clear message.
* A used-up *daily* quota is recognised and not retried.
"""
from __future__ import annotations

import logging
import random
import re
import threading
import time

log = logging.getLogger("scripture-detector")


class GeminiLimitError(RuntimeError):
    """Base class; ``done``/``total`` tell how many chunks were finished."""
    done: int | None = None
    total: int | None = None

    def progress_note(self) -> str:
        if self.done is None or not self.total:
            return ""
        return (f" {self.done} of {self.total} parts of the text were analysed; they are kept, "
                f"so running the analysis again continues with the remaining parts.")


class QuotaExhausted(GeminiLimitError):
    """The daily quota is used up - retrying later the same day will not help."""

    def user_message(self) -> str:
        return ("The daily Gemini quota for this model and key is used up."
                " Try again tomorrow, choose another model in Settings, or use a key/project with a higher quota."
                + self.progress_note())


class RateLimited(GeminiLimitError):
    """Rate limit (or overload) did not clear after the retries."""

    def __init__(self, message: str, retry_after: float | None = None, attempts: int = 0, rpm: int = 0):
        super().__init__(message)
        self.retry_after = retry_after
        self.attempts = attempts
        self.rpm = rpm

    def user_message(self) -> str:
        wait = f" The API asks to wait about {round(self.retry_after)} seconds." if self.retry_after else ""
        rpm = (f" Currently at most {self.rpm} requests per minute are sent."
               if self.rpm else "")
        return (f"Gemini's rate limit was hit and did not clear after {self.attempts} attempts.{wait}"
                f" Wait a minute and run the analysis again, or lower 'Max requests per minute' in Settings."
                f"{rpm}" + self.progress_note())


# ── classification ───────────────────────────────────────────────────────────

_DELAY_RES = [
    re.compile(r"retry in\s+([\d.]+)\s*s", re.I),
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)\s*s", re.I),
    re.compile(r"retry[- ]after['\"]?\s*[:=]\s*['\"]?([\d.]+)", re.I),
]


def classify(exc: BaseException) -> tuple[str, float | None]:
    """-> (kind, suggested delay in seconds); kind: 'rate', 'daily', 'busy' or 'other'."""
    msg = str(exc)
    low = msg.lower()
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    delay = None
    for r in _DELAY_RES:
        m = r.search(msg)
        if m:
            delay = float(m.group(1))
            break
    compact = low.replace(" ", "").replace("_", "")
    if code == 429 or "resource_exhausted" in low or "ratelimit" in compact or "toomanyrequests" in compact:
        if "perday" in compact or "requestsperday" in compact:
            return "daily", delay
        return "rate", delay
    if code in (500, 502, 503, 504) or "unavailable" in low or "overloaded" in low:
        return "busy", delay
    return "other", delay


# ── throttle ─────────────────────────────────────────────────────────────────

class Throttle:
    """Reserves request slots at least 60/rpm seconds apart, per key."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}

    def wait(self, key: str, rpm: int) -> float:
        if rpm <= 0:
            return 0.0
        interval = 60.0 / rpm
        with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(key, 0.0))
            self._next[key] = slot + interval
        delay = slot - now
        if delay > 0:
            self._sleep(delay)
        return max(delay, 0.0)

    def push_back(self, key: str, seconds: float) -> None:
        """After a rate-limit error: no new request before *seconds* have passed."""
        with self._lock:
            self._next[key] = max(self._next.get(key, 0.0), self._clock() + seconds)


THROTTLE = Throttle()


def call_with_retry(fn, *, key: str = "default", rpm: int = 0, max_attempts: int = 6,
                    max_wait: float = 120.0, sleep=time.sleep, throttle: Throttle | None = None):
    """Call *fn()* respecting the request rate; retry after 429/503 errors."""
    throttle = throttle or THROTTLE
    last_delay = None
    for attempt in range(1, max_attempts + 1):
        throttle.wait(key, rpm)
        try:
            return fn()
        except Exception as exc:                      # noqa: BLE001 - classified below
            kind, delay = classify(exc)
            if kind == "daily":
                raise QuotaExhausted(str(exc)) from exc
            if kind not in ("rate", "busy"):
                raise
            last_delay = delay
            if attempt == max_attempts or (delay is not None and delay > max_wait):
                raise RateLimited(str(exc), retry_after=delay, attempts=attempt, rpm=rpm) from exc
            wait = max(delay or 0.0, min(max_wait, 4.0 * 2 ** (attempt - 1))) + random.uniform(0, 1.5)
            wait = min(wait, max_wait)
            log.warning("Gemini %s error (attempt %d/%d); waiting %.0f s", kind, attempt, max_attempts, wait)
            throttle.push_back(key, wait)
            sleep(wait)
    raise RateLimited("retries exhausted", retry_after=last_delay, attempts=max_attempts, rpm=rpm)
