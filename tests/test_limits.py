import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gemini_limits as gl  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


MINUTE_429 = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota. quotaId: "
              "GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 53.2s. "
              "{'retryDelay': '53s'}")
DAY_429 = "429 RESOURCE_EXHAUSTED. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def test_classify():
    assert gl.classify(ApiError(429, MINUTE_429)) == ("rate", 53.2)
    assert gl.classify(ApiError(429, DAY_429))[0] == "daily"
    assert gl.classify(ApiError(503, "The model is overloaded. Please try again later."))[0] == "busy"
    assert gl.classify(Exception("RESOURCE_EXHAUSTED"))[0] == "rate"
    assert gl.classify(ApiError(400, "invalid argument"))[0] == "other"
    assert gl.classify(ValueError("boom"))[0] == "other"


def test_throttle_spaces_requests():
    c = FakeClock()
    th = gl.Throttle(clock=c.now, sleep=c.sleep)
    waits = [th.wait("k", 6) for _ in range(4)]          # 6 per minute -> 10 s apart
    assert waits == [0.0, 10.0, 10.0, 10.0]
    assert th.wait("other-key", 6) == 0.0                # keys are independent
    assert th.wait("k", 0) == 0.0                        # 0 = unlimited


def test_retry_waits_for_the_suggested_delay_then_succeeds():
    c = FakeClock()
    th = gl.Throttle(clock=c.now, sleep=c.sleep)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ApiError(429, MINUTE_429)
        return "ok"

    assert gl.call_with_retry(fn, key="k", rpm=0, sleep=c.sleep, throttle=th) == "ok"
    assert len(calls) == 3 and len(c.slept) >= 2
    assert all(53.2 <= w <= 120 for w in c.slept[:2])     # honours "retry in 53.2s"


def test_retry_gives_up_with_a_clear_error():
    c = FakeClock()
    th = gl.Throttle(clock=c.now, sleep=c.sleep)
    n = []

    def fn():
        n.append(1)
        raise ApiError(503, "overloaded")

    with pytest.raises(gl.RateLimited) as e:
        gl.call_with_retry(fn, key="k", rpm=5, max_attempts=3, sleep=c.sleep, throttle=th)
    assert len(n) == 3 and e.value.attempts == 3
    assert "Max requests per minute" in e.value.user_message() and "5 requests per minute" in e.value.user_message()


def test_daily_quota_and_other_errors_are_not_retried():
    c = FakeClock()
    th = gl.Throttle(clock=c.now, sleep=c.sleep)
    n = []

    def daily():
        n.append(1)
        raise ApiError(429, DAY_429)

    with pytest.raises(gl.QuotaExhausted) as e:
        gl.call_with_retry(daily, sleep=c.sleep, throttle=th)
    assert len(n) == 1 and "tomorrow" in e.value.user_message()
    with pytest.raises(ApiError):
        gl.call_with_retry(lambda: (_ for _ in ()).throw(ApiError(400, "bad")), sleep=c.sleep, throttle=th)
    assert c.slept == []


def test_far_away_retry_delay_is_not_waited_for():
    c = FakeClock()
    th = gl.Throttle(clock=c.now, sleep=c.sleep)
    with pytest.raises(gl.RateLimited) as e:
        gl.call_with_retry(lambda: (_ for _ in ()).throw(ApiError(429, "RESOURCE_EXHAUSTED retry in 900s")),
                           max_wait=120, sleep=c.sleep, throttle=th)
    assert e.value.retry_after == 900 and c.slept == []
