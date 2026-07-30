import threading

import pytest

from swarm_protocol.request_status import RequestPhaseFeed


def test_request_phase_feed_is_bounded_deduplicated_and_snapshot_safe():
    feed = RequestPhaseFeed(max_events=3)

    planning = feed.publish("request-a", "planning")
    assert feed.publish("request-a", "planning") is planning
    feed.publish("request-a", "reserving", epoch=1, route_id="route-a")
    feed.publish("request-b", "prefilling", epoch=2, route_id="route-b")
    completed = feed.publish("request-a", "completed", epoch=1, route_id="route-a")

    snapshot = feed.snapshot()
    assert snapshot["last_event_id"] == completed.event_id
    assert [item["request_id"] for item in snapshot["active_requests"]] == ["request-b"]
    events, gap = feed.read_after(0)
    assert gap is True
    assert events == ()
    events, gap = feed.read_after(planning.event_id)
    assert gap is False
    assert [event.phase for event in events] == ["reserving", "prefilling", "completed"]


def test_request_phase_wait_is_released_by_a_real_transition():
    feed = RequestPhaseFeed()
    result = []

    waiter = threading.Thread(
        target=lambda: result.append(feed.wait_after(0, timeout=5)),
        daemon=True,
    )
    waiter.start()
    event = feed.publish("request-a", "recovering", epoch=2)
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert result == [((event,), False)]


def test_request_phase_feed_rejects_invalid_or_post_close_events():
    feed = RequestPhaseFeed()
    with pytest.raises(ValueError, match="unsupported"):
        feed.publish("request-a", "guessing")
    with pytest.raises(ValueError, match="event id"):
        feed.read_after(-1)
    feed.close()
    with pytest.raises(RuntimeError, match="closed"):
        feed.publish("request-a", "planning")
