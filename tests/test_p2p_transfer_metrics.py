from parallax.p2p.server import _transfer_metrics


def test_transfer_metrics_handles_zero_duration_sample():
    size_mb, elapsed_ms, speed_mb_s = _transfer_metrics(1024, 0)

    assert size_mb > 0
    assert elapsed_ms > 0
    assert speed_mb_s > 0
