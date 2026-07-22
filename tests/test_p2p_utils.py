from unittest.mock import Mock

from parallax.p2p.utils import log_nat_traversal_preflight, mdns_enabled_for_topology


def test_mdns_defaults_to_lan_discovery_without_public_topology(monkeypatch):
    monkeypatch.delenv("PARALLAX_ENABLE_MDNS", raising=False)

    assert mdns_enabled_for_topology() is True


def test_mdns_coexists_with_bootstrap_and_relay_discovery(monkeypatch):
    monkeypatch.delenv("PARALLAX_ENABLE_MDNS", raising=False)

    assert mdns_enabled_for_topology(initial_peers=["bootstrap"]) is True
    assert mdns_enabled_for_topology(relay_servers=["relay"]) is True


def test_mdns_operator_override_wins_for_mixed_topologies(monkeypatch):
    monkeypatch.setenv("PARALLAX_ENABLE_MDNS", "true")
    assert mdns_enabled_for_topology(relay_servers=["relay"]) is True

    monkeypatch.setenv("PARALLAX_ENABLE_MDNS", "off")
    assert mdns_enabled_for_topology() is False


def test_symmetric_udp_nat_is_advisory_and_does_not_exit():
    lattica = Mock()
    lattica.is_symmetric_nat.return_value = True
    logger = Mock()

    result = log_nat_traversal_preflight(lattica, logger)

    assert result is True
    logger.warning.assert_called_once()
    assert "real direct RPC probe" in logger.warning.call_args.args[0]


def test_nat_preflight_failure_is_non_fatal():
    lattica = Mock()
    lattica.is_symmetric_nat.side_effect = RuntimeError("STUN unavailable")
    logger = Mock()

    result = log_nat_traversal_preflight(lattica, logger)

    assert result is None
    logger.exception.assert_called_once()
