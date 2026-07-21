from parallax.p2p.utils import mdns_enabled_for_topology


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
