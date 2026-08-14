"""Tests for the alert class registry and follow-up strategy lookup."""

import pytest

from chatterbox.alerts import (
    REGISTRY,
    MessengerClass,
    get_alert_class,
    get_strategy,
    sim_nights_for,
)

#: Every label ``forward_alerts.py`` can actually emit.
EMITTED_LABELS = (
    "GW_case_B",
    "GW_case_D",
    "GW_case_large",
    "lensed_BNS_case_A",
    "lensed_BNS_case_B",
    "BBH_case_A",
    "neutrino",
    "SN_Galactic",
)


@pytest.mark.parametrize("label", EMITTED_LABELS)
def test_every_emitted_label_is_registered(label):
    assert label in REGISTRY
    assert REGISTRY[label].produced is True


@pytest.mark.parametrize("label", EMITTED_LABELS)
def test_every_emitted_label_has_a_strategy(label):
    """No strategy would mean no simulation and no plan to report."""
    strategy = get_strategy(label)
    assert strategy.epochs, f"no follow-up strategy for {label}"
    assert strategy.bands
    assert strategy.total_visits > 0


@pytest.mark.parametrize("label", EMITTED_LABELS)
def test_every_emitted_label_has_positive_sim_nights(label):
    assert sim_nights_for(label) > 0


def test_lensed_bns_has_sim_nights():
    """Upstream sim drivers raise ValueError on these labels; we must not."""
    assert sim_nights_for("lensed_BNS_case_A") == 20
    assert sim_nights_for("lensed_BNS_case_B") == 20


@pytest.mark.parametrize(
    "label,messenger",
    [
        ("GW_case_B", MessengerClass.GRAVITATIONAL_WAVE),
        ("BBH_case_A", MessengerClass.GRAVITATIONAL_WAVE),
        ("lensed_BNS_case_A", MessengerClass.GRAVITATIONAL_WAVE),
        ("neutrino", MessengerClass.NEUTRINO),
        ("SN_Galactic", MessengerClass.GALACTIC_SN),
        ("SSO_night", MessengerClass.SOLAR_SYSTEM),
        ("SSO_twilight", MessengerClass.SOLAR_SYSTEM),
    ],
)
def test_messenger_mapping(label, messenger):
    assert get_alert_class(label).messenger is messenger


def test_sso_is_marked_unproduced():
    """No producer emits an asteroid alert; the post must say so."""
    for label in ("SSO_night", "SSO_twilight"):
        alert_class = get_alert_class(label)
        assert alert_class.produced is False
        assert "producer" in alert_class.criteria.lower() or alert_class.notes


def test_unknown_label_is_handled_gracefully():
    alert_class = get_alert_class("Something_New_v2")
    assert alert_class.produced is False
    assert alert_class.sim_nights > 0
    # Falls back to a GW-shaped default rather than raising.
    assert alert_class.messenger is MessengerClass.GRAVITATIONAL_WAVE
    assert get_strategy("Something_New_v2").epochs == []


def test_unknown_label_messenger_is_inferred_from_the_name():
    assert get_alert_class("neutrino_u").messenger is MessengerClass.NEUTRINO
    assert get_alert_class("BBH_case_Z").messenger is MessengerClass.GRAVITATIONAL_WAVE
    assert get_alert_class("SSO_dawn").messenger is MessengerClass.SOLAR_SYSTEM


def test_bbh_gets_the_longest_simulation():
    """BBH follow-up spans 39 days, so it needs more nights than BNS."""
    assert sim_nights_for("BBH_case_A") > sim_nights_for("GW_case_B")


def test_strategy_band_ordering_is_canonical():
    bands = get_strategy("GW_case_B").bands
    assert bands == sorted(bands, key="ugrizy".index)


def test_strategy_visits_per_band():
    """GW_case_B: gri x4 at three early epochs, ri x6 at three later ones."""
    per_band = get_strategy("GW_case_B").visits_per_band()
    assert per_band["g"] == 12
    assert per_band["r"] == 30
    assert per_band["i"] == 30
    assert "u" not in per_band


def test_neutrino_strategy_requests_u_band():
    """u band matters here: it can be dropped if not in the carousel."""
    assert "u" in get_strategy("neutrino").bands


def test_galactic_sn_is_a_single_epoch_tiling():
    strategy = get_strategy("SN_Galactic")
    assert strategy.bands == ["i"]
    assert strategy.span_hours == 0.0
    # Alternating 1 s and 15 s exposures.
    assert {e.exptime for e in strategy.epochs} == {1.0, 15.0}


def test_strategy_falls_back_to_the_snapshot(monkeypatch):
    """With ts_fbs_utils unavailable, the vendored snapshot must still work."""
    import chatterbox.alerts.classes as classes

    classes._live_strategies.cache_clear()
    monkeypatch.setattr(classes, "_live_strategies", lambda: None)
    strategy = get_strategy("GW_case_B")
    assert strategy.from_ts_fbs_utils is False
    assert strategy.visits_per_band() == {"g": 12, "r": 30, "i": 30}


def test_snapshot_agrees_with_ts_fbs_utils():
    """Guard against the vendored snapshot drifting from upstream."""
    import chatterbox.alerts.classes as classes

    live = classes._live_strategies()
    if live is None:
        pytest.skip("ts_fbs_utils not installed")
    for label in EMITTED_LABELS:
        snapshot = classes._SNAPSHOT.get(label)
        assert snapshot is not None, f"no snapshot for {label}"
        live_epochs = live[label]
        assert len(live_epochs) == len(snapshot), f"{label}: epoch count drifted"
        for epoch, (t_hours, bands, nvis, exptime) in zip(live_epochs, snapshot):
            assert epoch.bands == bands, f"{label}: bands drifted"
            assert epoch.nvis == nvis, f"{label}: nvis drifted"
            assert epoch.exptime == pytest.approx(exptime), f"{label}: exptime drifted"
            assert epoch.t_hours == pytest.approx(t_hours, abs=1e-3), f"{label}: epoch time drifted"
