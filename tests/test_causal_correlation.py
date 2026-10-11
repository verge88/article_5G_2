from fedsiggraph.data.causal_correlation import CausalCorrelationTracker


def test_later_identity_discovery_does_not_mutate_prior_snapshot():
    tracker = CausalCorrelationTracker()

    first = tracker.observe(ran_ngap_id="ran-7")
    assert first.ue_key is not None
    assert first.session_key is None

    second = tracker.observe(
        ran_ngap_id="ran-7",
        supi="imsi-001010000000001",
        pdu_session_id="10",
    )

    assert second.ue_key == first.ue_key
    assert second.session_key is not None
    # Frozen dataclass + scalar keys ensure the first event is not enriched
    # retroactively with the session learned from a future event.
    assert first.session_key is None


def test_pfcp_seid_can_recover_existing_ue_through_session_alias():
    tracker = CausalCorrelationTracker()

    sbi = tracker.observe(
        supi="imsi-001010000000001",
        pdu_session_id="10",
        seid_c="1001",
    )
    assert sbi.ue_key is not None
    assert sbi.session_key is not None

    pfcp = tracker.observe(seid="1001")

    assert pfcp.ue_key == sbi.ue_key
    assert pfcp.session_key == sbi.session_key


def test_current_event_can_merge_previously_separate_aliases_without_backfill():
    tracker = CausalCorrelationTracker()

    ran = tracker.observe(ran_ngap_id="ran-1")
    supi = tracker.observe(supi="imsi-001")
    assert ran.ue_key != supi.ue_key

    merged = tracker.observe(
        ran_ngap_id="ran-1",
        supi="imsi-001",
    )

    assert merged.ue_key == ran.ue_key
    future = tracker.observe(supi="imsi-001")
    assert future.ue_key == ran.ue_key
    # Earlier immutable snapshots keep exactly what was known then.
    assert supi.ue_key != ran.ue_key
