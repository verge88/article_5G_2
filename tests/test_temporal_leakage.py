import pytest

from fedsiggraph.data.schema import CanonicalEvent
from fedsiggraph.graph.temporal_store import CausalTemporalStore


def make_event(event_id: int, timestamp_ns: int, procedure: str) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        timestamp_ns=timestamp_ns,
        deployment_id="site-a",
        src_entity="amf-1",
        dst_entity="smf-1",
        src_type="NF",
        dst_type="NF",
        src_role="AMF",
        dst_role="SMF",
        protocol="SBI",
        relation="Nsmf_PDUSession",
        procedure=procedure,
        session_context="session-7",
        label=0,
    )


def test_current_event_is_not_visible_before_observe():
    store = CausalTemporalStore(max_events_per_key=8)
    first = make_event(1, 100, "CreateSMContext")
    second = make_event(2, 200, "UpdateSMContext")

    assert store.context_before(first).all_events == ()

    store.observe(first)
    context = store.context_before(second)

    assert [event.event_id for event in context.all_events] == [1]
    assert second.event_id not in {
        event.event_id for event in context.all_events
    }

    store.observe(second)
    future = make_event(3, 300, "ReleaseSMContext")
    assert [
        event.event_id for event in store.context_before(future).all_events
    ] == [1, 2]


def test_out_of_order_observation_is_rejected():
    store = CausalTemporalStore()
    store.observe(make_event(1, 200, "CreateSMContext"))

    with pytest.raises(ValueError, match="non-decreasing"):
        store.observe(make_event(2, 100, "UpdateSMContext"))
