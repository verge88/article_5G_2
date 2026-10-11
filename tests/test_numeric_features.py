import pytest

from fedsiggraph.data.features import CausalNumericFeatureEncoder
from fedsiggraph.data.schema import CanonicalEvent


def event(event_id: int, timestamp_ns: int) -> CanonicalEvent:
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
        relation="Nsmf",
        procedure="CreateSMContext",
        session_context="session-1",
        packet_length=128,
        label=0,
    )


def test_transform_uses_only_past_history():
    train = [event(1, 100_000_000), event(2, 200_000_000)]
    encoder = CausalNumericFeatureEncoder.fit(train)

    transformed = encoder.transform(
        [event(3, 300_000_000)],
        history=train,
    )

    assert transformed.shape == (1, 4)


def test_future_history_is_rejected():
    train = [event(1, 100_000_000), event(2, 200_000_000)]
    encoder = CausalNumericFeatureEncoder.fit(train)

    with pytest.raises(ValueError, match="later"):
        encoder.transform(
            [event(3, 150_000_000)],
            history=train,
        )
