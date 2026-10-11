from fedsiggraph.data.schema import CanonicalEvent
from fedsiggraph.graph.hard_negatives import HardNegativeSampler


def event(
    event_id: int,
    src_role: str,
    dst_role: str,
    protocol: str,
    procedure: str,
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        timestamp_ns=event_id,
        deployment_id="site-a",
        src_entity=f"{src_role.lower()}-{event_id}",
        dst_entity=f"{dst_role.lower()}-{event_id}",
        src_type="NF",
        dst_type="NF",
        src_role=src_role,
        dst_role=dst_role,
        protocol=protocol,
        relation=procedure,
        procedure=procedure,
        label=0,
    )


def test_procedure_negative_is_protocol_compatible_but_pair_incompatible():
    training = [
        event(1, "AMF", "SMF", "SBI", "CreateSMContext"),
        event(2, "AMF", "UDM", "SBI", "GetSubscriberData"),
    ]
    sampler = HardNegativeSampler(training, seed=1)

    negative = sampler.procedure_negative(training[0])

    assert negative.protocol == "SBI"
    assert negative.src_role == "AMF"
    assert negative.dst_role == "SMF"
    assert negative.procedure == "GetSubscriberData"
    assert negative.label == 1


def test_endpoint_negative_changes_role_without_changing_procedure():
    training = [
        event(1, "AMF", "SMF", "SBI", "CreateSMContext"),
        event(2, "NRF", "UDM", "SBI", "CreateSMContext"),
    ]
    sampler = HardNegativeSampler(training, seed=1)

    negative = sampler.endpoint_negative(training[0])

    assert negative.procedure == "CreateSMContext"
    assert negative.dst_role == "UDM"
    # The negative reuses a genuine benign entity, so identity is not a shortcut.
    assert negative.dst_entity == training[1].dst_entity
    assert negative.label == 1
