from fedsiggraph.data.schema import CanonicalEvent


def test_semantic_key_excludes_local_identity_and_labels():
    event = CanonicalEvent(
        event_id=1,
        timestamp_ns=10,
        deployment_id="site-a",
        src_entity="10.0.0.1",
        dst_entity="10.0.0.2",
        src_type="NF",
        dst_type="NF",
        src_role="AMF",
        dst_role="SMF",
        protocol="SBI",
        relation="Nsmf_PDUSession",
        procedure="CreateSMContext",
        label=1,
        attack_type="demo",
    )

    assert event.semantic_key() == (
        "AMF",
        "SMF",
        "SBI",
        "Nsmf_PDUSession",
        "CreateSMContext",
    )
    assert "10.0.0.1" not in event.semantic_key()
    assert "demo" not in event.semantic_key()
