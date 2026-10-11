from fedsiggraph.data.encoding import SemanticEncoder
from fedsiggraph.data.schema import CanonicalEvent
from fedsiggraph.data.split import split_benign_development


def event(i, ts, procedure, label=0):
    return CanonicalEvent(
        event_id=i,
        timestamp_ns=ts,
        deployment_id="site-a",
        src_entity="amf-1",
        dst_entity="smf-1",
        src_type="NF",
        dst_type="NF",
        src_role="AMF",
        dst_role="SMF",
        protocol="SBI",
        relation=procedure,
        procedure=procedure,
        packet_length=100 + i,
        label=label,
        attack_type=None if label == 0 else "attack",
    )


def test_encoder_maps_test_only_procedure_to_unk():
    train = [event(1, 100, "CreateSMContext"), event(2, 200, "UpdateSMContext")]
    encoder = SemanticEncoder.fit(train)

    unseen = event(3, 300, "FutureOnlyProcedure", label=1)

    assert encoder.procedure.encode("FutureOnlyProcedure") == encoder.procedure.unk_id
    assert (
        encoder.encode_categories(unseen)["procedure"]
        == encoder.procedure.unk_id
    )
    assert "FutureOnlyProcedure" not in encoder.procedure.token_to_id


def test_split_keeps_attack_capture_out_of_development_sets():
    benign = [event(i, i * 100, f"P{i}") for i in range(1, 11)]
    test = [
        event(20, 2_000, "Known", label=0),
        event(21, 2_100, "AttackOnly", label=1),
    ]

    splits = split_benign_development(
        benign,
        test,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    assert all(e.label == 0 for e in splits.train)
    assert all(e.label == 0 for e in splits.val)
    assert all(e.label == 0 for e in splits.calibration)
    assert [e.event_id for e in splits.test] == [20, 21]
    assert max(e.timestamp_ns for e in splits.train) <= min(
        e.timestamp_ns for e in splits.val
    )
