import pytest

from fedsiggraph.data.features import CausalNumericFeatureEncoder
from fedsiggraph.data.index import LocalNodeIndex
from fedsiggraph.data.pipeline import fit_preprocessing
from fedsiggraph.data.schema import CanonicalEvent
from fedsiggraph.data.vocab import SemanticVocabularies


def event(
    event_id: int,
    timestamp_ns: int,
    *,
    src: str = "amf-1",
    dst: str = "smf-1",
    src_role: str = "AMF",
    dst_role: str = "SMF",
    protocol: str = "SBI",
    procedure: str = "CreateSMContext",
    label: int = 0,
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        timestamp_ns=timestamp_ns,
        deployment_id="site-a",
        src_entity=src,
        dst_entity=dst,
        src_type="NF",
        dst_type="NF",
        src_role=src_role,
        dst_role=dst_role,
        protocol=protocol,
        relation=procedure,
        procedure=procedure,
        session_context="session-1",
        packet_length=100 + event_id,
        label=label,
    )


def test_test_only_semantics_map_to_unk_and_nodes_stay_local():
    benign = [
        event(1, 100),
        event(2, 200),
        event(3, 300),
        event(4, 400),
        event(5, 500),
        event(6, 600),
    ]
    test = [
        event(
            10,
            1000,
            src="amf-new",
            dst="udm-new",
            dst_role="UDM",
            procedure="NeverSeenProcedure",
            label=1,
        )
    ]

    fitted = fit_preprocessing(
        benign,
        test,
        train_fraction=0.5,
        val_fraction=0.25,
    )

    test_event = fitted.splits.test[0]
    encoded = fitted.vocab.encode_event(test_event)

    assert encoded["procedure_id"] == fitted.vocab.procedure.unk_id
    assert encoded["dst_role_id"] == fitted.vocab.dst_role.unk_id
    assert fitted.node_index.encode("amf-new") == fitted.node_index.unk_id
    assert fitted.node_index.encode("udm-new") == fitted.node_index.unk_id


def test_vocab_and_scaler_reject_attack_contamination():
    attack = event(99, 999, label=1)

    with pytest.raises(ValueError, match="benign"):
        SemanticVocabularies.fit([event(1, 100), attack])

    with pytest.raises(ValueError, match="benign"):
        CausalNumericFeatureEncoder.fit([event(1, 100), attack])


def test_local_node_index_is_fitted_only_on_given_events():
    index = LocalNodeIndex.fit([event(1, 100)])
    assert index.encode("amf-1") != index.unk_id
    assert index.encode("unseen-nf") == index.unk_id
